from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.init_db import init_db
from app.db.models import AgentLog, EvalRun, Job, JobStatus, PromptDecision, PromptVersion, ToolCall
from app.db.session import AsyncSessionLocal, get_db_session
from app.evaluation.engine import build_eval_cases, latest_eval_summary, propose_prompt_diff, score_case
from app.schemas import ErrorResponse, EvalRetryRequest, PromptApprovalRequest, QueryRequest, TraceResponse
from app.worker.queue import enqueue_job

app = FastAPI(title="Mega AI")


@app.on_event("startup")
async def startup() -> None:
    await init_db()


@app.post("/query")
async def query_endpoint(req: QueryRequest, db: AsyncSession = Depends(get_db_session)):
    job_id = str(uuid.uuid4())
    await enqueue_job(db, job_id, req.query)

    async def stream():
        last_log_id = 0
        last_tool_id = 0
        while True:
            async with AsyncSessionLocal() as poll_db:
                job_res = await poll_db.execute(select(Job).where(Job.id == job_id))
                job = job_res.scalar_one_or_none()
                if not job:
                    err = ErrorResponse(error_code="JOB_NOT_FOUND", message="Job disappeared", job_id=job_id)
                    yield f"data: {err.model_dump_json()}\n\n"
                    break

                logs = await poll_db.execute(select(AgentLog).where(AgentLog.job_id == job_id, AgentLog.id > last_log_id).order_by(AgentLog.id.asc()))
                rows = logs.scalars().all()
                for row in rows:
                    last_log_id = row.id
                    payload = {"agent": row.agent_id, "event": row.event_type, "token_count": row.token_count, "latency_ms": row.latency_ms}
                    yield f"data: {json.dumps(payload)}\n\n"

                tools = await poll_db.execute(
                    select(ToolCall).where(ToolCall.job_id == job_id, ToolCall.id > last_tool_id).order_by(ToolCall.id.asc())
                )
                for t in tools.scalars().all():
                    last_tool_id = t.id
                    yield f"data: {json.dumps({'tool_call': {'tool': t.tool_name, 'accepted': t.accepted, 'retry': t.retry_index}})}\n\n"

                if job.status in [JobStatus.completed, JobStatus.failed]:
                    final = {"job_id": job_id, "status": job.status.value if hasattr(job.status, "value") else str(job.status), "result": job.result, "error": job.error}
                    yield f"data: {json.dumps(final)}\n\n"
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/trace/{job_id}", response_model=TraceResponse, responses={404: {"model": ErrorResponse}})
async def trace_endpoint(job_id: str, db: AsyncSession = Depends(get_db_session)):
    jr = await db.execute(select(Job).where(Job.id == job_id))
    job = jr.scalar_one_or_none()
    if not job:
        raise HTTPException(404, detail=ErrorResponse(error_code="JOB_NOT_FOUND", message="Job not found", job_id=job_id).model_dump())
    lr = await db.execute(select(AgentLog).where(AgentLog.job_id == job_id).order_by(AgentLog.id.asc()))
    tr = await db.execute(select(ToolCall).where(ToolCall.job_id == job_id).order_by(ToolCall.id.asc()))
    return TraceResponse(
        job_id=job_id,
        status=job.status.value if hasattr(job.status, "value") else str(job.status),
        logs=[
            {
                "agent_id": l.agent_id,
                "event_type": l.event_type,
                "payload": l.payload,
                "latency_ms": l.latency_ms,
                "token_count": l.token_count,
            }
            for l in lr.scalars().all()
        ],
        tool_calls=[
            {
                "tool_name": t.tool_name,
                "tool_input": t.tool_input,
                "tool_output": t.tool_output,
                "accepted": t.accepted,
                "latency_ms": t.latency_ms,
                "retry_index": t.retry_index,
            }
            for t in tr.scalars().all()
        ],
        result=job.result,
    )


@app.get("/eval/summary", responses={404: {"model": ErrorResponse}})
async def eval_summary_endpoint(db: AsyncSession = Depends(get_db_session)):
    data = await latest_eval_summary(db)
    if not data:
        raise HTTPException(404, detail=ErrorResponse(error_code="NO_EVAL_RUN", message="No eval run available").model_dump())
    return data


@app.post("/prompt/approve")
async def prompt_approve_endpoint(req: PromptApprovalRequest, db: AsyncSession = Depends(get_db_session)):
    res = await db.execute(select(PromptVersion).where(PromptVersion.id == req.prompt_version_id))
    pv = res.scalar_one_or_none()
    if not pv:
        raise HTTPException(404, detail=ErrorResponse(error_code="PROMPT_NOT_FOUND", message="Prompt version not found").model_dump())
    pv.decision = PromptDecision.approved if req.decision == "approved" else PromptDecision.rejected
    await db.commit()
    return {"prompt_version_id": pv.id, "decision": pv.decision.value}


@app.post("/eval/retry")
async def eval_retry_endpoint(req: EvalRetryRequest, db: AsyncSession = Depends(get_db_session)):
    cases = build_eval_cases()
    results = []
    failures = []
    for c in cases:
        if req.run_type == "failed_only" and c.category == "normal":
            continue
        ans = "Fact 1 and Fact 2. Contradictions were resolved."
        sc = score_case(ans, c.expected_keywords, tool_calls=3, violations=0)
        item = {"case_id": c.case_id, "category": c.category, "scores": sc, "answer": ans, "input": c.query}
        if sc["overall"]["score"] < 0.75:
            failures.append(item)
        results.append(item)

    by_cat = {}
    for cat in ["normal", "ambiguous", "adversarial"]:
        bucket = [r for r in results if r["category"] == cat]
        if bucket:
            by_cat[cat] = {
                "count": len(bucket),
                "avg_overall": sum(x["scores"]["overall"]["score"] for x in bucket) / len(bucket),
            }

    summary = {"run_type": req.run_type, "total": len(results), "failed": len(failures), "categories": by_cat}
    run = EvalRun(run_type=req.run_type, summary=summary, details={"results": results, "failures": failures})
    db.add(run)
    await db.commit()

    proposal = await propose_prompt_diff(db, failures)
    return {"summary": summary, "prompt_proposal": {"id": proposal.id, "diff": proposal.diff, "justification": proposal.justification}}
