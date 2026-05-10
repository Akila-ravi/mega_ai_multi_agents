from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.agents import CritiqueAgent, DecompositionAgent, RetrievalAgent, SynthesisAgent
from app.config import settings
from app.context_manager.budget_manager import ContextBudgetManager
from app.db.models import Job, JobStatus, ToolCall
from app.logging_utils import Stopwatch, log_event
from app.schemas import ContextEvent, SharedContext, ToolResult
from app.tools.interfaces import ToolInput
from app.tools.registry import TOOL_REGISTRY


class Orchestrator:
    def __init__(self) -> None:
        self.decomposition = DecompositionAgent()
        self.retrieval = RetrievalAgent()
        self.critique = CritiqueAgent()
        self.synthesis = SynthesisAgent()

    async def _call_tool(self, db: AsyncSession, context: SharedContext, agent_id: str, tool_name: str, payload: dict[str, Any]) -> ToolResult:
        retries = settings.max_tool_retries
        last: ToolResult | None = None
        for retry_idx in range(retries + 1):
            sw = Stopwatch()
            tool = TOOL_REGISTRY[tool_name]
            result = await tool(ToolInput(payload=payload, timeout_s=2.0))
            accepted = result.ok
            last = result
            db.add(
                ToolCall(
                    job_id=context.job_id,
                    agent_id=agent_id,
                    tool_name=tool_name,
                    tool_input=payload,
                    tool_output=result.model_dump(),
                    latency_ms=sw.elapsed_ms(),
                    accepted=accepted,
                    retry_index=retry_idx,
                )
            )
            await db.commit()
            context.tool_results.append(result.model_dump())
            if accepted:
                return result
            if result.failure and result.failure.mode == "malformed":
                payload = {"query": str(payload)}
            elif result.failure and result.failure.mode == "empty":
                payload = {"query": payload.get("query", "") + " overview"}
            elif result.failure and result.failure.mode == "timeout":
                payload = dict(payload)
        return last if last else ToolResult(ok=False, tool_name=tool_name, data={})

    async def run_job(self, db: AsyncSession, query: str, job_id: str | None = None) -> dict[str, Any]:
        job_id = job_id or str(uuid.uuid4())
        existing = await db.get(Job, job_id)
        if existing:
            job = existing
            job.status = JobStatus.running
            job.updated_at = datetime.utcnow()
            await db.commit()
        else:
            job = Job(id=job_id, query=query, status=JobStatus.running, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
            db.add(job)
            await db.commit()

        context = SharedContext(job_id=job_id, user_query=query)
        budget = ContextBudgetManager(context)

        # dynamic routing decision with justification
        route = ["decomposition_agent", "retrieval_agent", "critique_agent", "synthesis_agent"]
        if "calculate" in query.lower() or "python" in query.lower():
            route.insert(2, "python_tool_check")

        context.events.append(ContextEvent(event_type="routing", actor="orchestrator", payload={"route": route, "justification": "Query intent-based runtime routing"}))

        budget.declare_budget(self.decomposition.agent_id, self.decomposition.max_budget)
        out = await self.decomposition.run(context)
        context.agent_outputs[out.agent_id] = out
        budget.consume(self.decomposition.agent_id, out.content)
        await log_event(db, job_id=job_id, agent_id="orchestrator", event_type="decomposition_done", payload=out.model_dump(), input_obj=query, output_obj=out.model_dump(), latency_ms=1, token_count=len(out.content)//4)

        budget.declare_budget(self.retrieval.agent_id, self.retrieval.max_budget)
        search = await self._call_tool(db, context, self.retrieval.agent_id, "web_search", {"query": query})
        chunks = search.data.get("results", []) if search.ok else []
        ret = await self.retrieval.run(context, chunks)
        context.agent_outputs[ret.agent_id] = ret
        budget.consume(self.retrieval.agent_id, ret.content)
        await log_event(db, job_id=job_id, agent_id="orchestrator", event_type="retrieval_done", payload=ret.model_dump(), input_obj=query, output_obj=ret.model_dump(), latency_ms=1, token_count=len(ret.content)//4)

        if "python_tool_check" in route:
            py = await self._call_tool(db, context, "orchestrator", "python_sandbox", {"code": "print(2+2)"})
            context.events.append(ContextEvent(event_type="tool_check", actor="orchestrator", payload=py.model_dump()))

        budget.declare_budget(self.critique.agent_id, self.critique.max_budget)
        cr = await self.critique.run(context)
        context.agent_outputs[cr.agent_id] = cr
        budget.consume(self.critique.agent_id, cr.content)

        ref = await self._call_tool(db, context, self.critique.agent_id, "self_reflection", {"outputs": [v.content for v in context.agent_outputs.values()]})
        context.events.append(ContextEvent(event_type="self_reflection", actor=self.critique.agent_id, payload=ref.model_dump()))

        budget.declare_budget(self.synthesis.agent_id, self.synthesis.max_budget)
        sy = await self.synthesis.run(context)
        context.agent_outputs[sy.agent_id] = sy
        budget.consume(self.synthesis.agent_id, sy.content)

        final = {
            "job_id": job_id,
            "answer": sy.content,
            "provenance": [c.model_dump() for c in sy.citations],
            "policy_violations": context.policy_violations,
            "events": [e.model_dump(mode="json") for e in context.events],
        }

        job.status = JobStatus.completed
        job.updated_at = datetime.utcnow()
        job.result = final
        await db.commit()
        return final
