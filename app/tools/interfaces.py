from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from io import StringIO
from typing import Any

from sqlalchemy import create_engine, text

from app.config import settings
from app.agents.llm import llm_enabled
from app.schemas import ToolFailure, ToolResult
from app.tools.nl2sql import is_safe_jobs_select, nl_question_to_sql


@dataclass
class ToolInput:
    payload: dict[str, Any]
    timeout_s: float = 2.0


def _malformed(tool_name: str, message: str) -> ToolResult:
    return ToolResult(ok=False, tool_name=tool_name, failure=ToolFailure(mode="malformed", message=message))


def _timeout(tool_name: str) -> ToolResult:
    return ToolResult(ok=False, tool_name=tool_name, failure=ToolFailure(mode="timeout", message="Tool call timed out"))


def _empty(tool_name: str) -> ToolResult:
    return ToolResult(ok=False, tool_name=tool_name, failure=ToolFailure(mode="empty", message="No results"))


async def web_search_stub(tool_input: ToolInput) -> ToolResult:
    tool_name = "web_search"
    query = tool_input.payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return _malformed(tool_name, "query must be non-empty string")
    try:
        await asyncio.wait_for(asyncio.sleep(0.05), timeout=tool_input.timeout_s)
    except asyncio.TimeoutError:
        return _timeout(tool_name)
    results = [
        {"url": "https://example.com/a", "title": f"{query} overview", "relevance": 0.91, "chunk_id": "c1", "content": f"Fact 1 about {query}"},
        {"url": "https://example.com/b", "title": f"{query} deep-dive", "relevance": 0.87, "chunk_id": "c2", "content": f"Fact 2 about {query}"},
    ]
    if not results:
        return _empty(tool_name)
    return ToolResult(ok=True, tool_name=tool_name, data={"results": results})


async def python_sandbox(tool_input: ToolInput) -> ToolResult:
    tool_name = "python_sandbox"
    code = tool_input.payload.get("code")
    if not isinstance(code, str):
        return _malformed(tool_name, "code must be string")
    stdout, stderr = StringIO(), StringIO()
    g = {"__builtins__": {"print": print, "range": range, "len": len, "sum": sum}}
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout, stderr
    exit_code = 0
    try:
        await asyncio.wait_for(asyncio.to_thread(exec, code, g, {}), timeout=tool_input.timeout_s)
    except asyncio.TimeoutError:
        return _timeout(tool_name)
    except Exception as ex:
        exit_code = 1
        stderr.write(str(ex))
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return ToolResult(ok=exit_code == 0, tool_name=tool_name, data={"stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "exit_code": exit_code})


def _execute_jobs_sql(sql: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        engine = create_engine(settings.sync_database_url)
        with engine.connect() as conn:
            proxy = conn.execute(text(sql))
            rows = [dict(r) for r in proxy.mappings().all()]
            return rows[:100], None
    except Exception as ex:
        return [], str(ex)


async def sql_lookup(tool_input: ToolInput) -> ToolResult:
    tool_name = "sql_lookup"
    nl_query = tool_input.payload.get("question")
    if not isinstance(nl_query, str) or not nl_query.strip():
        return _malformed(tool_name, "question must be non-empty string")

    mapping = {
        "count jobs": "SELECT COUNT(*) AS count FROM jobs",
        "list failed jobs": "SELECT id, status FROM jobs WHERE status='failed' LIMIT 10",
    }
    sql = ""
    if llm_enabled():
        sql = await nl_question_to_sql(nl_query)
    if not sql:
        sql = mapping.get(nl_query.lower().strip(), "")
    if not sql.strip():
        return _empty(tool_name)
    sql = sql.strip().rstrip(";").strip()
    if not is_safe_jobs_select(sql):
        return _malformed(tool_name, "generated SQL rejected by safety validator")
    try:
        rows, err = await asyncio.wait_for(asyncio.to_thread(_execute_jobs_sql, sql), timeout=tool_input.timeout_s)
    except asyncio.TimeoutError:
        return _timeout(tool_name)
    if err:
        return ToolResult(ok=False, tool_name=tool_name, data={"sql": sql, "rows": [], "error": err}, failure=ToolFailure(mode="malformed", message=err[:200]))
    return ToolResult(ok=True, tool_name=tool_name, data={"sql": sql, "rows": rows})


async def self_reflection(tool_input: ToolInput) -> ToolResult:
    tool_name = "self_reflection"
    outputs = tool_input.payload.get("outputs")
    if not isinstance(outputs, list):
        return _malformed(tool_name, "outputs must be list")
    contradictions = []
    for i, txt in enumerate(outputs):
        if isinstance(txt, str) and "not" in txt.lower() and "is" in txt.lower():
            contradictions.append({"index": i, "reason": "Potential self-contradiction marker"})
    if not outputs:
        return _empty(tool_name)
    return ToolResult(ok=True, tool_name=tool_name, data={"contradictions": contradictions})
