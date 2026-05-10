from __future__ import annotations

import re

from app.agents.llm import chat_completion, llm_enabled

_JOBS_SELECT_RE = re.compile(r"^select\b[\s\S]*\bfrom\s+jobs\b", re.IGNORECASE | re.DOTALL)


def is_safe_jobs_select(sql: str) -> bool:
    s = (sql or "").strip()
    if not s.lstrip().lower().startswith("select"):
        return False
    if ";" in s:
        return False
    lower = s.lower()
    banned = ["insert ", "update ", "delete ", "drop ", "alter ", "truncate ", "grant ", "revoke "]
    if any(b in lower for b in banned):
        return False
    if "--" in s or "/*" in s:
        return False
    return bool(_JOBS_SELECT_RE.match(s))


async def nl_question_to_sql(question: str) -> str:
    if not llm_enabled():
        return ""
    system = (
        "You convert natural language questions into a SINGLE PostgreSQL SELECT statement. "
        "Only query the jobs table columns: id, query, status, created_at, updated_at, result, error.\n"
        "Rules: SELECT only, no semicolons, no comments.\n"
        "Respond with ONLY the SQL (you may wrap lines sparingly)."
    )
    user = f'Question: """{question}"""'
    raw = await chat_completion(system=system, user=user, temperature=0.0, max_tokens=220)
    # Take first plausible line/block without semicolon
    line = raw.strip().splitlines()[0].strip().rstrip(";").strip()
    return line
