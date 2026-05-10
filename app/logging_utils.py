import hashlib
import json
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentLog


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def log_event(
    db: AsyncSession,
    *,
    job_id: str,
    agent_id: str,
    event_type: str,
    payload: dict[str, Any],
    input_obj: Any,
    output_obj: Any,
    latency_ms: float,
    token_count: int,
    policy_violation: bool = False,
) -> None:
    row = AgentLog(
        job_id=job_id,
        agent_id=agent_id,
        event_type=event_type,
        payload=payload,
        input_hash=_stable_hash(input_obj),
        output_hash=_stable_hash(output_obj),
        latency_ms=latency_ms,
        token_count=token_count,
        policy_violation=policy_violation,
    )
    db.add(row)
    await db.commit()


class Stopwatch:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000
