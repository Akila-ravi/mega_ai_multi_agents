import asyncio

from app.config import settings
from app.db.init_db import init_db
from app.db.models import JobStatus
from app.db.session import AsyncSessionLocal
from app.orchestrator.orchestrator import Orchestrator
from app.worker.queue import fetch_next_queued_job


async def run_worker() -> None:
    await init_db()
    orch = Orchestrator()
    while True:
        async with AsyncSessionLocal() as db:
            job = await fetch_next_queued_job(db)
            if job:
                try:
                    await orch.run_job(db, job.query, job_id=job.id)
                except Exception as ex:
                    job.status = JobStatus.failed
                    job.error = str(ex)
                    await db.commit()
        await asyncio.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    asyncio.run(run_worker())
