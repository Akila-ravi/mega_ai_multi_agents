from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Job, JobStatus


async def enqueue_job(db: AsyncSession, job_id: str, query: str) -> Job:
    job = Job(id=job_id, query=query, status=JobStatus.queued, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.add(job)
    await db.commit()
    return job


async def fetch_next_queued_job(db: AsyncSession) -> Job | None:
    res = await db.execute(select(Job).where(Job.status == JobStatus.queued).order_by(Job.created_at.asc()).limit(1))
    job = res.scalar_one_or_none()
    if job:
        job.status = JobStatus.running
        job.updated_at = datetime.utcnow()
        await db.commit()
    return job
