from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CorpusChunk

# Small in-repo knowledge base (no network). Tuned for local demos and tests.
DEFAULT_CORPUS_ROWS: list[dict[str, str]] = [
    {
        "external_id": "cap-001",
        "title": "CAP theorem (local note)",
        "url": "internal://docs/cap",
        "body": "The CAP theorem states that a distributed datastore can provide at most two of consistency, availability, and partition tolerance at once. "
        "Practical systems often choose AP with eventual consistency plus conflict resolution, or CP with reduced availability during partitions.",
    },
    {
        "external_id": "ec-001",
        "title": "Eventual consistency",
        "url": "internal://docs/eventual-consistency",
        "body": "Eventual consistency means replicas converge over time if updates stop. Reads may be stale briefly. "
        "Vector clocks, last-write-wins, and CRDTs help resolve conflicts. Typical in DNS, CDNs, and many NoSQL databases.",
    },
    {
        "external_id": "k8s-001",
        "title": "Kubernetes scheduling overview",
        "url": "internal://docs/kubernetes-scheduling",
        "body": "The Kubernetes scheduler assigns pods to nodes using filters and scoring: resource requests, affinity, taints, topology spread, and priority. "
        "It repeats until a feasible node is chosen; preemption may evict lower-priority workloads when needed.",
    },
    {
        "external_id": "docker-001",
        "title": "Docker networking primer",
        "url": "internal://docs/docker-networking",
        "body": "Docker provides bridge, host, overlay, and macvlan drivers. Bridge is default for single-host dev. "
        "Overlay networks connect multi-host clusters; DNS-based service discovery resolves container names.",
    },
    {
        "external_id": "sql-001",
        "title": "Postgres transactions",
        "url": "internal://docs/postgres-tx",
        "body": "PostgreSQL uses MVCC for concurrency. READ COMMITTED is default; SERIALIZABLE prevents anomalies at higher cost. "
        "Indexes accelerate lookups; EXPLAIN ANALYZE helps tune query plans.",
    },
]


async def ingest_docs_txt(session: AsyncSession) -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "docs"
    if not root.is_dir():
        return
    for path in sorted(root.glob("*.txt")):
        external_id = f"doc-{path.stem.lower()}"
        exists = await session.execute(select(CorpusChunk).where(CorpusChunk.external_id == external_id))
        if exists.scalar_one_or_none():
            continue
        body = path.read_text(encoding="utf-8")
        session.add(
            CorpusChunk(
                external_id=external_id,
                title=path.stem.replace("-", " ").title(),
                url=f"file://docs/{path.name}",
                body=body,
            )
        )
    await session.commit()


async def seed_corpus_if_empty(session: AsyncSession) -> None:
    res = await session.execute(select(func.count()).select_from(CorpusChunk))
    n = int(res.scalar_one() or 0)
    if n > 0:
        return
    for row in DEFAULT_CORPUS_ROWS:
        session.add(
            CorpusChunk(
                external_id=row["external_id"],
                title=row["title"],
                url=row["url"],
                body=row["body"],
            )
        )
    await session.commit()
