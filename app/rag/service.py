from __future__ import annotations

import asyncio
import math
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import CorpusChunk
from app.rag.faiss_retrieval import afetch_faiss_chunks
from app.rag.scoring import rank_documents


def _normalize_scores(raw: list[float]) -> list[float]:
    if not raw:
        return []
    m = max(raw) or 1e-9
    return [min(1.0, max(0.0, s / m)) for s in raw]


async def fetch_rag_chunks(session: AsyncSession, query: str, top_k: int = 8) -> list[dict[str, Any]]:
    if settings.vectorstore_path.strip():
        k = min(max(1, top_k), max(1, settings.rag_faiss_k))
        faiss_hits = await afetch_faiss_chunks(query, k=k)
        if faiss_hits:
            return faiss_hits

    rows = (await session.execute(select(CorpusChunk).order_by(CorpusChunk.id.asc()))).scalars().all()
    if not rows:
        return []
    bodies = [r.body for r in rows]
    scores = rank_documents(query, bodies)
    norm = _normalize_scores(scores)
    ranked: list[tuple[float, float, CorpusChunk]] = sorted(zip(scores, norm, rows, strict=True), key=lambda t: t[0], reverse=True)
    out: list[dict[str, Any]] = []
    for raw, nscore, row in ranked[: max(1, top_k)]:
        out.append(
            {
                "chunk_id": row.external_id,
                "url": row.url or f"corpus://{row.external_id}",
                "title": row.title,
                "content": row.body,
                "relevance": round(float(nscore), 4),
                "rag_score": round(float(raw), 4),
                "source": "corpus",
            }
        )
    return out


def merge_retrieval_chunks(rag: list[dict[str, Any]], web: list[dict[str, Any]], cap: int = 6) -> list[dict[str, Any]]:
    """Merge heterogeneous chunk dicts; prefer higher relevance, break ties with corpus first."""
    merged = [*rag, *web]

    def sort_key(c: dict[str, Any]) -> tuple[float, float, str]:
        rel = float(c.get("relevance", 0))
        src_boost = 1.0 if c.get("source") in {"corpus", "faiss"} else 0.0
        return (rel, src_boost, c.get("chunk_id", ""))

    merged.sort(key=sort_key, reverse=True)
    # Jitter tiny dedupe on chunk_id preserving best score
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for c in merged:
        cid = str(c.get("chunk_id", ""))
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(c)
        if len(deduped) >= cap:
            break
    # Re-normalize relevance faintly for downstream agents
    rels = [float(c.get("relevance", 0)) for c in deduped]
    m = max(rels) if rels else 1.0
    m = m or 1.0
    for c in deduped:
        c["relevance"] = round(float(c.get("relevance", 0)) / m, 4) if not math.isnan(m) else 0.0
    return deduped
