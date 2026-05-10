from __future__ import annotations

import asyncio
from typing import Any

from app.config import settings
from app.tools.vector_store import similarity_chunks


def faiss_similarity_chunks(query: str, k: int | None = None) -> list[dict[str, Any]]:
    if not settings.vectorstore_path.strip():
        return []
    kk = k or settings.rag_faiss_k
    pairs = similarity_chunks(query, kk)
    out: list[dict[str, Any]] = []
    for i, (doc, rel) in enumerate(pairs):
        md = dict(doc.metadata or {})
        cid = md.get("chunk_id")
        if not cid:
            cid = f"c{i}"
        out.append(
            {
                "chunk_id": str(cid),
                "content": doc.page_content,
                "title": str(md.get("title", "") or ""),
                "url": str(md.get("url", "") or ""),
                "relevance": round(float(rel), 4),
                "source": "faiss",
                "rag_score": round(float(rel), 4),
            }
        )
    return out


async def afetch_faiss_chunks(query: str, k: int | None = None) -> list[dict[str, Any]]:
    return await asyncio.to_thread(faiss_similarity_chunks, query, k)
