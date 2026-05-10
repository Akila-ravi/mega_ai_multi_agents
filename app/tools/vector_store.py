"""FAISS + OpenAI embeddings loader for local RAG (LangChain)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from app.config import settings

log = logging.getLogger(__name__)

_lock = threading.Lock()
_store_unavailable: BaseException | None = None
_store: Any = None


def _index_present(root: Path) -> bool:
    return (root / "index.faiss").is_file() and (root / "index.pkl").is_file()


def load_faiss_vectorstore() -> Any | None:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    root = Path(settings.vectorstore_path).expanduser()
    if not root.is_dir() or not _index_present(root):
        log.info("faiss_skip_missing_index dir=%s", root)
        return None
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(str(root), embeddings, allow_dangerous_deserialization=True)


def get_faiss_store() -> Any | None:
    global _store, _store_unavailable
    with _lock:
        if _store_unavailable is not None:
            return None
        if _store is not None:
            return _store
        try:
            st = load_faiss_vectorstore()
            if st is None:
                return None
            _store = st
            return _store
        except BaseException as ex:
            _store_unavailable = ex
            log.warning("faiss_load_failed: %s", ex)
            return None


def similarity_chunks(query: str, k: int) -> list[tuple[Any, float]]:
    vs = get_faiss_store()
    if vs is None:
        return []
    kk = min(max(1, k), 32)
    return vs.similarity_search_with_relevance_scores(query, k=kk)
