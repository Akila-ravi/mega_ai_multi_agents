"""
Build persistable LangChain vectors under ./vectorstore (FAISS + OpenAI embeddings).

Requires OPENAI_API_KEY and network access once. Run from repo root:

  python scripts/build_faiss_vectorstore.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env")

    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings

    from app.rag.seed import DEFAULT_CORPUS_ROWS

    out = _REPO / "vectorstore"
    out.mkdir(parents=True, exist_ok=True)
    docs = [
        Document(
            page_content=row["body"],
            metadata={"chunk_id": row["external_id"], "title": row["title"], "url": row["url"]},
        )
        for row in DEFAULT_CORPUS_ROWS
    ]
    vs = FAISS.from_documents(docs, OpenAIEmbeddings())
    vs.save_local(str(out))


if __name__ == "__main__":
    main()
