from app.rag.scoring import rank_documents
from app.rag.service import merge_retrieval_chunks


def test_bm25_prioritizes_relevant_document():
    docs = ["zebra zebra zebra", "kubernetes scheduler assigns pods using filters and scoring", "offline cooking recipes"]
    q = "How does kubernetes scheduling assign pods?"
    scores = rank_documents(q, docs)
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_merge_dedupes_chunk_ids_prefers_best_score():
    rag = [
        {"chunk_id": "a", "content": "corp", "relevance": 0.9, "url": "", "title": "", "source": "corpus"},
        {"chunk_id": "b", "content": "corp2", "relevance": 0.8, "url": "", "title": "", "source": "corpus"},
    ]
    web = [{"chunk_id": "a", "content": "web dup", "relevance": 0.4, "url": "", "title": "", "source": "web_stub"}]
    merged = merge_retrieval_chunks(rag, web, cap=4)
    ids = [m["chunk_id"] for m in merged]
    assert ids[0] == "a"
