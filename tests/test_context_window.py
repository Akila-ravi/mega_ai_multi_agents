from app.context_manager.budget_manager import ContextBudgetManager
from app.context_manager.context_window import ContextWindow
from app.schemas import SharedContext


def test_context_window_records_rag_snippets():
    ctx = SharedContext(job_id="j", user_query="q")
    mgr = ContextBudgetManager(ctx)
    window = ContextWindow(ctx, mgr)
    hits = [
        {
            "chunk_id": "c1",
            "title": "t",
            "url": "u",
            "content": "sentence one " + ("x" * 800),
            "relevance": 0.42,
            "source": "corpus",
        }
    ]
    window.record_rag_hits(hits, snippet_chars=120)
    assert ctx.rag_hits and ctx.rag_hits[0]["chunk_id"] == "c1"
    assert isinstance(ctx.messages[-1].get("snippet"), str)
    assert len(ctx.messages[-1]["snippet"]) <= 120


def test_context_window_trim_snippets():
    ctx = SharedContext(job_id="j", user_query="q")
    mgr = ContextBudgetManager(ctx)
    window = ContextWindow(ctx, mgr)
    ctx.messages = [{"kind": "citation", "snippet": "a" * 400}] * 30
    window.trim_snippets(max_messages=4, max_snippet_chars=50)
    assert len(ctx.messages) == 4
    assert len(ctx.messages[0]["snippet"]) <= 50
