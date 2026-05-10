from __future__ import annotations

from typing import Any

from app.context_manager.budget_manager import ContextBudgetManager
from app.schemas import ContextEvent, SharedContext


class ContextWindow:
    """Accumulates retrieval-friendly messages and exposes light-touch compression hooks."""

    def __init__(self, context: SharedContext, budget: ContextBudgetManager) -> None:
        self.context = context
        self.budget = budget

    def estimated_total_tokens(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for aid, used in self.context.token_usage.items():
            totals[aid] = totals.get(aid, 0) + used
        return totals

    def record_routing(self, payload: dict[str, Any]) -> None:
        self.context.events.append(ContextEvent(event_type="routing_ack", actor="context_window", payload=payload))

    def record_rag_hits(self, hits: list[dict[str, Any]], *, snippet_chars: int = 360) -> None:
        """Push RAG snippets into conversational memory for critique/budget tooling."""
        self.context.rag_hits = list(hits)
        for h in hits:
            body = str(h.get("content", ""))
            self.context.messages.append(
                {
                    "kind": "citation",
                    "source": "rag",
                    "chunk_id": h.get("chunk_id"),
                    "title": h.get("title"),
                    "url": h.get("url"),
                    "snippet": body[:snippet_chars],
                    "relevance": h.get("relevance"),
                }
            )

    def trim_snippets(self, max_messages: int = 24, max_snippet_chars: int = 220) -> None:
        msgs = self.context.messages[-max_messages:]
        trimmed: list[dict[str, Any]] = []
        for m in msgs:
            if isinstance(m, dict) and isinstance(m.get("snippet"), str):
                trimmed.append({**m, "snippet": m["snippet"][:max_snippet_chars]})
            else:
                trimmed.append(m if isinstance(m, dict) else {"kind": "summary", "text": str(m)[:max_snippet_chars]})
        self.context.messages = trimmed
