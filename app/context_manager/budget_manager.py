from __future__ import annotations

from typing import Literal

from app.schemas import SharedContext

ConsumeMode = Literal["strict", "truncate"]


class ContextBudgetManager:
    def __init__(self, context: SharedContext) -> None:
        self.context = context

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def declare_budget(self, agent_id: str, max_tokens: int) -> None:
        self.context.context_budget[agent_id] = max_tokens
        self.context.token_usage.setdefault(agent_id, 0)

    def remaining(self, agent_id: str) -> int:
        budget = self.context.context_budget.get(agent_id, 0)
        used = self.context.token_usage.get(agent_id, 0)
        return budget - used

    def consume(self, agent_id: str, text: str, *, mode: ConsumeMode = "truncate") -> str:
        """
        Charge token usage against the agent budget.

        strict: unchanged legacy behavior — exceed → policy violation + ValueError.

        truncate: fit into remaining budget by trimming text (never increases usage above budget).

        Returns the substring of `text` that was billed (truncate mode may shorten input).
        """
        budget = self.context.context_budget.get(agent_id, 0)
        used = self.context.token_usage.get(agent_id, 0)
        need = self.estimate_tokens(text)

        if used + need <= budget:
            self.context.token_usage[agent_id] = used + need
            return text

        remaining_tokens = budget - used
        if mode == "strict" or remaining_tokens <= 0:
            self.context.policy_violations.append(
                f"{agent_id} exceeded budget by {used + need - budget} (need={need}, budget={budget})"
            )
            raise ValueError(f"{agent_id} exceeded context budget")

        allowed_chars = max(4, remaining_tokens * 4)
        trimmed = text[:allowed_chars].rstrip()
        if not trimmed:
            trimmed = text[:4]
        billed = self.estimate_tokens(trimmed)
        self.context.token_usage[agent_id] = used + billed
        self.context.policy_violations.append(f"{agent_id} output truncated to respect context budget ({billed}/{need} tokens)")
        return trimmed

    def compress_if_needed(self, agent_id: str) -> None:
        """Best-effort: shrink recent message snippets when over budget (non-destructive to tool payloads)."""
        if self.remaining(agent_id) >= 0:
            return
        trimmed: list[dict] = []
        for msg in self.context.messages[-12:]:
            if isinstance(msg, dict) and msg.get("kind") in {"tool", "score", "citation"}:
                sn = msg.get("snippet")
                if isinstance(sn, str) and len(sn) > 120:
                    trimmed.append({**msg, "snippet": sn[:120]})
                else:
                    trimmed.append(msg)
            else:
                trimmed.append({"kind": "summary", "text": str(msg)[:80]})
        self.context.messages = trimmed
        # Reconcile usage with declared budget to avoid stuck negative state after compression
        self.context.token_usage[agent_id] = min(self.context.token_usage.get(agent_id, 0), self.context.context_budget.get(agent_id, 0))
