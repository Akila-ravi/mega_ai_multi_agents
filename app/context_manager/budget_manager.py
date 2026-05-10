from app.schemas import SharedContext


class ContextBudgetManager:
    def __init__(self, context: SharedContext) -> None:
        self.context = context

    def declare_budget(self, agent_id: str, max_tokens: int) -> None:
        self.context.context_budget[agent_id] = max_tokens
        self.context.token_usage.setdefault(agent_id, 0)

    def remaining(self, agent_id: str) -> int:
        budget = self.context.context_budget.get(agent_id, 0)
        used = self.context.token_usage.get(agent_id, 0)
        return budget - used

    def consume(self, agent_id: str, text: str) -> None:
        approx_tokens = max(1, len(text) // 4)
        self.context.token_usage[agent_id] = self.context.token_usage.get(agent_id, 0) + approx_tokens
        if self.context.token_usage[agent_id] > self.context.context_budget.get(agent_id, 0):
            violation = f"{agent_id} exceeded budget by {self.context.token_usage[agent_id] - self.context.context_budget.get(agent_id, 0)}"
            self.context.policy_violations.append(violation)
            raise ValueError(violation)

    def compress_if_needed(self, agent_id: str) -> None:
        if self.remaining(agent_id) >= 0:
            return
        # lossless for structured fields, lossy for conversational filler
        trimmed = []
        for msg in self.context.messages[-10:]:
            if isinstance(msg, dict) and msg.get("kind") in {"tool", "score", "citation"}:
                trimmed.append(msg)
            else:
                trimmed.append({"kind": "summary", "text": str(msg)[:80]})
        self.context.messages = trimmed
        self.context.token_usage[agent_id] = max(0, self.context.context_budget.get(agent_id, 0) - 5)
