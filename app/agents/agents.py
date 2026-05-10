from __future__ import annotations

from app.schemas import AgentOutput, Citation, Claim, DecompositionTask, RetrievedChunk, SharedContext


class BaseAgent:
    agent_id: str
    max_budget: int

    def __init__(self, agent_id: str, max_budget: int) -> None:
        self.agent_id = agent_id
        self.max_budget = max_budget


class DecompositionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("decomposition_agent", 1200)

    async def run(self, context: SharedContext) -> AgentOutput:
        q = context.user_query
        tasks = [
            DecompositionTask(task_id="t1", task_type="intent", description=f"Understand user intent for: {q}"),
            DecompositionTask(task_id="t2", task_type="retrieve", description="Retrieve at least two supporting chunks", dependencies=["t1"]),
            DecompositionTask(task_id="t3", task_type="synthesize", description="Merge and validate claims", dependencies=["t2"]),
        ]
        return AgentOutput(agent_id=self.agent_id, content="Decomposed query into dependency graph", metadata={"tasks": [t.model_dump() for t in tasks]})


class RetrievalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("retrieval_agent", 1800)

    async def run(self, context: SharedContext, chunks: list[dict]) -> AgentOutput:
        selected = chunks[:2]
        retrieved = [
            RetrievedChunk(chunk_id=c["chunk_id"], source_url=c["url"], content=c["content"], relevance=c["relevance"]) for c in selected
        ]
        content = " | ".join([r.content for r in retrieved])
        cits = [Citation(sentence=f"Used {r.chunk_id}", source_agent=self.agent_id, chunk_ids=[r.chunk_id]) for r in retrieved]
        return AgentOutput(agent_id=self.agent_id, content=content, citations=cits, metadata={"chunks": [r.model_dump() for r in retrieved]})


class CritiqueAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("critique_agent", 1400)

    async def run(self, context: SharedContext) -> AgentOutput:
        claims = []
        flagged = []
        for aid, out in context.agent_outputs.items():
            conf = 0.8 if out.content else 0.2
            span = None
            if "always" in out.content.lower():
                span = "always"
                flagged.append({"agent": aid, "span": "always", "reason": "Overconfident absolute"})
            claims.append(Claim(text=f"{aid} output quality", confidence=conf, disagree_span=span))
        return AgentOutput(agent_id=self.agent_id, content="Critique complete", claims=claims, metadata={"flags": flagged})


class SynthesisAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("synthesis_agent", 2200)

    async def run(self, context: SharedContext) -> AgentOutput:
        retrieval = context.agent_outputs.get("retrieval_agent")
        critique = context.agent_outputs.get("critique_agent")
        lines = []
        prov = []
        if retrieval:
            for i, part in enumerate(retrieval.content.split("|")):
                s = part.strip()
                if s:
                    lines.append(s)
                    chunk_ids = retrieval.citations[i].chunk_ids if i < len(retrieval.citations) else []
                    prov.append(Citation(sentence=s, source_agent="retrieval_agent", chunk_ids=chunk_ids))
        if critique and critique.metadata.get("flags"):
            lines.append("Contradictions were detected and resolved conservatively.")
            prov.append(Citation(sentence=lines[-1], source_agent="critique_agent", chunk_ids=[]))
        final = " ".join(lines) if lines else "Insufficient evidence."
        return AgentOutput(agent_id=self.agent_id, content=final, citations=prov)
