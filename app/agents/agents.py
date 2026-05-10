from __future__ import annotations

from pathlib import Path

from app.agents.llm import chat_completion, llm_enabled, parse_json_object
from app.schemas import AgentOutput, Citation, Claim, DecompositionTask, RetrievedChunk, SharedContext

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def get_active_prompt(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _fallback_route(user_query: str) -> tuple[list[str], str]:
    route = ["decomposition_agent", "retrieval_agent", "critique_agent", "synthesis_agent"]
    if "calculate" in user_query.lower() or "python" in user_query.lower():
        idx = route.index("retrieval_agent") + 1
        route.insert(idx, "python_tool_check")
    return route, "Keyword fallback routing"


async def plan_route_with_llm(user_query: str) -> tuple[list[str], str]:
    if not llm_enabled():
        return _fallback_route(user_query)
    system = get_active_prompt("routing") or "Return JSON {\"route\":[...],\"justification\":\"...\"}."
    raw = await chat_completion(system=system, user=f"User query:\n{user_query}", temperature=0.1, max_tokens=400)
    data = parse_json_object(raw)
    route = data.get("route")
    justification = str(data.get("justification", "") or "")
    allowed = {"decomposition_agent", "retrieval_agent", "critique_agent", "synthesis_agent", "python_tool_check"}
    fixed: list[str] = []
    seen: set[str] = set()
    if isinstance(route, list):
        for step in route:
            if isinstance(step, str) and step in allowed and step not in seen:
                fixed.append(step)
                seen.add(step)
    if "decomposition_agent" not in seen:
        fixed.insert(0, "decomposition_agent")
        seen.add("decomposition_agent")
    for req in ["retrieval_agent", "critique_agent", "synthesis_agent"]:
        if req not in seen:
            fixed.append(req)
            seen.add(req)
    return fixed, justification or "LLM routing"


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
        if llm_enabled():
            system = get_active_prompt("decomposition")
            raw = await chat_completion(system=system or "Return JSON.", user=q, temperature=0.2, max_tokens=600)
            data = parse_json_object(raw)
            tasks_out: list[DecompositionTask] = []
            for t in data.get("tasks", [])[:6]:
                if not isinstance(t, dict):
                    continue
                try:
                    tasks_out.append(
                        DecompositionTask(
                            task_id=str(t.get("task_id", "")),
                            task_type=str(t.get("task_type", "")),
                            description=str(t.get("description", "")),
                            dependencies=list(t.get("dependencies", [])) if isinstance(t.get("dependencies"), list) else [],
                        )
                    )
                except Exception:
                    continue
            if len(tasks_out) < 3:
                tasks_out = [
                    DecompositionTask(task_id="t1", task_type="intent", description=f"Understand user intent for: {q}", dependencies=[]),
                    DecompositionTask(task_id="t2", task_type="retrieve", description="Retrieve supporting chunks", dependencies=["t1"]),
                    DecompositionTask(task_id="t3", task_type="synthesize", description="Merge and validate claims", dependencies=["t2"]),
                ]
            summary = str(data.get("summary", "") or raw[:280] or "Planned decomposition")
            return AgentOutput(agent_id=self.agent_id, content=summary, metadata={"tasks": [t.model_dump() for t in tasks_out], "mode": "llm"})
        tasks = [
            DecompositionTask(task_id="t1", task_type="intent", description=f"Understand user intent for: {q}", dependencies=[]),
            DecompositionTask(task_id="t2", task_type="retrieve", description="Retrieve at least two supporting chunks", dependencies=["t1"]),
            DecompositionTask(task_id="t3", task_type="synthesize", description="Merge and validate claims", dependencies=["t2"]),
        ]
        return AgentOutput(
            agent_id=self.agent_id,
            content="Decomposed query into dependency graph",
            metadata={"tasks": [t.model_dump() for t in tasks], "mode": "heuristic"},
        )


class RetrievalAgent(BaseAgent):
    SEP = "\n---\n"

    def __init__(self) -> None:
        super().__init__("retrieval_agent", 1800)

    async def run(self, context: SharedContext, chunks: list[dict]) -> AgentOutput:
        ranked = sorted(chunks, key=lambda c: float(c.get("relevance", 0)), reverse=True)
        take = min(3, len(ranked))
        selected = ranked[:take] if take else []
        retrieved = [
            RetrievedChunk(chunk_id=c["chunk_id"], source_url=c["url"], content=c["content"], relevance=float(c["relevance"])) for c in selected
        ]
        base = self.SEP.join([r.content for r in retrieved])
        llm_note = ""
        if llm_enabled() and base.strip():
            system = get_active_prompt("retrieval")
            blob = "\n".join([f"[{r.chunk_id}] {r.content}" for r in retrieved])
            user = f"Query:\n{context.user_query}\n\nPassages:\n{blob}"
            llm_note = await chat_completion(system=system or "Summarize grounding.", user=user, temperature=0.2, max_tokens=350)
        content = f"{llm_note}{self.SEP}{base}" if llm_note else base
        cits = [Citation(sentence=f"Evidence {r.chunk_id}", source_agent=self.agent_id, chunk_ids=[r.chunk_id]) for r in retrieved]
        chunk_meta = [
            {"chunk_id": c["chunk_id"], "url": c.get("url"), "title": c.get("title"), "content": c.get("content"), "relevance": c.get("relevance")}
            for c in selected
        ]
        return AgentOutput(
            agent_id=self.agent_id,
            content=content,
            citations=cits,
            metadata={"chunks": chunk_meta, "llm_note": llm_note, "mode": "llm" if llm_note else "heuristic"},
        )


class CritiqueAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("critique_agent", 1400)

    async def run(self, context: SharedContext) -> AgentOutput:
        if llm_enabled():
            system = get_active_prompt("critique")
            digest = []
            for aid, out in context.agent_outputs.items():
                digest.append({"agent": aid, "content": out.content[:2000]})
            raw = await chat_completion(
                system=system or "Return JSON critique.",
                user=f"Artifacts:\n{digest}",
                temperature=0.1,
                max_tokens=700,
            )
            data = parse_json_object(raw)
            claims = []
            for c in data.get("claims", [])[:24]:
                if not isinstance(c, dict):
                    continue
                claims.append(
                    Claim(
                        text=str(c.get("text", "claim")),
                        confidence=float(c.get("confidence", 0.5)),
                        disagree_span=str(c.get("disagree_span")) if c.get("disagree_span") else None,
                    )
                )
            flags = []
            for f in data.get("flags", [])[:24]:
                if isinstance(f, dict):
                    flags.append({k: f.get(k, "") for k in ["agent", "span", "reason"]})
            content = str(data.get("content", "Critique complete"))
            return AgentOutput(agent_id=self.agent_id, content=content, claims=claims, metadata={"flags": flags, "mode": "llm"})
        claims = []
        flagged = []
        for aid, out in context.agent_outputs.items():
            conf = 0.8 if out.content else 0.2
            span = None
            if "always" in out.content.lower():
                span = "always"
                flagged.append({"agent": aid, "span": "always", "reason": "Overconfident absolute"})
            claims.append(Claim(text=f"{aid} output quality", confidence=conf, disagree_span=span))
        return AgentOutput(agent_id=self.agent_id, content="Critique complete", claims=claims, metadata={"flags": flagged, "mode": "heuristic"})


class SynthesisAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("synthesis_agent", 2200)

    async def run(self, context: SharedContext) -> AgentOutput:
        retrieval = context.agent_outputs.get("retrieval_agent")
        critique = context.agent_outputs.get("critique_agent")
        if llm_enabled():
            system = get_active_prompt("synthesis")
            retrieval_blob = retrieval.content if retrieval else ""
            critique_blob = critique.content if critique else ""
            user = f"User query:\n{context.user_query}\n\nRetrieval bundle:\n{retrieval_blob}\n\nCritique:\n{critique_blob}"
            answer = await chat_completion(system=system or "Answer concisely.", user=user, temperature=0.2, max_tokens=700)
            if answer.strip():
                return AgentOutput(agent_id=self.agent_id, content=answer.strip(), citations=[], metadata={"mode": "llm"})

        lines: list[str] = []
        prov: list[Citation] = []
        if retrieval:
            chunk_meta = retrieval.metadata.get("chunks") if retrieval.metadata else None
            if isinstance(chunk_meta, list) and chunk_meta:
                for i, ch in enumerate(chunk_meta):
                    s = str(ch.get("content", "")).strip()
                    if s:
                        lines.append(s)
                        chunk_ids = retrieval.citations[i].chunk_ids if i < len(retrieval.citations) else []
                        prov.append(Citation(sentence=s[:500], source_agent="retrieval_agent", chunk_ids=chunk_ids))
            else:
                for i, part in enumerate(retrieval.content.split(RetrievalAgent.SEP)):
                    s = part.strip()
                    if s:
                        lines.append(s)
                        chunk_ids = retrieval.citations[i].chunk_ids if i < len(retrieval.citations) else []
                        prov.append(Citation(sentence=s, source_agent="retrieval_agent", chunk_ids=chunk_ids))
        if critique and critique.metadata.get("flags"):
            lines.append("Contradictions were detected and resolved conservatively.")
            prov.append(Citation(sentence=lines[-1], source_agent="critique_agent", chunk_ids=[]))
        final = "\n".join(lines) if lines else "Insufficient evidence."
        return AgentOutput(agent_id=self.agent_id, content=final, citations=prov, metadata={"mode": "heuristic"})
