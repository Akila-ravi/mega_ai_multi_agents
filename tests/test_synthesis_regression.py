import pytest

from app.agents.agents import SynthesisAgent
from app.schemas import AgentOutput, Citation, SharedContext


@pytest.mark.asyncio
async def test_synthesis_extra_pipe_segments_need_no_matching_citations():
    syn = SynthesisAgent()
    ctx = SharedContext(job_id="j1", user_query="q")
    ctx.agent_outputs["retrieval_agent"] = AgentOutput(
        agent_id="retrieval_agent",
        content="unused",
        citations=[
            Citation(sentence="", source_agent="retrieval_agent", chunk_ids=["c1"]),
            Citation(sentence="", source_agent="retrieval_agent", chunk_ids=["c2"]),
            Citation(sentence="", source_agent="retrieval_agent", chunk_ids=["c3"]),
        ],
        metadata={
            "chunks": [
                {"content": "Part A"},
                {"content": "Part B"},
                {"content": "extra tail"},
            ]
        },
    )
    ctx.agent_outputs["critique_agent"] = AgentOutput(agent_id="critique_agent", content="ok", metadata={})
    out = await syn.run(ctx)
    assert "Part A" in out.content
    assert "Part B" in out.content
    assert "extra tail" in out.content
