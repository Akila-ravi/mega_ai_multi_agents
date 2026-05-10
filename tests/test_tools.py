import pytest

from app.tools.interfaces import ToolInput, python_sandbox, self_reflection, sql_lookup, web_search_stub


@pytest.mark.asyncio
async def test_web_search_malformed():
    res = await web_search_stub(ToolInput(payload={}))
    assert not res.ok
    assert res.failure.mode == "malformed"


@pytest.mark.asyncio
async def test_python_sandbox_ok():
    res = await python_sandbox(ToolInput(payload={"code": "print(1+1)"}))
    assert res.ok
    assert "2" in res.data["stdout"]


@pytest.mark.asyncio
async def test_sql_lookup_empty():
    res = await sql_lookup(ToolInput(payload={"question": "unknown"}))
    assert not res.ok
    assert res.failure.mode == "empty"


@pytest.mark.asyncio
async def test_self_reflection_malformed():
    res = await self_reflection(ToolInput(payload={"outputs": "bad"}))
    assert not res.ok
    assert res.failure.mode == "malformed"
