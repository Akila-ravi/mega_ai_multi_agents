from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    job_id: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class TraceResponse(BaseModel):
    job_id: str
    status: str
    logs: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    result: dict[str, Any] | None = None


class PromptApprovalRequest(BaseModel):
    prompt_version_id: int
    decision: Literal["approved", "rejected"]


class EvalRetryRequest(BaseModel):
    run_type: Literal["failed_only"] = "failed_only"


class ToolFailure(BaseModel):
    mode: Literal["timeout", "empty", "malformed"]
    message: str


class ToolResult(BaseModel):
    ok: bool
    tool_name: str
    data: dict[str, Any] = Field(default_factory=dict)
    failure: ToolFailure | None = None
    latency_ms: float = 0


class Claim(BaseModel):
    text: str
    confidence: float
    disagree_span: str | None = None


class DecompositionTask(BaseModel):
    task_id: str
    task_type: str
    description: str
    dependencies: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_url: str
    content: str
    relevance: float


class Citation(BaseModel):
    sentence: str
    source_agent: str
    chunk_ids: list[str] = Field(default_factory=list)


class AgentOutput(BaseModel):
    agent_id: str
    content: str
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextEvent(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str
    actor: str
    payload: dict[str, Any] = Field(default_factory=dict)


class SharedContext(BaseModel):
    job_id: str
    user_query: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    agent_outputs: dict[str, AgentOutput] = Field(default_factory=dict)
    events: list[ContextEvent] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    context_budget: dict[str, int] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    policy_violations: list[str] = Field(default_factory=list)
