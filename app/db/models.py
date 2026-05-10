import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, Text, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class PromptDecision(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class AgentLog(Base):
    __tablename__ = "agent_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.id"), index=True)
    agent_id: Mapped[str] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    input_hash: Mapped[str] = mapped_column(String(128))
    output_hash: Mapped[str] = mapped_column(String(128))
    latency_ms: Mapped[float] = mapped_column(Float, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    policy_violation: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ToolCall(Base):
    __tablename__ = "tool_calls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.id"), index=True)
    agent_id: Mapped[str] = mapped_column(String(64), index=True)
    tool_name: Mapped[str] = mapped_column(String(64), index=True)
    tool_input: Mapped[dict] = mapped_column(JSON)
    tool_output: Mapped[dict] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float, default=0)
    accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    retry_index: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_type: Mapped[str] = mapped_column(String(32), default="full")
    summary: Mapped[dict] = mapped_column(JSON)
    details: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PromptVersion(Base):
    __tablename__ = "prompt_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt_key: Mapped[str] = mapped_column(String(64), index=True)
    old_prompt: Mapped[str] = mapped_column(Text)
    new_prompt: Mapped[str] = mapped_column(Text)
    diff: Mapped[str] = mapped_column(Text)
    justification: Mapped[str] = mapped_column(Text)
    decision: Mapped[PromptDecision] = mapped_column(Enum(PromptDecision), default=PromptDecision.pending)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
