from app.context_manager.budget_manager import ContextBudgetManager
from app.schemas import SharedContext


def test_budget_violation_strict_mode():
    ctx = SharedContext(job_id="j1", user_query="q")
    mgr = ContextBudgetManager(ctx)
    mgr.declare_budget("a", 2)
    raised = False
    try:
        mgr.consume("a", "this text is long", mode="strict")
    except ValueError:
        raised = True
    assert raised


def test_budget_truncates_when_space_tight():
    ctx = SharedContext(job_id="j2", user_query="q")
    mgr = ContextBudgetManager(ctx)
    mgr.declare_budget("a", 3)
    out = mgr.consume("a", "this text is noticeably longer than a couple of tokens")
    assert len(out) < len("this text is noticeably longer than a couple of tokens")
    assert mgr.remaining("a") >= 0
