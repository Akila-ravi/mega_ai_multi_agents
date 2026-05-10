from app.context_manager.budget_manager import ContextBudgetManager
from app.schemas import SharedContext


def test_budget_violation():
    ctx = SharedContext(job_id="j1", user_query="q")
    mgr = ContextBudgetManager(ctx)
    mgr.declare_budget("a", 2)
    raised = False
    try:
        mgr.consume("a", "this text is long")
    except ValueError:
        raised = True
    assert raised
