from statistics import mean

from app.evaluation.engine import build_eval_cases, score_case


def test_build_eval_cases_size():
    assert len(build_eval_cases()) == 15


def _count_processed(*, failed_only: bool) -> int:
    cases = build_eval_cases()
    n = 0
    for c in cases:
        if failed_only and c.category == "normal":
            continue
        n += 1
    return n


def test_eval_processing_counts_match_endpoint_logic():
    assert _count_processed(failed_only=True) == 10
    assert _count_processed(failed_only=False) == 15


def test_score_case_overall_is_mean_of_dimension_scores():
    s = score_case("Fact one and Fact two. Resolved cleanly.", ["Fact", "Resolved"], tool_calls=3, violations=0)
    dims = [
        "correctness",
        "citation_accuracy",
        "contradiction_resolution",
        "tool_efficiency",
        "context_compliance",
        "critique_agreement",
    ]
    assert set(dims).issubset(s.keys())
    assert "overall" in s
    expected = round(mean(s[k]["score"] for k in dims), 3)
    assert s["overall"]["score"] == expected


def test_tool_efficiency_penalty_when_excess_tool_calls():
    low = score_case("Fact here", ["Fact"], tool_calls=10, violations=0)
    baseline = score_case("Fact here", ["Fact"], tool_calls=3, violations=0)
    assert low["tool_efficiency"]["score"] < baseline["tool_efficiency"]["score"]
