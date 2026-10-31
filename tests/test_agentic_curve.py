import asyncio

import pytest

from archlab.evaluation.agentic_curve import (
    bind_invocation,
    completed_records,
    error_outcome,
    paired_difference,
    policy_cap_reason,
    sample_seed,
    summarize,
)


def test_missing_infrastructure_does_not_improve_or_zero_score():
    rows = [dict(task="a", seed=1, status="scored", reward=1.0),
            dict(task="a", seed=2, status="missing", reward=None)]
    result = summarize(rows, ["a"], [1, 2])
    assert not result["complete"] and result["pass_at_1"] is None
    assert result["missing_samples"] == 1 and result["complete_task_count"] == 0


def test_complete_grid_keeps_failed_attempt_then_scores_retry():
    rows = [dict(task="a", seed=1, status="missing", reward=None),
            dict(task="a", seed=1, status="scored", reward=1.0),
            dict(task="a", seed=2, status="scored", reward=0.0),
            dict(task="b", seed=1, status="scored", reward=0.0),
            dict(task="b", seed=2, status="scored", reward=0.0)]
    result = summarize(rows, ["a", "b"], [1, 2], bootstrap=100)
    assert result["complete"] and result["pass_at_1"] == .25 and result["success_at_k"] == .5
    assert result["infrastructure_attempts"] == 1


def test_duplicate_scored_or_unexpected_episode_rejected():
    row = dict(task="a", seed=1, status="scored", reward=0.0)
    with pytest.raises(ValueError, match="Duplicate"):
        completed_records([row, row], ["a"], [1])
    with pytest.raises(ValueError, match="outside"):
        completed_records([row], ["b"], [1])


def test_wall_timer_includes_infrastructure_and_must_remain_missing():
    assert error_outcome(asyncio.TimeoutError("session exceeded wall timeout of7200s"))["termination"] == "wall_budget_unclassified"
    for exc in [TimeoutError("connection timed out"), RuntimeError("verifier failed"), ValueError("unknown")]:
        assert error_outcome(exc)["status"] == "missing"


def test_seed_identity_shared_across_checkpoints_and_distinct_for_tasks():
    assert sample_seed("a", 101) == sample_seed("a", 101)
    assert len({sample_seed(t, s) for t in ["a", "b"] for s in [101, 211]}) == 4


def test_backend_errors_are_missing_even_when_verifier_completed():
    with pytest.raises(RuntimeError, match="backend failure"):
        policy_cap_reason(dict(agent_status="ModelQueryError", result="Connection refused", reward=0), 1)
    assert policy_cap_reason(dict(agent_status="ModelQueryError", result="maximum context length exceeded"), 42) == "context_limit"
    assert policy_cap_reason(dict(agent_status="StepLimitExceeded"), 300) == "turn_limit"
    assert policy_cap_reason(dict(agent_completed=True), 3) == "none"


def test_only_seed_prefix_expansion_permitted(tmp_path):
    path = tmp_path / "invocation.json"
    bind_invocation(path, dict(seeds=[101, 211], model_bindings={"sha": "abc"}))
    bind_invocation(path, dict(seeds=[101, 211, 307], model_bindings={"sha": "abc"}))
    with pytest.raises(ValueError, match="different"):
        bind_invocation(path, dict(seeds=[101, 211, 307], model_bindings={"sha": "changed"}))
    with pytest.raises(ValueError, match="different"):
        bind_invocation(path, dict(seeds=[211, 101, 307], model_bindings={"sha": "abc"}))


def test_paired_comparison_accounts_for_task_dependence():
    a = [dict(task=t, seed=1, status="scored", reward=0.) for t in ["a", "b"]]
    b = [dict(task=t, seed=1, status="scored", reward=1.) for t in ["a", "b"]]
    result = paired_difference(a, b, ["a", "b"], [1], bootstrap=100)
    assert result["delta_pass_at_1"] == 1. and result["task_paired_95ci"] == [1., 1.]
