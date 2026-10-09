import pytest

pytest.importorskip("math_verify")

from archlab.rl.limite_data import math_reward, split_rows


def test_reward_requires_answer_and_rejects_incomplete_reasoning():
    assert math_reward(r"Therefore, $\boxed{\frac{1}{2}}$.", "0.5") == 1
    assert math_reward(r"Therefore, $\boxed{3}$.", "2") == 0
    assert math_reward("I saw 42 in an intermediate calculation.", "42") == 0
    assert math_reward(r"<think>Still reasoning: \boxed{42}", "42") == 0
    assert math_reward("42", "42") == 1
    assert math_reward("Therefore the sum is 42.\nThank you.", "42") == 1
    assert math_reward("Therefore the sum is 42.\nCorrection: 43.", "42") == 0
    assert (
        math_reward(
            "Intermediate work gives 12.\n\nTherefore, the sum is 42. Option A is 42.", "42"
        )
        == 1
    )


def test_problem_disjoint_split_deduplicates_and_excludes_conflicting_gold():
    def row(key, answer, uuid):
        return dict(
            problem_sha256=key,
            expected_answer=answer,
            uuid=uuid,
            prompt_text="question",
            input_ids=[1],
        )

    rows = [
        row("a", "2", "1"),
        row("a", "2", "2"),
        row("b", "3", "3"),
        row("c", "4", "4"),
        row("c", "5", "5"),
        row("d", "6", "6"),
    ]
    train, heldout, excluded = split_rows(rows, heldout_count=1)
    assert len(train) == 2 and len(heldout) == 1 and len(excluded) == 3
    assert not {r["problem_sha256"] for r in train} & {r["problem_sha256"] for r in heldout}
    assert sum(r["reason"] == "duplicate_problem" for r in excluded) == 1
