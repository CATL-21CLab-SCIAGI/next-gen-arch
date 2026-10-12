import pytest

pytest.importorskip("math_verify")

from archlab.evaluation.math_reward import math_reward


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
