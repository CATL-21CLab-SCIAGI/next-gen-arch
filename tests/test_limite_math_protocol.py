import pytest
import torch

from archlab.rl.limite_data import math_reward
from archlab.rl.limite_protocol import (
    MathRolloutProtocol,
    degenerate_repetition,
    score_math_rollout,
)


def test_unfinished_correct_text_is_a_trainable_failure_not_accuracy():
    protocol = MathRolloutProtocol(max_tokens=16, overlong_buffer=4)
    tokens = [42] * 16
    row = score_math_rollout(r"\boxed{42}", "42", tokens, "length", protocol, training=True)
    assert row["accuracy"] == 0 and row["reward"] == -1
    # Its negative advantage has a real derivative; masking the whole rollout
    # would destroy this learning signal even with the same penalty.
    logp = torch.tensor(-2.0, requires_grad=True)
    loss = -logp.exp() * row["reward"]
    loss.backward()
    assert logp.grad > 0
    evaluated = score_math_rollout(r"\boxed{42}", "42", tokens, "length", protocol, training=False)
    assert evaluated["reward"] == 0


def test_native_eos_retains_verifiable_accuracy_and_length_ramp():
    protocol = MathRolloutProtocol(max_tokens=16, overlong_buffer=4)
    for size, penalty in ((4, 0), (12, 0), (14, -.25), (16, -.5)):
        row = score_math_rollout(r"\boxed{42}", "42", [42] * (size - 1) + [151645], "eos", protocol, training=True)
        assert row["accuracy"] == 1 and row["overlong_penalty"] == penalty
        assert row["reward"] == 1 + penalty
        assert row["unfinished_penalty"] == 0


@pytest.mark.parametrize("reason", ["length", "stop", "repetition"])
def test_censored_boxed_calculation_is_never_a_success(reason):
    row = score_math_rollout("<think>" + r"\boxed{42}", "42", [42] * 3, reason, MathRolloutProtocol(), training=True)
    assert row["accuracy"] == 0 and row["reward"] < 0
    assert not row["reasoning_closed"]


def test_claimed_eos_must_match_sampled_token():
    with pytest.raises(ValueError, match="sampled token"):
        score_math_rollout("42", "42", [42], "eos", MathRolloutProtocol(), training=True)


@pytest.mark.parametrize("reason,tokens", [
    ("length", [42, 151645]), ("stop", [42, 151643]),
    ("repetition", [42, 151645]), ("eos", [151645, 42, 151645]),
])
def test_finish_receipt_cannot_hide_sampled_eos_or_tokens_after_eos(reason, tokens):
    with pytest.raises(ValueError, match="sampled token"):
        score_math_rollout("42", "42", tokens, reason, MathRolloutProtocol(), training=True)


def test_unknown_finish_reason_fails_instead_of_changing_reward():
    with pytest.raises(ValueError, match="finish reason"):
        score_math_rollout("42", "42", [42], "server_error", MathRolloutProtocol(), training=True)


@pytest.mark.parametrize("text", [
    "</think><think>" + r"\boxed{42}",
    "<think><think>42</think>" + r"\boxed{42}",
])
def test_unbalanced_reasoning_cannot_earn_accuracy_at_natural_eos(text):
    row = score_math_rollout(text, "42", [42, 151645], "eos", MathRolloutProtocol(), training=True)
    assert not row["reasoning_closed"] and row["accuracy"] == 0
    assert row["reward"] == -0.5
    assert math_reward(text, "42") == 0


def test_mimo_repetition_calibration_does_not_flag_short_math():
    assert not degenerate_repetition("1 + 1 = 2 " * 20)
    assert degenerate_repetition("wait " * 600)
    assert not degenerate_repetition(" ".join(str(n) for n in range(700)))
