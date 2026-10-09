import pytest
import torch

from archlab.rl.limite_rollout import with_behavior_logprobs


def test_behavior_denominator_is_detached_and_preserves_mask():
    behavior = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    mask = torch.tensor([[1, 0]])
    original = dict(sampling_per_token_logps=behavior, completion_mask=mask)
    adapted = with_behavior_logprobs(original)
    assert "old_per_token_logps" not in original
    assert not adapted["old_per_token_logps"].requires_grad
    assert adapted["completion_mask"] is mask
    current = torch.tensor([[-0.9, -1.8]], requires_grad=True)
    ratio = (current - adapted["old_per_token_logps"]).exp()
    assert torch.allclose(ratio[:, :1], torch.tensor([[0.1]]).exp())
    assert adapted["old_per_token_logps"][0, 1] == 0
    (ratio * mask).sum().backward()
    assert behavior.grad is None
    assert current.grad[0, 0] > 1 and current.grad[0, 1] == 0


def test_actual_behavior_denominator_preserves_policy_expectation_and_gradient():
    # Different finite-precision sampling/replay backends can represent Q and P
    # at identical checkpoint weights. Exact categorical enumeration verifies
    # the importance estimator, without sampling noise or a ratio=1 assumption.
    logits = torch.nn.Parameter(torch.tensor([.36, .26, .38], dtype=torch.float64).log())
    sampling = torch.tensor([.34, .25, .41], dtype=torch.float64)
    behavior = sampling.log()[:, None].requires_grad_()
    inputs = with_behavior_logprobs(dict(sampling_per_token_logps=behavior,
                                       completion_mask=torch.ones(3, 1)))
    current = logits.log_softmax(-1)[:, None]
    ratio = (current - inputs["old_per_token_logps"]).exp().flatten()
    advantages = torch.tensor([1., -.5, .3], dtype=torch.float64)
    assert not torch.allclose(ratio, torch.ones_like(ratio))
    assert bool(((ratio > .8) & (ratio < 1.2)).all())
    actual = -(sampling * torch.minimum(ratio * advantages,
                                       ratio.clamp(.8, 1.2) * advantages)).sum()
    expected = -(logits.softmax(-1) * advantages).sum()
    torch.testing.assert_close(actual, expected)
    actual_gradient, behavior_gradient = torch.autograd.grad(actual, (logits, behavior),
                                                           allow_unused=True, retain_graph=True)
    torch.testing.assert_close(actual_gradient, torch.autograd.grad(expected, logits)[0])
    assert behavior_gradient is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), .1])
def test_invalid_sampled_scores_fail_but_padding_is_not_a_sampled_distribution(value):
    original = dict(sampling_per_token_logps=torch.tensor([[-1., value]]),
                    completion_mask=torch.tensor([[1, 0]]))
    result = with_behavior_logprobs(original)
    assert result["old_per_token_logps"].tolist() == [[-1., 0.]]
    original["completion_mask"][0, 1] = 1
    with pytest.raises(FloatingPointError, match="sampled behavior"):
        with_behavior_logprobs(original)


def test_behavior_scores_must_match_binary_completion_mask():
    for mask in (torch.ones(1, 3), torch.tensor([[1., .5]])):
        with pytest.raises(ValueError, match="mask"):
            with_behavior_logprobs(dict(sampling_per_token_logps=torch.tensor([[-1., -2.]]),
                                        completion_mask=mask))
