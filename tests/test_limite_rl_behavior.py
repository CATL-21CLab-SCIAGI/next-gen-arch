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
