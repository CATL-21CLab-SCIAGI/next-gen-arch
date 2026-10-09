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
    assert torch.allclose(ratio, torch.tensor([[0.1, 0.2]]).exp())
    (ratio * mask).sum().backward()
    assert behavior.grad is None
    assert current.grad[0, 0] > 1 and current.grad[0, 1] == 0
