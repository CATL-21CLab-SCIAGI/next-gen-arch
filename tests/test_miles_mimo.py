import copy

import pytest
import torch

from archlab.optimizers.muown import Muown
from archlab.rl.miles_mimo import masked_importance_objective


def test_mimo_group_normalization_and_detached_ratio():
    scores = [torch.tensor([-2.0, -3.0], requires_grad=True),
              torch.tensor([-4.0], requires_grad=True)]
    advantages = [0.7, -0.7]
    losses = [masked_importance_objective(x, x.detach(), torch.full_like(x, a),
                                          torch.ones_like(x), 3, 2)[0]
              for x, a in zip(scores, advantages, strict=True)]
    # Miles averages the two trajectory contributions into one prompt loss.
    (sum(losses) / 2).backward()
    for x, a in zip(scores, advantages, strict=True):
        torch.testing.assert_close(x.grad, torch.full_like(x, -a / 3))


def test_ratio_outliers_are_masked_for_both_advantage_signs():
    for advantage in (-1.0, 1.0):
        logp = torch.log(torch.tensor([0.1, 1.0, 6.0])).requires_grad_()
        loss, _ = masked_importance_objective(logp, torch.zeros(3),
                                              torch.full((3,), advantage), torch.ones(3), 3, 1)
        loss.backward()
        torch.testing.assert_close(logp.grad, torch.tensor([0.0, -advantage / 3, 0.0]))


def test_muown_radial_gradient_and_checkpoint_continuation():
    torch.manual_seed(37)
    p = torch.nn.Parameter(torch.randn(7, 11))
    original = p.detach().clone()
    optimizer = Muown([p], lr=1e-3)
    p.grad = p.detach() / p.detach().norm(dim=1, keepdim=True)
    optimizer.step()
    # Pure radial gradient has zero direction component and an Adam magnitude update.
    torch.testing.assert_close(p.norm(dim=1), original.norm(dim=1) - 1e-3, atol=2e-6, rtol=2e-6)
    state = copy.deepcopy(optimizer.state_dict())
    restored = torch.nn.Parameter(p.detach().clone())
    other = Muown([restored], lr=1e-3)
    other.load_state_dict(state)
    for _ in range(3):
        grad = torch.randn_like(p)
        p.grad = grad.clone()
        restored.grad = grad.clone()
        optimizer.step()
        other.step()
        torch.testing.assert_close(p, restored, rtol=0, atol=0)
    assert not torch.equal(p, original)


def test_nonfinite_scores_fail_closed():
    with pytest.raises(FloatingPointError):
        masked_importance_objective(torch.tensor([float("nan")]), torch.zeros(1),
                                     torch.ones(1), torch.ones(1), 1, 1)
