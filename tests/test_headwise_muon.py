import copy

import pytest
import torch

from archlab.optimizers.headwise_muon import HeadwiseMuon, orthogonalized_direction


def test_headwise_update_matches_independent_matrices():
    torch.manual_seed(81)
    packed = torch.nn.Parameter(torch.randn(12, 16))
    heads = [torch.nn.Parameter(t.clone()) for t in packed.detach().split(4)]
    a = HeadwiseMuon([{"params": [packed], "head_dim": 4}])
    b = HeadwiseMuon(heads)
    for _ in range(3):
        packed.grad = torch.randn_like(packed)
        for h, g in zip(heads, packed.grad.split(4), strict=True):
            h.grad = g.clone()
        a.step()
        b.step()
        torch.testing.assert_close(packed, torch.cat(heads), rtol=2e-6, atol=1e-6)


def test_zero_direction_and_zero_output_can_start():
    zero = torch.zeros(2, 4, 8)
    assert torch.equal(orthogonalized_direction(zero), zero)
    p = torch.nn.Parameter(torch.zeros(4, 8))
    opt = HeadwiseMuon([p])
    p.grad = torch.zeros_like(p)
    opt.step()
    assert p.count_nonzero() == 0
    p.grad = torch.randn_like(p)
    opt.step()
    assert p.isfinite().all() and p.count_nonzero()


def test_optimizer_reload_preserves_next_update_and_grouping():
    p = torch.nn.Parameter(torch.randn(8, 16))
    original = HeadwiseMuon([{"params": [p], "head_dim": 4}])
    p.grad = torch.randn_like(p)
    original.step()
    q = torch.nn.Parameter(p.detach().clone())
    resumed = HeadwiseMuon([q])
    resumed.load_state_dict(copy.deepcopy(original.state_dict()))
    p.grad = torch.randn_like(p)
    q.grad = p.grad.clone()
    original.step()
    resumed.step()
    torch.testing.assert_close(q, p, rtol=0, atol=0)
    assert resumed.param_groups[0]["head_dim"] == 4


def test_reject_nonmatrix_and_low_precision_master():
    for p in [torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.ones(4, 8, dtype=torch.bfloat16))]:
        with pytest.raises(ValueError):
            HeadwiseMuon([p])
