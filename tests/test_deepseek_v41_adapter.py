"""Leaf correctness only, not a claim of pretrained or distributed support."""

from dataclasses import replace

import pytest
import torch

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter


def small_config():
    return V41AdapterConfig(width=16, query_heads=4, kv_heads=2, head_dim=16,
                            short_window=2, long_window=3)


def test_production_shapes_and_count():
    c = V41AdapterConfig()
    with torch.device("meta"):
        m = V41SimplicialAdapter(c)
    assert sum(p.numel() for p in m.parameters()) == c.parameter_count() == 20_977_032
    assert m.q.weight.shape == (1024, 5120)
    assert m.k1.weight.shape == (256, 5120)
    assert m.output.weight.shape == (5120, 1024)
    assert m.read_logits.shape == m.write_logits.shape == (4,)


@pytest.mark.parametrize("kwargs", [
    {"width": 0}, {"streams": True}, {"head_dim": 192}, {"query_heads": 3},
    {"short_window": 513}, {"norm_eps": float("nan")}, {"initializer_std": 0},
])
def test_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        replace(V41AdapterConfig(), **kwargs)


def test_identity_rng_and_two_step_gradient_onset():
    state = torch.get_rng_state().clone()
    m = V41SimplicialAdapter(small_config(), backend="reference")
    assert torch.equal(state, torch.get_rng_state())
    torch.manual_seed(67)
    x = torch.randn(2, 4, 4, 16, requires_grad=True)
    dy = torch.randn_like(x)
    y = m(x)
    assert torch.equal(y, x)
    y.backward(dy)
    assert torch.equal(x.grad, dy)
    for name, p in m.named_parameters():
        assert p.grad is not None and p.grad.isfinite().all(), name
        assert bool(p.grad.count_nonzero()) == (name == "output.weight"), name
    with torch.no_grad():
        m.output.weight.add_(m.output.weight.grad, alpha=-0.01)
    m.zero_grad(set_to_none=True)
    m(x).backward(dy)
    for name, p in m.named_parameters():
        assert p.grad is not None and p.grad.isfinite().all() and p.grad.count_nonzero(), name


def test_causal_prefix_and_state_reload():
    torch.manual_seed(99)
    m = V41SimplicialAdapter(small_config(), backend="reference")
    with torch.no_grad():
        m.output.weight.normal_(std=.02)
    x = torch.randn(1, 5, 4, 16)
    torch.testing.assert_close(m(x)[:, :3], m(x[:, :3]))
    restored = V41SimplicialAdapter(small_config(), backend="reference", seed=700)
    restored.load_state_dict(m.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), m(x), atol=0, rtol=0)
