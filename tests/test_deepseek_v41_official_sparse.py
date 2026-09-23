"""Check sparse reduction order, native numerics, and private dispatch binding."""

from pathlib import Path

import pytest
import torch

from archlab.automodel.deepseek_v41_official_sparse import (
    _ordered_index_sum, deterministic_sparse_attention, install_official_deterministic_sparse,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_key_reduction_is_repeatable_and_handles_empty_input(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires frozen-container GPU")
    torch.manual_seed(52)
    indices = torch.randint(0, 11, (233,), device=device)
    values = torch.randn(233, 32, device=device)
    expected = torch.zeros(11, 32, device=device)
    for i in range(indices.numel()):
        expected[indices[i]] += values[i]
    repeats = []
    for _ in range(4):
        actual = torch.zeros_like(expected)
        _ordered_index_sum(actual, indices, values)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        repeats.append(actual)
    assert all(torch.equal(value, repeats[0]) for value in repeats)
    _ordered_index_sum(repeats[0], indices[:0], values[:0])
    assert torch.equal(repeats[0], repeats[1])


def test_dispatch_binding_preserves_forward_code_globals_and_parameters():
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.attention import DeepseekV41Attention
    from archlab.automodel.deepseek_v41_official_execution import tiny_official_config

    with torch.device("meta"):
        model = DeepseekV41Attention(tiny_official_config(Path("unused")).text_config, 0,
                                     BackendConfig(attn="tilelang", linear="torch", rms_norm="torch_fp32"))
    model.requires_grad_(False)
    original = model.forward.__func__
    dispatcher = original.__globals__["dsv4_sparse_attention"]
    parameters = dict(model.named_parameters())
    report = install_official_deterministic_sparse(model)
    assert report["original_parameters_preserved"]
    assert model.forward.__func__.__code__ is original.__code__
    assert model.forward.__func__.__globals__ is not original.__globals__
    assert original.__globals__["dsv4_sparse_attention"] is dispatcher
    assert model.forward.__func__.__globals__["dsv4_sparse_attention"] is deterministic_sparse_attention
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())
    with pytest.raises(ValueError, match="once"):
        install_official_deterministic_sparse(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires frozen-container GPU")
def test_sparse_native_forward_gradients_and_tail_mask_repeat_exactly():
    from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sparse_attention

    torch.manual_seed(53)
    q = torch.randn(1, 33, 4, 64, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, 45, 64, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(4, device="cuda", dtype=torch.float32)
    indices = torch.randint(-1, 45, (1, 33, 17), device="cuda", dtype=torch.int32)
    indices[:, -1] = -1
    cotangent = torch.randn_like(q)
    results = []
    for fn in (dsv4_sparse_attention, *[deterministic_sparse_attention] * 3):
        x, k = q.clone().requires_grad_(), kv.clone().requires_grad_()
        y = fn(x, k, sinks, indices, 64 ** -.5, backend="tilelang", reference_rounding=True)
        y.backward(cotangent)
        results.append((y.detach(), x.grad, k.grad))
    assert torch.equal(results[0][0], results[1][0])
    for actual, expected in zip(results[1], results[0], strict=True):
        assert actual.isfinite().all()
        relative = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)
        assert relative < .01
    for repeated in results[2:]:
        assert all(torch.equal(value, expected) for value, expected in zip(repeated, results[1], strict=True))
    assert not results[1][0][:, -1].count_nonzero()
    assert not results[1][1][:, -1].count_nonzero()
