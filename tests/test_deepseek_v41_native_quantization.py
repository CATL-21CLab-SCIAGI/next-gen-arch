from types import SimpleNamespace

import pytest
import torch

from archlab.architectures.deepseek_v41_torch import rounded_activation
from archlab.automodel.deepseek_v41_native_quantization import _fp8_quantize, install_native_row_padding


@pytest.mark.parametrize("block_size", [32, 128])
@pytest.mark.parametrize("shape", [(17, 128), (1, 2048, 128)])
def test_native_format_matches_rounded_activation_and_repeats(shape, block_size):
    torch.manual_seed(42)
    x = (torch.randn(shape) * .02).bfloat16()
    values, scales = _fp8_quantize(x, block_size, "ue8m0", torch.float8_e8m0fnu)
    decoded = (values.float().unflatten(-1, (-1, block_size)) * scales.float().unsqueeze(-1)).flatten(-2)
    expected = rounded_activation(x, block_size=block_size, straight_through=False)
    torch.testing.assert_close(decoded.bfloat16(), expected, rtol=0, atol=0)
    repeated, repeated_scales = _fp8_quantize(x, block_size, "ue8m0", torch.float8_e8m0fnu)
    assert torch.equal(values.view(torch.uint8), repeated.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), repeated_scales.view(torch.uint8))
    assert values.float().isfinite().all() and scales.float().isfinite().all()


def test_unrounded_scales_and_inplace_alias_keep_native_equations():
    x = torch.linspace(-.37, .42, 17 * 128).reshape(17, 128).bfloat16()
    grouped = x.float().unflatten(-1, (4, 32))
    expected_scales = grouped.abs().amax(-1).clamp_min(1e-4) * (1. / 448)
    expected_values = (grouped / expected_scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    values, scales = _fp8_quantize(x, 32)
    assert torch.equal(values.view(torch.uint8), expected_values.flatten(-2).view(torch.uint8))
    assert torch.equal(scales, expected_scales)
    mutable = x.clone()
    assert _fp8_quantize(mutable, 32, inplace=True) is mutable
    torch.testing.assert_close(mutable, (expected_values.float() * scales.unsqueeze(-1)).flatten(-2).bfloat16(),
                               rtol=0, atol=0)


def test_power_of_two_ceiling_zero_and_noncontiguous_input():
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    x[0, 0], x[1, 0] = 448., 450.
    values, scales = _fp8_quantize(x, 32, "ue8m0", torch.float8_e8m0fnu)
    assert scales.float()[0, 0] == 1 and scales.float()[1, 0] == 2
    assert values.float()[2:].count_nonzero() == 0
    view = x[:, ::2]
    quantized, _ = _fp8_quantize(view, 32, "ue8m0")
    assert quantized.shape == view.shape and quantized.is_contiguous()
    assert _fp8_quantize(view, 32, "ue8m0", inplace=True) is view


def test_installer_uses_stable_fp8_keeps_gemm_padding_and_records_evidence():
    calls = []
    def old_linear(x, weight, bias=None):
        calls.append(x.shape)
        return torch.zeros(x.shape[0], weight.shape[0], dtype=torch.bfloat16)
    def forbidden_fp8(*args, **kwargs):
        raise AssertionError("unstable native FP8 quantizer called")
    reference = SimpleNamespace(linear=old_linear, act_quant=forbidden_fp8, fp4_act_quant=lambda *args: None)
    evidence = install_native_row_padding(reference)
    x = torch.ones(17, 64, dtype=torch.bfloat16)
    values, scales = reference.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    assert values.shape == x.shape and scales.shape == (17, 2)
    weight = torch.ones(128, 64).to(torch.float8_e4m3fn)
    assert reference.linear(x, weight).shape == (17, 128)
    assert calls == [torch.Size([32, 64])]
    assert reference._archlab_native_quantization == evidence
    assert evidence["native_gemms_retained"] and evidence["fp8_activation_backend"].startswith("pytorch")
