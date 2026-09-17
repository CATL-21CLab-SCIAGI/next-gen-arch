"""Stabilize the imported native quantization interface without changing math.

The native FP8 quantizer produced changing FP8 values, including NaNs, from
identical finite inputs while its scales remained identical. PyTorch generates
the same packed FP8 values/scales before the existing native GEMMs. Native GEMM
row padding and the FP4 interface remain in place; reference and runtime files
are never edited.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def _fp8_quantize(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Native group-wise FP8 equations, including its exact power-of-two ceiling.

Keep raw FP32 scales for division and in-place dequantization: the native
kernel casts only the separately returned scale tensor to ``scale_dtype``.
The bit construction mirrors native fast_round_scale, avoiding log2 rounding
at exact exponent boundaries. FP8 conversion itself uses PyTorch's RNE cast.
"""
    if block_size < 1 or x.ndim < 1 or x.shape[-1] % block_size:
        raise ValueError("FP8 activation width must divide its positive block size")
    if scale_dtype not in (torch.float32, torch.float8_e8m0fnu):
        raise ValueError("native FP8 scales require FP32 or E8M0 storage")
    grouped = x.float().unflatten(-1, (-1, block_size))
    scales = grouped.abs().amax(-1).clamp_min(1e-4) * (1. / 448)
    if scale_fmt is not None:
        bits = scales.view(torch.int32)
        exponent = ((bits >> 23) & 255) + ((bits & ((1 << 23) - 1)) != 0)
        scales = (exponent << 23).view(torch.float32)
    values = (grouped / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((values.float() * scales.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return values.flatten(-2).contiguous(), scales.to(scale_dtype).contiguous()


def install_native_row_padding(reference) -> dict[str, object]:
    """Install stable FP8 generation and native GEMM/FP4 row padding once."""
    if getattr(reference, "_archlab_row_padding", False):
        raise ValueError("native row padding already installed")
    original_linear = reference.linear
    original_fp4 = reference.fp4_act_quant

    def padded_rows(x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Flatten x [...,features] and pad its row count to a multiple of 32."""
        flat = x.reshape(-1, x.shape[-1])
        rows = flat.shape[0]
        return F.pad(flat, (0, 0, 0, -rows % 32)), rows

    def linear(x, weight, bias=None):
        """Map x [...,in] and weight [out,in] (FP4 packed) to [...,out]."""
        if (weight.dtype not in (torch.float8_e4m3fn, getattr(torch, "float4_e2m1fn_x2", None))
                or x.numel() // x.shape[-1] % 32 == 0):
            return original_linear(x, weight, bias)
        padded, rows = padded_rows(x)
        result = original_linear(padded, weight, bias)
        return result[:rows].reshape(*x.shape[:-1], result.shape[-1])

    def fp8(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        return _fp8_quantize(x, block_size, scale_fmt, scale_dtype, inplace)

    def fp4(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        """Quantize x [...,features] into packed FP4 or native in-place BF16."""
        if x.numel() // x.shape[-1] % 32 == 0:
            return original_fp4(x, block_size, inplace, scale_dtype)
        padded, rows = padded_rows(x)
        result = original_fp4(padded, block_size, inplace, scale_dtype)
        if inplace:
            x.copy_(result[:rows].reshape_as(x))
            return x
        quantized, scales = result
        return (quantized[:rows].reshape(*x.shape[:-1], quantized.shape[-1]),
                scales[:rows].reshape(*x.shape[:-1], scales.shape[-1]))

    reference.linear, reference.act_quant, reference.fp4_act_quant = linear, fp8, fp4
    reference._archlab_row_padding = True
    evidence = {"native_quantizer_row_tile": 32, "tail_rows_padded": True,
                "fp8_activation_backend": "pytorch-packed-fp8-native-equations",
                "fp8_stabilization_reason": "native FP8 values were nondeterministic/nonfinite for identical finite inputs",
                "fp4_activation_backend": "native-with-row-padding",
                "native_gemms_retained": True, "reference_files_modified": False}
    reference._archlab_native_quantization = evidence
    return evidence
