"""Differentiable scalar/tensor mechanisms from the released V4.1 reference.

Equations follow inference/kernel.py at model revision
df42c109f1defefcbfcedbe7d905718a12266e40. These functions do not choose a training
backend or claim full-model support. In particular, mHC coefficient ordering
and the number of Sinkhorn normalizations must not be changed to V4 defaults.
"""

from __future__ import annotations

import torch


def hc_split_sinkhorn(mixes, scale, base, streams=4, iterations=20, eps=1e-6):
    """Return pre/post/comb coefficients without a forward-only opaque kernel."""
    if streams < 1 or iterations < 1 or eps <= 0:
        raise ValueError("positive stream count, iterations and epsilon are required")
    if mixes.shape[-1] != (streams + 2) * streams:
        raise ValueError("wrong mHC projection width")
    if scale.shape != (3,) or base.shape != (mixes.shape[-1],):
        raise ValueError("wrong mHC scale/base shape")
    # FP64 is supported solely for finite-difference tests; production is FP32.
    dtype = torch.float64 if mixes.dtype == torch.float64 else torch.float32
    mixes, scale, base = (t.to(dtype) for t in (mixes, scale, base))
    pre = (mixes[..., :streams] * scale[0] + base[:streams]).sigmoid() + eps
    post = 2 * (mixes[..., streams:2 * streams] * scale[1] + base[streams:2 * streams]).sigmoid()
    comb = (mixes[..., 2 * streams:] * scale[2] + base[2 * streams:]).unflatten(-1, (streams, streams))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def dequantize_frozen_weight(weight, scale):
    """Read-only FP8-block32 / packed-FP4-group32 decoding, returning BF16.

    FP4 nibble order and codebook follow the official inference/convert.py.
    This is for a frozen-weight dInput oracle/bridge, not a training quantizer.
    Large weights should be sliced by the caller; do not decode Engram tables
    wholesale. FP4 is accepted as the on-disk I8 or the runtime packed dtype.
    """
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("expected a matrix and its two-dimensional scales")
    if weight.dtype == torch.float8_e4m3fn:
        rows, columns = weight.shape
        if columns % 32 or scale.shape != ((rows + 31) // 32, columns // 32):
            raise ValueError("expected native FP8 block-32 scales")
        factors = scale.float().repeat_interleave(32, 0)[:rows].repeat_interleave(32, 1)
        return (weight.float() * factors).bfloat16()
    packed_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if weight.dtype not in (torch.int8, packed_dtype):
        raise ValueError("expected native FP8 or packed FP4 frozen weights")
    rows, packed_columns = weight.shape
    if packed_columns % 16 or scale.shape != (rows, packed_columns // 16):
        raise ValueError("expected native FP4 group-32 scales")
    codebook = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                            device=weight.device, dtype=torch.float32)
    bits = weight.view(torch.uint8)
    values = torch.stack((codebook[(bits & 15).long()], codebook[(bits >> 4).long()]), -1).flatten(-2)
    return (values * scale.float().repeat_interleave(32, -1)).bfloat16()
