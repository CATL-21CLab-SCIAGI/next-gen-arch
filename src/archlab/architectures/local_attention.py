"""Causal single-key attention: an independent oracle and container kernel.

Windows include the current token. This leaf owns no projections, positional
encoding, normalization, dropout, or trainer dependency.
"""

from __future__ import annotations

import math

import torch


def _validate(q, k, v, window):
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("expected [batch, sequence, heads, head_dim]")
    if type(window) is not int or window < 1 or min(q.shape) < 1:
        raise ValueError("attention dimensions and window must be positive")
    if (k.shape != v.shape or k.shape[:2] != q.shape[:2]
            or k.shape[-1] != q.shape[-1] or k.shape[2] < 1
            or q.shape[2] % k.shape[2]):
        raise ValueError("incompatible grouped-query attention shapes")
    if any(x.dtype != q.dtype or x.device != q.device for x in (k, v)):
        raise ValueError("attention inputs must have the same dtype and device")


def reference_local_attention(q, k, v, window, *, query_positions=None):
    """Explicit FP32/FP64 softmax equation, independent of FlashAttention."""
    _validate(q, k, v, window)
    repeats = q.shape[2] // k.shape[2]
    k, v = (x.repeat_interleave(repeats, dim=2) for x in (k, v))
    positions = range(q.shape[1]) if query_positions is None else query_positions
    outputs = []
    for index in positions:
        if not 0 <= index < q.shape[1]:
            raise ValueError("query position outside sequence")
        keys = slice(max(0, index - window + 1), index + 1)
        scores = torch.einsum("bhd,bkhd->bhk", q[:, index], k[:, keys]) / math.sqrt(q.shape[-1])
        outputs.append(torch.einsum("bhk,bkhd->bhd", scores.softmax(-1), v[:, keys]))
    return torch.stack(outputs, dim=1)


def deterministic_local_attention(q, k, v, window):
    """Use the unchanged container FlashAttention kernel and stable backward.

    BF16 projections feed the standard BF16 attention API; its score/softmax
    accumulation uses FP32. This precision is explicit in the control contract.
    No dropout, rotary transform, or attention bias is added.
    """
    _validate(q, k, v, window)
    if q.device.type != "cuda" or q.dtype != torch.bfloat16:
        raise ValueError("the production control requires CUDA BF16 projections")
    from flash_attn import flash_attn_func

    return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=1 / math.sqrt(q.shape[-1]),
                           causal=True, window_size=(window - 1, 0), deterministic=True)
