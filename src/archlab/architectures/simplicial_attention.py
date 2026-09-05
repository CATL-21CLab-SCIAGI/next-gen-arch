"""Causal local 2-simplicial attention and an independent numerical oracle.

Tensor layout is [batch, sequence, heads, head_dim]. Two KV groups need not
imply model parallelism: all groups are computed locally. This leaf mechanism
does not choose projections, normalization, gates, or positional embeddings.
Those belong to the experiment's enclosing attention module.
"""

from __future__ import annotations

import math

import torch


def validate_inputs(q, k1, k2, v1, v2, short_window, long_window):
    if q.ndim != 4 or any(x.ndim != 4 for x in (k1, k2, v1, v2)):
        raise ValueError("expected [batch, sequence, heads, head_dim]")
    if min(short_window, long_window) < 1 or short_window > long_window:
        raise ValueError("require 1 <= short_window <= long_window")
    b, n, heads, d = q.shape
    if min(b, n, heads, d) < 1:
        raise ValueError("empty attention dimensions are unsupported")
    if any(x.shape != k1.shape for x in (k2, v1, v2)):
        raise ValueError("both key/value branches must have matching shapes")
    if k1.shape[2] < 1 or k1.shape[:2] != (b, n) or k1.shape[-1] != d or heads % k1.shape[2]:
        raise ValueError("incompatible grouped-query geometry")
    if any(x.dtype != q.dtype or x.device != q.device for x in (k1, k2, v1, v2)):
        raise ValueError("all inputs must have the same dtype and device")


def reference_simplicial(q, k1, k2, v1, v2, short_window, long_window, *, query_positions=None):
    """Tiny-sequence oracle; explicitly joint-normalizes all causal key pairs.

    Intentionally slow, but never allocates a full sequence-cubed tensor.
    Includes diagonal pairs and the query's own position on both key axes.
    No extra KV bias, dropout, or positional transformation is implicit.
    """
    validate_inputs(q, k1, k2, v1, v2, short_window, long_window)
    groups = q.shape[2] // k1.shape[2]
    k1, k2, v1, v2 = [x.repeat_interleave(groups, dim=2) for x in (k1, k2, v1, v2)]
    outputs = []
    positions = range(q.shape[1]) if query_positions is None else query_positions
    for i in positions:
        if not 0 <= i < q.shape[1]:
            raise ValueError("oracle query position outside sequence")
        j = slice(max(0, i - short_window + 1), i + 1)
        k = slice(max(0, i - long_window + 1), i + 1)
        scores = torch.einsum("bhd,bjhd,bkhd->bhjk", q[:, i], k1[:, j], k2[:, k])
        probabilities = (scores / math.sqrt(q.shape[-1])).flatten(-2).softmax(-1)
        probabilities = probabilities.reshape_as(scores)
        outputs.append(torch.einsum("bhjk,bjhd,bkhd->bhd", probabilities, v1[:, j], v2[:, k]))
    return torch.stack(outputs, dim=1)


def simplicial_attention(q, k1, k2, v1, v2, short_window, long_window):
    """Stock-Triton local-GQA kernel; GPU-only, forward and backward.

    Backward uses FP32 atomic accumulation for shared K/V gradients. It is
    numerically checked but not bitwise deterministic. No runtime is patched.
    """
    validate_inputs(q, k1, k2, v1, v2, short_window, long_window)
    if q.device.type != "cuda" or q.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("kernel supports CUDA float32/bfloat16")
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError("unsupported head dimension")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("simplicial backward uses nondeterministic FP32 atomic accumulation")
    from archlab.architectures.simplicial_kernels import SimplicialFunction

    return SimplicialFunction.apply(q, k1, k2, v1, v2, short_window, long_window)
