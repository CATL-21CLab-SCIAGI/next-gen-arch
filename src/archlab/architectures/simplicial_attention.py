"""Causal local 2-simplicial attention: the independent numerical oracle.

GPU execution uses the TileLang joint kernel in ``tilelang_attention``.

Tensor layout is [batch, sequence, heads, head_dim]. Two KV groups need not
imply model parallelism: all groups are computed locally. This leaf mechanism
does not choose projections, normalization, gates, or positional embeddings.
Those belong to the experiment's enclosing attention module.
"""

from __future__ import annotations

import math
from numbers import Real

import torch


def validate_inputs(q, k1, k2, v1, v2, short_window, long_window, *, allow_value_dim=False):
    if q.ndim != 4 or any(x.ndim != 4 for x in (k1, k2, v1, v2)):
        raise ValueError("expected [batch, sequence, heads, head_dim]")
    if min(short_window, long_window) < 1 or short_window > long_window:
        raise ValueError("require 1 <= short_window <= long_window")
    b, n, heads, d = q.shape
    if min(b, n, heads, d) < 1:
        raise ValueError("empty attention dimensions are unsupported")
    if k2.shape != k1.shape or v2.shape != v1.shape or v1.shape[:3] != k1.shape[:3]:
        raise ValueError("both key/value branches must have matching batch, sequence and heads")
    if v1.shape[-1] < 1 or (not allow_value_dim and v1.shape[-1] != d):
        raise ValueError("unsupported value head dimension")
    if k1.shape[2] < 1 or k1.shape[:2] != (b, n) or k1.shape[-1] != d or heads % k1.shape[2]:
        raise ValueError("incompatible grouped-query geometry")
    if any(x.dtype != q.dtype or x.device != q.device for x in (k1, k2, v1, v2)):
        raise ValueError("all inputs must have the same dtype and device")


def _score_scale(scale, dim):
    if scale is None:
        return 1 / math.sqrt(dim)
    if not isinstance(scale, Real) or not math.isfinite(scale):
        raise ValueError("scale must be a finite real number")
    return float(scale)


def reference_simplicial(q, k1, k2, v1, v2, short_window, long_window, *,
                        query_positions=None, scale=None):
    """Tiny-sequence oracle; explicitly joint-normalizes all causal key pairs.

    Intentionally slow, but never allocates a full sequence-cubed tensor.
    Includes diagonal pairs and the query's own position on both key axes.
    No extra KV bias, dropout, or positional transformation is implicit.
    Values may have a different last dimension from the score operands.
    ``scale`` defaults to 1 / sqrt(score_dim); it is not a trainable tensor.
    """
    validate_inputs(q, k1, k2, v1, v2, short_window, long_window, allow_value_dim=True)
    scale = _score_scale(scale, q.shape[-1])
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
        probabilities = (scores * scale).flatten(-2).softmax(-1)
        probabilities = probabilities.reshape_as(scores)
        outputs.append(torch.einsum("bhjk,bjhd,bkhd->bhd", probabilities, v1[:, j], v2[:, k]))
    return torch.stack(outputs, dim=1)
