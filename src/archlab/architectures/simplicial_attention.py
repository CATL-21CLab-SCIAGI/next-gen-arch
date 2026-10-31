"""Causal local 2-simplicial attention and an independent numerical oracle.

Tensor layout is [batch, sequence, heads, head_dim]. Two KV groups need not
imply model parallelism: all groups are computed locally. This leaf mechanism
does not choose projections, normalization, gates, or positional embeddings.
Those belong to the experiment's enclosing attention module.
"""

from __future__ import annotations

import math
from numbers import Real

import torch


def validate_inputs(q, k1, k2, v1, v2, short_window, long_window, *,
                    allow_scaled_query=False, allow_value_dim=False):
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
    dtype = (
        k1.dtype
        if allow_scaled_query and q.dtype == torch.float32 and k1.dtype == torch.bfloat16
        else q.dtype
    )
    if any(x.dtype != dtype or x.device != q.device for x in (k1, k2, v1, v2)):
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


def simplicial_attention(q, k1, k2, v1, v2, short_window, long_window, *, scale=None):
    """Stock-Triton local-GQA kernel; GPU-only, forward and backward.

    Backward uses FP32 atomic accumulation for shared K/V gradients. It is
    numerically checked but not bitwise deterministic. No runtime is patched.
    Score and value widths are independent; ``scale`` defaults to
    1 / sqrt(score_dim), preserving the original equal-width API.
    """
    validate_inputs(q, k1, k2, v1, v2, short_window, long_window, allow_value_dim=True)
    scale = _score_scale(scale, q.shape[-1])
    if q.device.type != "cuda" or q.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("kernel supports CUDA float32/bfloat16")
    if q.shape[-1] not in (16, 32, 64, 128, 256, 512):
        raise ValueError("unsupported score head dimension")
    if v1.shape[-1] not in (16, 32, 64, 128, 256):
        raise ValueError("unsupported value head dimension")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("simplicial backward uses nondeterministic FP32 atomic accumulation")
    from archlab.architectures.simplicial_kernels import SimplicialFunction

    return SimplicialFunction.apply(q, k1, k2, v1, v2, short_window, long_window, scale)


def _determinant_shifts(dim: int, device):
    """Within-chunk cyclic shifts for the Sarrus expansion over ``dim // 3`` chunks of 3 coordinates."""
    if dim < 3:
        raise ValueError("determinant scores require at least 3 coordinates")
    used = 3 * (dim // 3)
    base = torch.arange(0, used, 3, device=device)[:, None]
    shift1 = (base + torch.tensor([1, 2, 0], device=device)).flatten()  # (b2, b3, b1)
    shift2 = (base + torch.tensor([2, 0, 1], device=device)).flatten()  # (b3, b1, b2)
    return used, shift1, shift2


def determinant_operands(q, k1, k2):
    """Elementwise-trilinear operands whose logit equals sum over chunks of det([q, k1, k2]).

    Sarrus: det(a, b, c) = <a, s1(b), s2(c)> - <a, s2(b), s1(c)> per 3-chunk. Concatenating both
    terms gives one elementwise trilinear form over 2 * used dimensions, zero-padded to a power of
    two for the kernel. A trailing ``dim % 3`` coordinates take no part in the score.
    """
    used, shift1, shift2 = _determinant_shifts(q.shape[-1], q.device)
    if k1.shape[-1] != q.shape[-1] or k2.shape[-1] != q.shape[-1]:
        raise ValueError("determinant operands must have matching score widths")
    width = max(16, 1 << (2 * used - 1).bit_length())
    pad = width - 2 * used

    def assemble(*parts):
        return torch.nn.functional.pad(torch.cat(parts, dim=-1), (0, pad))

    q_used = q[..., :used]
    return (assemble(q_used, -q_used), assemble(k1[..., shift1], k1[..., shift2]),
            assemble(k2[..., shift2], k2[..., shift1]))


def determinant_simplicial_attention(q, k1, k2, v1, v2, short_window, long_window, *, scale=None):
    """Causal attention with signed 3D chunk determinants as scores.

    Implements the sum in Fast and Simplex, sec. 5, eq. 9
    (https://arxiv.org/html/2507.02754v1#S5), with an explicit temperature.
    Our default scale is 1 / sqrt(original score_dim); eq. 9 is unscaled.
    This sum is invariant to a common proper rotation of q/k1/k2 within
    each 3D chunk, not to arbitrary rotations mixing chunks or reflections.
    Trailing score_dim % 3 coordinates are excluded; value coordinates are
    all retained. No positional rotation is applied by this leaf operation.

    Sarrus expansion doubles the used score width and pads to a power of two.
    Value width is unchanged. Autograd propagates through the expansion.
    """
    validate_inputs(q, k1, k2, v1, v2, short_window, long_window, allow_value_dim=True)
    scale = _score_scale(scale, q.shape[-1])
    qs, k1s, k2s = determinant_operands(q, k1, k2)
    return simplicial_attention(qs, k1s, k2s, v1, v2, short_window, long_window, scale=scale)


def reference_determinant_simplicial(q, k1, k2, v1, v2, short_window, long_window, *,
                                    query_positions=None, scale=None):
    """Independent torch.linalg.det oracle for the determinant attention contract.

    Use float32 or float64 inputs; torch.linalg.det does not support BF16.
    ``query_positions`` bounds reference work when qualifying longer windows.
    """
    validate_inputs(q, k1, k2, v1, v2, short_window, long_window, allow_value_dim=True)
    if q.shape[-1] < 3:
        raise ValueError("determinant scores require at least 3 coordinates")
    if q.dtype not in (torch.float32, torch.float64):
        raise ValueError("determinant oracle requires float32/float64")
    scale = _score_scale(scale, q.shape[-1])
    groups = q.shape[2] // k1.shape[2]
    k1, k2, v1, v2 = [x.repeat_interleave(groups, dim=2) for x in (k1, k2, v1, v2)]
    used = 3 * (q.shape[-1] // 3)
    qc, k1c, k2c = [x[..., :used].unflatten(-1, (used // 3, 3)) for x in (q, k1, k2)]
    outputs = []
    positions = range(q.shape[1]) if query_positions is None else query_positions
    for i in positions:
        if not 0 <= i < q.shape[1]:
            raise ValueError("oracle query position outside sequence")
        j = slice(max(0, i - short_window + 1), i + 1)
        k = slice(max(0, i - long_window + 1), i + 1)
        a = qc[:, i][:, None, None].expand(-1, k1c[:, j].shape[1], k2c[:, k].shape[1], -1, -1, -1)
        b = k1c[:, j][:, :, None].expand_as(a)
        c = k2c[:, k][:, None].expand_as(a)
        scores = torch.linalg.det(torch.stack((a, b, c), dim=-2)).sum(-1).permute(0, 3, 1, 2)
        probabilities = (scores * scale).flatten(-2).softmax(-1).reshape_as(scores)
        outputs.append(torch.einsum("bhjk,bjhd,bkhd->bhd", probabilities, v1[:, j], v2[:, k]))
    return torch.stack(outputs, dim=1)


def oriented_determinant_simplicial_attention(q, k1, k2, v1, v2, window_1, window_2, *,
                                              scale=None, backend="kernel"):
    """Causal determinant 2-simplicial attention supporting any positive (window_1, window_2).

    ``window_1`` bounds the (k1, v1) causal prefix and ``window_2`` bounds (k2, v2).
    When ``window_1 > window_2``, uses the exact 3x3 determinant antisymmetry
    ``det([-q, k2, k1]) == det([q, k1, k2])`` and commutativity ``v2 * v1 == v1 * v2``
    so the shorter window always occupies the kernel's outer loop and the longer
    window occupies its Tensor Core tiled loop.
    """
    if type(window_1) is not int or type(window_2) is not int or min(window_1, window_2) < 1:
        raise ValueError("window_1 and window_2 must be positive integers")
    if backend not in ("kernel", "reference"):
        raise ValueError("backend must be kernel or reference")
    core = determinant_simplicial_attention if backend == "kernel" else reference_determinant_simplicial
    if window_1 <= window_2:
        return core(q, k1, k2, v1, v2, window_1, window_2, scale=scale)
    return core(-q, k2, k1, v2, v1, window_2, window_1, scale=scale)
