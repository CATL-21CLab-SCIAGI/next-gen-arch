"""Triadic Gated DeltaNet: TileLang GPU kernels and an independent oracle.

The joint key is k2_t outer k_t at one token, with learned feature slices E.
This differs from LinSimp's independent long token j and short anchor c: no
random features or token-pair softmax are used here. Slice decay precedes one
joint delta correction, so the erase term reads all slices together.

GPU execution uses the chunked kernels of ``tilelang_delta``. The small
differentiable recurrence below is a numerical oracle, not a GPU training
implementation. Every batch row is one document.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

SUPPORTED_FEATURE_DIMS = (1, 2, 4, 8, 12, 16)


def _validate(q, k, v, k2, q2, g, beta):
    if q.ndim != 4 or min(q.shape) < 1:
        raise ValueError("q must be nonempty [batch, sequence, heads, key_dim]")
    batch, length, heads, _ = q.shape
    if k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("q/k must match and v must share batch/sequence/heads")
    if k2.ndim != 4 or k2.shape[:3] != (batch, length, heads) or k2.shape[-1] < 1:
        raise ValueError("k2 must be [batch, sequence, heads, feature_dim]")
    if q2.shape != k2.shape or g.shape != k2.shape or beta.shape != q.shape[:3]:
        raise ValueError("q2/g must match k2 and beta must be [batch, sequence, heads]")
    if any(t.device != q.device or not t.is_floating_point() for t in (q, k, v, k2, q2, g, beta)):
        raise ValueError("all inputs must be floating tensors on one device")


def triadic_gdn_reference(q, k, v, k2, q2, g, beta, *, scale=None):
    """Direct per-token slice recurrence, preserving FP64 inputs for the oracle.

    S_e <- exp(g_e) S_e
    residual <- v - sum_e k2_e k^T S_e
    S_e <- S_e + beta k2_e k residual^T
    o <- scale sum_e q2_e q^T S_e

    Unlike separate E delta updates, one residual erases the joint memory.
    """
    _validate(q, k, v, k2, q2, g, beta)
    batch, length, heads, dim = q.shape
    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (q, k, v, k2, q2, g, beta))
        else torch.float32
    )
    output_dtype = q.dtype
    q, k, v, k2, q2, g, beta = [t.to(dtype) for t in (q, k, v, k2, q2, g, beta)]
    multiplier = dim**-0.5 if scale is None else scale
    if not math.isfinite(multiplier):
        raise ValueError("read scale must be finite")
    state = q.new_zeros(batch, heads, k2.shape[-1], dim, v.shape[-1])
    outputs = []
    for index in range(length):
        state = state * g[:, index].exp()[..., None, None]
        predicted = torch.einsum("bhd,bhedv,bhe->bhv", k[:, index], state, k2[:, index])
        residual = v[:, index] - predicted
        state = state + torch.einsum(
            "bh,bhe,bhd,bhv->bhedv", beta[:, index], k2[:, index], k[:, index], residual
        )
        outputs.append(
            multiplier * torch.einsum("bhd,bhedv,bhe->bhv", q[:, index], state, q2[:, index])
        )
    return torch.stack(outputs, dim=1).to(output_dtype)


def triadic_gdn_attention(q, k, v, k2, q2, g, beta, *, backend="tilelang", scale=None):
    """Explicit backend dispatch; production never silently falls back to Torch."""
    _validate(q, k, v, k2, q2, g, beta)
    if backend == "reference":
        return triadic_gdn_reference(q, k, v, k2, q2, g, beta, scale=scale)
    if backend != "tilelang":
        raise ValueError("Triadic backend must be tilelang or reference")
    if k2.shape[-1] not in SUPPORTED_FEATURE_DIMS:
        raise ValueError("unsupported Triadic feature dimension")
    from archlab.architectures.tilelang_delta import tilelang_triadic_gdn

    return tilelang_triadic_gdn(
        q, k, v, k2, q2, g, beta, scale=q.shape[-1] ** -0.5 if scale is None else scale)


def causal_conv(x, weight, act_channels):
    """Depthwise causal convolution over [B,T,C]; SiLU on the first act_channels."""
    if x.ndim != 3 or weight.ndim != 2 or weight.shape[0] != x.shape[-1]:
        raise ValueError("convolution needs [B,T,C] inputs and [C,W] weights")
    if not 0 <= act_channels <= x.shape[-1] or weight.shape[1] < 1:
        raise ValueError("invalid activation channels or convolution width")
    padded = F.pad(x.transpose(1, 2), (weight.shape[1] - 1, 0))
    mixed = F.conv1d(padded, weight.unsqueeze(1), groups=x.shape[-1]).transpose(1, 2)
    return torch.cat((F.silu(mixed[..., :act_channels]), mixed[..., act_channels:]), -1)
