"""Linearized 2-simplicial attention (LinSimp) and one-mode linear control.

Implements the one-mode positive-random-feature reduction from
arXiv:2608.09307, with a short explicit anchor window on the second token
mode. The linear control uses the same feature map and prefix state without
the second mode, so only the linearized trilinear part is the experimental
factor.
"""

from __future__ import annotations

import math

import torch


def _validate_pair(q, k, v):
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("expected [batch, sequence, heads, head_dim]")
    if min(q.shape) < 1:
        raise ValueError("empty attention input")
    if (k.shape != v.shape or k.shape[:2] != q.shape[:2] or k.shape[-1] != q.shape[-1]
            or k.shape[2] < 1 or q.shape[2] % k.shape[2]):
        raise ValueError("incompatible grouped-query attention shapes")
    if any(x.dtype != q.dtype or x.device != q.device for x in (k, v)):
        raise ValueError("attention inputs must share dtype and device")


def _expand_kv(q, *tensors):
    repeats = q.shape[2] // tensors[0].shape[2]
    if repeats == 1:
        return tensors
    return tuple(t.repeat_interleave(repeats, dim=2) for t in tensors)


def orthogonal_feature_bank(heads, head_dim, rank, *, device, dtype, generator=None, seed=0):
    """Positive orthogonal random features with Gaussian marginals per row."""
    if type(rank) is not int or rank < 1 or rank % head_dim:
        raise ValueError("feature rank must be a positive multiple of head_dim")
    cpu_gen = generator
    if cpu_gen is None:
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(seed)
    blocks = []
    for _ in range(rank // head_dim):
        gaussian = torch.randn(heads, head_dim, head_dim, dtype=torch.float32, generator=cpu_gen)
        q_factor, r_factor = torch.linalg.qr(gaussian, mode="reduced")
        # QR fixes column signs by convention; undo that convention to obtain
        # Haar directions. Each row needs a chi_D radius, not abs(N(0,1)).
        signs = r_factor.diagonal(dim1=-2, dim2=-1).sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        directions = (q_factor * signs.unsqueeze(-2)).transpose(-1, -2)
        norms = torch.randn(heads, head_dim, head_dim, dtype=torch.float32,
                            generator=cpu_gen).norm(dim=-1)
        blocks.append(directions * norms.unsqueeze(-1))
    return torch.cat(blocks, dim=1).to(device=device, dtype=dtype)


def positive_features(x, omega):
    """phi(x) = m^{-1/2} exp(Omega x - ||x||^2 / 2), elementwise exp."""
    squared = x.float().square().sum(-1, keepdim=True) * 0.5
    projected = torch.einsum("...hd,hmd->...hm", x.float(), omega.float())
    return (projected - squared).exp().mul(omega.shape[-2] ** -0.5)


def _query_features(x, omega, *, anchor_axis=None, valid=None):
    """One scale per causal query/head, shared by all anchors and features.

    Scaling cancels between the numerator and denominator. Normalizing each
    anchor separately would change the operator; including padded anchors in
    the maximum can underflow every valid anchor at large temperatures.
    """
    logs = torch.einsum("...hd,hmd->...hm", x.float(), omega.float())
    logs = logs - x.float().square().sum(-1, keepdim=True) * 0.5
    if valid is not None:
        logs = logs.masked_fill(~valid[..., None, None], -torch.inf)
    axes = (-1,) if anchor_axis is None else (anchor_axis, -1)
    shift = logs.amax(dim=axes, keepdim=True).detach()
    return (logs - shift).exp().mul(omega.shape[-2] ** -0.5)


def _l2_normalize(x, eps=1e-6):
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(eps)


def _prefix_states(k_feat, values):
    # k_feat: [B,T,H,m], values: [B,T,H,D] -> inclusive prefix M,a
    outer = k_feat.unsqueeze(-1) * values.unsqueeze(-2)
    return outer.cumsum(1), k_feat.cumsum(1)


def linear_rf_attention(q, k, v, omega, *, temperature=None):
    """Causal one-mode random-feature linear attention (LinSimp without anchors)."""
    _validate_pair(q, k, v)
    k, v = _expand_kv(q, k, v)
    _, _, heads, dim = q.shape
    if omega.shape != (heads, omega.shape[1], dim):
        raise ValueError("feature bank must be [heads, rank, head_dim]")
    tau = math.sqrt(dim) if temperature is None else temperature
    q_feat = _query_features(_l2_normalize(q) * tau, omega)
    k_feat = positive_features(_l2_normalize(k), omega)
    state, normalizer = _prefix_states(k_feat, v.float())
    numerator = torch.einsum("bthm,bthmd->bthd", q_feat, state)
    denom = torch.einsum("bthm,bthm->bth", q_feat, normalizer).unsqueeze(-1).clamp_min(1e-30)
    return (numerator / denom).to(dtype=q.dtype)


def _linsimp_attention_loop(q, k, r, v, u, omega, *, window, temperature=None):
    """Reference loop implementation; used only for parity tests."""
    _validate_pair(q, k, v)
    if r.shape != k.shape or u.shape != v.shape:
        raise ValueError("r/u must match k/v geometry")
    if type(window) is not int or window < 1:
        raise ValueError("anchor window must be a positive integer")
    k, r, v, u = _expand_kv(q, k, r, v, u)
    _, _, heads, dim = q.shape
    if omega.shape != (heads, omega.shape[1], dim):
        raise ValueError("feature bank must be [heads, rank, head_dim]")
    tau = math.sqrt(dim) if temperature is None else temperature
    qn = _l2_normalize(q) * tau
    rn = _l2_normalize(r)
    k_feat = positive_features(_l2_normalize(k), omega)
    state, normalizer = _prefix_states(k_feat, v.float())
    anchors_u = u.float()
    outputs = []
    for index in range(q.shape[1]):
        start = max(0, index - window + 1)
        composite = qn[:, index].unsqueeze(1) * rn[:, start : index + 1]
        feature = _query_features(composite, omega, anchor_axis=1)
        numerator = torch.einsum("bwhm,bhmd->bwhd", feature, state[:, index]) * anchors_u[
            :, start : index + 1
        ]
        denom = torch.einsum("bwhm,bhm->bwh", feature, normalizer[:, index])
        outputs.append(numerator.sum(1) / denom.sum(1).unsqueeze(-1).clamp_min(1e-30))
    return torch.stack(outputs, dim=1).to(dtype=q.dtype)


def linsimp_attention(q, k, r, v, u, omega, *, window, temperature=None):
    """Causal LinSimp: global RF state on j, explicit short window on c.

    Vectorized over the sequence axis so production 2k contexts stay linear-time
    in practice; math matches the short-window one-mode reduction in arXiv:2608.09307.
    """
    _validate_pair(q, k, v)
    if r.shape != k.shape or u.shape != v.shape:
        raise ValueError("r/u must match k/v geometry")
    if type(window) is not int or window < 1:
        raise ValueError("anchor window must be a positive integer")
    k, r, v, u = _expand_kv(q, k, r, v, u)
    batch, length, heads, dim = q.shape
    if omega.shape != (heads, omega.shape[1], dim):
        raise ValueError("feature bank must be [heads, rank, head_dim]")
    tau = math.sqrt(dim) if temperature is None else temperature
    qn = _l2_normalize(q) * tau
    rn = _l2_normalize(r)
    k_feat = positive_features(_l2_normalize(k), omega)
    state, normalizer = _prefix_states(k_feat, v.float())
    anchors_u = u.float()
    pad = window - 1
    # Left-pad so each position's causal window is a fixed-width slice.
    rn_pad = torch.nn.functional.pad(rn, (0, 0, 0, 0, pad, 0))
    u_pad = torch.nn.functional.pad(anchors_u, (0, 0, 0, 0, pad, 0))
    # [B, T, H, D, W] -> [B, T, W, H, D]
    rn_win = rn_pad.unfold(1, window, 1).permute(0, 1, 4, 2, 3).contiguous()
    u_win = u_pad.unfold(1, window, 1).permute(0, 1, 4, 2, 3).contiguous()
    local = torch.arange(window, device=q.device)
    time = torch.arange(length, device=q.device)
    valid = local.view(1, 1, window) >= (pad - time).clamp_min(0).view(1, length, 1)
    composite = qn.unsqueeze(2) * rn_win
    feature = _query_features(composite, omega, anchor_axis=2, valid=valid)
    numerator = torch.einsum("btwhm,bthmd->btwhd", feature, state) * u_win
    denom = torch.einsum("btwhm,bthm->btwh", feature, normalizer)
    weight = valid.to(dtype=numerator.dtype)
    numerator = numerator * weight.unsqueeze(-1).unsqueeze(-1)
    denom = denom * weight.unsqueeze(-1)
    return (numerator.sum(2) / denom.sum(2).unsqueeze(-1).clamp_min(1e-30)).to(dtype=q.dtype)
