"""Exact global 2-simplicial attention as container cuDNN attention over virtual queries.

Row (i, t) is the virtual query q_i * k1_{i-t}; it attends query i's causal long
keys as one head of an ordinary GQA layer with HQ * w query heads. Its log
normalizer weights the joint softmax over t. Backward passes cuDNN the joint
softmax delta through a corrected output O' = o + lambda * dO with
dO . O' = weight * (do . out), so every input gradient is the joint-softmax
gradient. Layout is [batch, sequence, heads, 128].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_forward = torch.ops.aten._scaled_dot_product_cudnn_attention
_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward


def _shifted(x, w):
    """[B, N, H, D] -> view [B, N, H, w, D] with slot t holding x[:, i - t] (zero before 0)."""
    return F.pad(x, (0, 0, 0, 0, w - 1, 0)).unfold(1, w, 1).flip(-1).transpose(-1, -2)


def _unshifted(x):
    """Adjoint of ``_shifted``: [B, N, H, w, D] -> [B, N, H, D]."""
    out = x[:, :, :, 0].clone()
    for t in range(1, x.shape[3]):
        out[:, : x.shape[1] - t] += x[:, t:, :, t]
    return out


def _virtual(q, k1, w):
    """Virtual queries [B, HQ*w, N, D]; head h*w + t uses KV head h // (HQ // HK)."""
    b, n, hq, d = q.shape
    hk = k1.shape[2]
    a = q.view(b, n, hk, hq // hk, 1, d) * _shifted(k1, w)[:, :, :, None]
    return a.reshape(b, n, hq * w, d).transpose(1, 2)


class _VirtualSimplicial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k1, k2, v1, v2, w, scale):
        b, n, hq, d = q.shape
        hk = k1.shape[2]
        g = hq // hk
        o, lse, *meta = _forward(_virtual(q, k1, w), k2.transpose(1, 2), v2.transpose(1, 2), None,
                                 True, 0.0, True, False, scale=scale)
        o = o.transpose(1, 2).reshape(b, n, hk, g, w, d)
        lse = lse.reshape(b, hq, w, n).permute(0, 3, 1, 2)
        invalid = torch.arange(w, device=q.device)[None, :] > torch.arange(n, device=q.device)[:, None]
        weights = lse.masked_fill(invalid[None, :, None], float("-inf")).softmax(-1)
        weights = weights.reshape(b, n, hk, g, w)
        v1s = _shifted(v1, w)[:, :, :, None].float()
        out = (weights[..., None] * v1s * o.float()).sum(4).reshape(b, n, hq, d)
        ctx.save_for_backward(q, k1, k2, v1, v2, o, lse, weights, out)
        ctx.meta, ctx.cfg = meta[:6], (w, scale)
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, grad):
        q, k1, k2, v1, v2, o, lse, weights, out = ctx.saved_tensors
        w, scale = ctx.cfg
        cq, ck, mq, mk, seed, offset = ctx.meta
        b, n, hq, d = q.shape
        hk = k1.shape[2]
        g = hq // hk
        grad = grad.float().view(b, n, hk, g, 1, d)
        weighted = weights[..., None] * grad
        do = (weighted * _shifted(v1, w)[:, :, :, None].float()).to(q.dtype)
        dof, of = do.float(), o.float()
        target = weights * (grad[:, :, :, :, 0] * out.view(b, n, hk, g, d)).sum(-1)[..., None]
        norm = (dof * dof).sum(-1)
        lam = torch.where(norm > 0, (target - (dof * of).sum(-1)) / norm.clamp_min(1e-30), 0)
        corrected = (of + lam[..., None] * dof).to(q.dtype)
        rows = lambda x: x.reshape(b, n, hq * w, d).transpose(1, 2)
        da, dk2, dv2 = _backward(
            rows(do), _virtual(q, k1, w), k2.transpose(1, 2), v2.transpose(1, 2), rows(corrected),
            lse.permute(0, 2, 3, 1).reshape(b, hq * w, n, 1), seed, offset, None,
            cq, ck, mq, mk, 0.0, True, scale=scale,
        )
        da = da.transpose(1, 2).float().reshape(b, n, hk, g, w, d)
        dq = (da * _shifted(k1, w)[:, :, :, None]).sum(4).reshape(b, n, hq, d)
        dk1 = _unshifted((da * q.view(b, n, hk, g, 1, d)).sum(3))
        dv1 = _unshifted((weighted * of).sum(3))
        grads = (dq, dk1, dk2.transpose(1, 2), dv1, dv2.transpose(1, 2))
        return *(x.to(q.dtype) for x in grads), None, None


def virtual_simplicial(q, k1, k2, v1, v2, *, short_window, scale):
    """Full-causal joint 2-simplicial softmax with ``reference_simplicial`` semantics.

    The long window spans the whole sequence. Requires CUDA BF16 head_dim=128 inputs
    and 1 <= short_window <= sequence length.
    """
    if not 1 <= short_window <= q.shape[1]:
        raise ValueError("require 1 <= short_window <= sequence length")
    if not q.is_cuda or any(x.dtype != torch.bfloat16 for x in (q, k1, k2, v1, v2)) or q.shape[-1] != 128:
        raise ValueError("virtual simplicial attention requires CUDA BF16 head_dim=128 inputs")
    return _VirtualSimplicial.apply(q, k1, k2, v1, v2, short_window, scale)
