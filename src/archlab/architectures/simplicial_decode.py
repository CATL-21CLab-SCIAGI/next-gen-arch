"""One-query inference form of the trained deterministic simplicial kernel.

The two key windows retain their chronological order. The arithmetic and
online-softmax loop are the forward half of ``simplicial_kernels._forward``;
only the query and key sequence strides differ. No backward is exposed.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode(Q, K1, K2, V1, V2, OUT,
            HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
            G: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr,
            H: tl.constexpr, T: tl.constexpr):
    bg = tl.program_id(0)
    b, g = bg // HK, bg % HK
    h, d, t = tl.arange(0, H), tl.arange(0, D), tl.arange(0, T)
    qo = (b * HQ + g * G + h[:, None]) * D + d[None, :]
    q = tl.load(Q + qo, h[:, None] < G, other=0)
    acc = tl.full((H, D), 0, tl.float32)
    m = tl.full((H,), -float("inf"), tl.float32)
    z = tl.full((H,), 0, tl.float32)
    for j in range(N1):
        ko = ((b * N1 + j) * HK + g) * D + d
        k1 = tl.load(K1 + ko)
        v1 = tl.load(V1 + ko).to(tl.float32)
        a = (q.to(tl.float32) * k1[None, :].to(tl.float32)).to(q.dtype)
        for start in range(0, N2, T):
            k = start + t
            off = ((b * N2 + k[:, None]) * HK + g) * D + d[None, :]
            k2 = tl.load(K2 + off, k[:, None] < N2, other=0)
            v2 = tl.load(V2 + off, k[:, None] < N2, other=0)
            s = tl.dot(a, tl.trans(k2), input_precision="tf32x3") * (D ** -0.5)
            s = tl.where(k[None, :] < N2, s, -float("inf"))
            nm = tl.maximum(m, tl.max(s, 1))
            p = tl.exp(s - nm[:, None])
            alpha = tl.exp(m - nm)
            z = z * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v2.dtype), v2, input_precision="tf32x3") * v1[None, :]
            m = nm
    tl.store(OUT + qo, acc / z[:, None], h[:, None] < G)


def simplicial_decode_attention(q, k1, k2, v1, v2):
    """Attend one query to bounded recent keys with trained FP32 arithmetic."""
    if (q.ndim != 4 or q.shape[1] != 1 or any(x.ndim != 4 for x in (k1, k2, v1, v2))
            or min(q.shape) < 1 or min(k1.shape) < 1 or min(k2.shape) < 1
            or k1.shape != v1.shape or k2.shape != v2.shape
            or k1.shape[0] != q.shape[0] or k2.shape[0] != q.shape[0]
            or k1.shape[2:] != k2.shape[2:] or q.shape[-1] != k1.shape[-1]
            or q.shape[2] % k1.shape[2] or min(k1.shape[1], k2.shape[1]) < 1
            or k1.shape[1] > k2.shape[1]):
        raise ValueError("incompatible one-query simplicial windows")
    if (q.device.type != "cuda" or q.dtype != torch.float32 or q.shape[-1] not in (16, 32, 64, 128, 256)
            or any(x.device != q.device or x.dtype != q.dtype for x in (k1, k2, v1, v2))):
        raise ValueError("simplicial decode requires CUDA FP32 projections")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k1, k2, v1, v2)):
        raise ValueError("simplicial decode has no backward; use the training kernel")
    q, k1, k2, v1, v2 = (x.contiguous() for x in (q, k1, k2, v1, v2))
    batch, _, heads, dim = q.shape
    groups = k1.shape[2]
    group_size = heads // groups
    if group_size > 128:
        raise ValueError("at most 128 query heads per KV group")
    out = torch.empty_like(q)
    _decode[(batch * groups,)](q, k1, k2, v1, v2, out, HQ=heads, HK=groups, D=dim,
                               G=group_size, N1=k1.shape[1], N2=k2.shape[1],
                               H=max(16, triton.next_power_of_2(group_size)), T=32)
    return out
