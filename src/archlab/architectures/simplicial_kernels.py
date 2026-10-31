"""Project-owned stock-Triton kernels for a single-GPU simplicial experiment.

One program owns a query position and KV group, with masked head padding.
Online softmax spans *both* windows. Backward recomputes probabilities instead
of saving pair scores; shared K/V gradients accumulate in FP32 with atomics.
The operation is scale * sum_d(q*k1*k2) over the score dimension DS (q, k1, k2); values (v1, v2,
output) have their own dimension DV. The determinant form (simplicial_attention.determinant_*) maps onto
this with a doubled score dimension, so the kernel stays elementwise.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(Q, K1, K2, V1, V2, OUT, LSE,
             N: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr, DS: tl.constexpr, DV: tl.constexpr,
             SCALE: tl.constexpr, G: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr,
             H: tl.constexpr, T: tl.constexpr):
    i = tl.program_id(0)
    bg = tl.program_id(1)
    b, g = bg // HK, bg % HK
    h, d, e, t = tl.arange(0, H), tl.arange(0, DS), tl.arange(0, DV), tl.arange(0, T)
    qo = ((b * N + i) * HQ + g * G + h[:, None]) * DS + d[None, :]
    oo = ((b * N + i) * HQ + g * G + h[:, None]) * DV + e[None, :]
    q = tl.load(Q + qo, h[:, None] < G, other=0)
    acc = tl.full((H, DV), 0, tl.float32)
    m = tl.full((H,), -float("inf"), tl.float32)
    z = tl.full((H,), 0, tl.float32)
    for j in range(tl.maximum(0, i - W1 + 1), i + 1):
        ko = ((b * N + j) * HK + g) * DS + d
        k1 = tl.load(K1 + ko)
        v1 = tl.load(V1 + ((b * N + j) * HK + g) * DV + e).to(tl.float32)
        a = (q.to(tl.float32) * k1[None, :].to(tl.float32)).to(q.dtype)
        for start in range(tl.maximum(0, i - W2 + 1), i + 1, T):
            k = start + t
            off = ((b * N + k[:, None]) * HK + g) * DS + d[None, :]
            offv = ((b * N + k[:, None]) * HK + g) * DV + e[None, :]
            k2 = tl.load(K2 + off, k[:, None] <= i, other=0)
            v2 = tl.load(V2 + offv, k[:, None] <= i, other=0)
            s = tl.dot(a, tl.trans(k2), input_precision="tf32x3") * SCALE
            s = tl.where(k[None, :] <= i, s, -float("inf"))
            nm = tl.maximum(m, tl.max(s, 1))
            p = tl.exp(s - nm[:, None])
            alpha = tl.exp(m - nm)
            z = z * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v2.dtype), v2, input_precision="tf32x3") * v1[None, :]
            m = nm
    tl.store(OUT + oo, acc / z[:, None], h[:, None] < G)
    lo = (b * N + i) * HQ + g * G + h
    tl.store(LSE + lo, m + tl.log(z), h < G)


@triton.jit
def _backward(Q, K1, K2, V1, V2, OUT, LSE, DO, DQ, DK1, DK2, DV1, DV2,
              N: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr, DS: tl.constexpr, DV: tl.constexpr,
              SCALE: tl.constexpr, G: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr,
              H: tl.constexpr, T: tl.constexpr):
    i, bg = tl.program_id(0), tl.program_id(1)
    b, g = bg // HK, bg % HK
    h, d, e, t = tl.arange(0, H), tl.arange(0, DS), tl.arange(0, DV), tl.arange(0, T)
    qo = ((b * N + i) * HQ + g * G + h[:, None]) * DS + d[None, :]
    oo = ((b * N + i) * HQ + g * G + h[:, None]) * DV + e[None, :]
    q = tl.load(Q + qo, h[:, None] < G, other=0)
    do = tl.load(DO + oo, h[:, None] < G, other=0)
    out = tl.load(OUT + oo, h[:, None] < G, other=0).to(tl.float32)
    delta = tl.sum(do.to(tl.float32) * out, 1)
    lo = (b * N + i) * HQ + g * G + h
    lse = tl.load(LSE + lo, h < G, other=0)
    dq = tl.full((H, DS), 0, tl.float32)
    for j in range(tl.maximum(0, i - W1 + 1), i + 1):
        ko = ((b * N + j) * HK + g) * DS + d
        kov = ((b * N + j) * HK + g) * DV + e
        k1 = tl.load(K1 + ko)
        v1 = tl.load(V1 + kov)
        a = (q.to(tl.float32) * k1[None, :].to(tl.float32)).to(q.dtype)
        dov1 = (do.to(tl.float32) * v1[None, :].to(tl.float32)).to(do.dtype)
        dk1 = tl.full((H, DS), 0, tl.float32)
        dv1 = tl.full((H, DV), 0, tl.float32)
        for start in range(tl.maximum(0, i - W2 + 1), i + 1, T):
            k = start + t
            off = ((b * N + k[:, None]) * HK + g) * DS + d[None, :]
            offv = ((b * N + k[:, None]) * HK + g) * DV + e[None, :]
            k2 = tl.load(K2 + off, k[:, None] <= i, other=0)
            v2 = tl.load(V2 + offv, k[:, None] <= i, other=0)
            s = tl.dot(a, tl.trans(k2), input_precision="tf32x3") * SCALE
            p = tl.exp(s - lse[:, None])
            p = tl.where((k[None, :] <= i) & (h[:, None] < G), p, 0)
            dp = tl.dot(dov1, tl.trans(v2), input_precision="tf32x3")
            ds = (p * (dp - delta[:, None]) * SCALE).to(q.dtype)
            # With exactly one valid pair, softmax is constant: score
            # derivatives are identically zero, even with rounded BF16 O.
            ds = tl.where((i == 0) | ((W1 == 1) & (W2 == 1)), 0, ds)
            da = tl.dot(ds, k2, input_precision="tf32x3")
            dq += da * k1[None, :].to(tl.float32)
            dk1 += da * q.to(tl.float32)
            dk2 = tl.dot(tl.trans(ds), a, input_precision="tf32x3")
            pv2 = tl.dot(p.to(v2.dtype), v2, input_precision="tf32x3")
            dv1 += pv2 * do.to(tl.float32)
            dv2 = tl.dot(tl.trans(p.to(do.dtype)), dov1, input_precision="tf32x3")
            tl.atomic_add(DK2 + off, dk2, k[:, None] <= i, sem="relaxed")
            tl.atomic_add(DV2 + offv, dv2, k[:, None] <= i, sem="relaxed")
        tl.atomic_add(DK1 + ko, tl.sum(dk1, 0), sem="relaxed")
        tl.atomic_add(DV1 + kov, tl.sum(dv1, 0), sem="relaxed")
    tl.store(DQ + qo, dq, h[:, None] < G)


class SimplicialFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k1, k2, v1, v2, w1, w2, scale):
        q, k1, k2, v1, v2 = [x.contiguous() for x in (q, k1, k2, v1, v2)]
        b, n, hq, d = q.shape
        hk, dv = k1.shape[2], v1.shape[-1]
        group = hq // hk
        if group > 128:
            raise ValueError("at most 128 query heads per KV group")
        output = q.new_empty(b, n, hq, dv)
        lse = torch.empty((b, n, hq), device=q.device, dtype=torch.float32)
        wide = d > 256
        config = dict(N=n, HQ=hq, HK=hk, DS=d, DV=dv, SCALE=scale,
                      G=group, W1=w1, W2=w2, H=max(16, triton.next_power_of_2(group)),
                      # Expanded FP32 scores need smaller tiles and less staging
                      # to fit both forward and backward in B300 shared memory.
                      T=16 if wide else 32,
                      num_stages=1 if wide else 3,
                      num_warps=8 if wide else 4)
        _forward[(n, b * hk)](q, k1, k2, v1, v2, output, lse, **config)
        ctx.save_for_backward(q, k1, k2, v1, v2, output, lse)
        ctx.config = config
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k1, k2, v1, v2, output, lse = ctx.saved_tensors
        dq = torch.empty_like(q)
        # Shared K/V accumulate without BF16 atomic rounding after each query.
        grads = [torch.zeros_like(x, dtype=torch.float32) for x in (k1, k2, v1, v2)]
        _backward[(q.shape[1], q.shape[0] * k1.shape[2])](
            q, k1, k2, v1, v2, output, lse, grad_output.contiguous(), dq, *grads,
            **ctx.config,
        )
        return dq, *(x.to(q.dtype) for x in grads), None, None, None
