"""Cache-aware 2-simplicial kernels packing short-axis pairs into GEMM rows.

Native head geometry is unchanged. Forward reuses long K/V tiles across short
keys; backward also reuses them across two adjacent queries, combining their
long-axis gradient writes. Joint softmax still normalizes both key axes.
Training retains native BF16 inputs and the original TF32x3 forward rounding.
Backward compensates BF16 tensor products with their residuals. Softmax,
outputs and shared gradients accumulate in FP32. Two-stage forward and
three-stage backward pipelines overlap tile loading with computation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compensated_dot(a, b, accurate: tl.constexpr = False):
    """BF16 tensor products with FP32 accumulation and residual compensation."""
    if accurate:
        # Preserve the forward rounding used before this checkpoint. Across 48
        # BF16 residual blocks, otherwise tiny attention differences amplify.
        return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="tf32x3")
    ah, bh = a.to(tl.bfloat16), b.to(tl.bfloat16)
    if a.dtype == tl.bfloat16 and b.dtype == tl.bfloat16:
        return tl.dot(ah, bh)
    if a.dtype == tl.bfloat16:
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        return tl.dot(ah, bh, tl.dot(ah, bl))
    al = (a - ah.to(tl.float32)).to(tl.bfloat16)
    residual = tl.dot(al, bh)
    if b.dtype != tl.bfloat16:
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        residual = tl.dot(ah, bl, residual)
    return tl.dot(ah, bh, residual)


@triton.jit
def _forward(
    Q,
    K1,
    K2,
    V1,
    V2,
    OUT,
    LSE,
    N: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    T: tl.constexpr,
):
    i, bg = tl.program_id(0), tl.program_id(1)
    b, g = bg // HQ, (bg % HQ) // (HQ // HK)
    row = tl.arange(0, S * H)
    d = tl.arange(0, D)
    t = tl.arange(0, T)
    heads = tl.arange(0, H)
    h = row % H
    j = i - row // H
    valid = (h < G) & (j >= 0) & (row // H < W1)
    qo = ((b * N + i) * HQ + (bg % HQ) + h[:, None]) * D + d[None, :]
    ko = ((b * N + j[:, None]) * HK + g) * D + d[None, :]
    q = tl.load(Q + qo, h[:, None] < G, other=0).to(tl.float32)
    k1 = tl.load(K1 + ko, valid[:, None], other=0)
    v1 = tl.load(V1 + ko, valid[:, None], other=0)
    a = q * k1
    acc = tl.full((S * H, D), 0, tl.float32)
    m = tl.full((H,), -float("inf"), tl.float32)
    z = tl.full((H,), 0, tl.float32)
    for start in range(tl.maximum(0, i - W2 + 1), i + 1, T):
        k = start + t
        off = ((b * N + k[:, None]) * HK + g) * D + d[None, :]
        k2 = tl.load(K2 + off, k[:, None] <= i, other=0)
        v2 = tl.load(V2 + off, k[:, None] <= i, other=0)
        scores = _compensated_dot(a, tl.trans(k2), accurate=True) * (D**-0.5)
        scores = tl.where(valid[:, None] & (k[None, :] <= i), scores, -float("inf"))
        maximum = tl.max(tl.max(tl.reshape(scores, (S, H, T)), 2), 0)
        nm = tl.maximum(m, maximum)
        # Invalid padded heads need finite normalization to avoid NaNs in dot inputs.
        nm = tl.where(heads < G, nm, 0.0)
        p = tl.exp(scores - tl.reshape(tl.broadcast_to(nm[None, :], (S, H)), (S * H,))[:, None])
        alpha = tl.exp(m - nm)
        acc = acc * tl.reshape(tl.broadcast_to(alpha[None, :], (S, H)), (S * H,))[
            :, None
        ] + _compensated_dot(p, v2, accurate=True)
        z = z * alpha + tl.sum(tl.sum(tl.reshape(p, (S, H, T)), 2), 0)
        m = nm
    out = tl.sum(tl.reshape(acc * v1, (S, H, D)), 0) / z[:, None]
    oo = ((b * N + i) * HQ + (bg % HQ) + heads[:, None]) * D + d[None, :]
    tl.store(OUT + oo, out, heads[:, None] < G)
    tl.store(LSE + (b * N + i) * HQ + (bg % HQ) + heads, m + tl.log(z), heads < G)


@triton.jit
def _backward(
    Q,
    K1,
    K2,
    V1,
    V2,
    OUT,
    LSE,
    DO,
    DQ,
    DK1,
    DK2,
    DV1,
    DV2,
    N: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    T: tl.constexpr,
    BQ: tl.constexpr,
):
    first, bg = tl.program_id(0) * BQ, tl.program_id(1)
    b, g = bg // HQ, (bg % HQ) // (HQ // HK)
    row = tl.arange(0, BQ * S)
    d = tl.arange(0, D)
    t = tl.arange(0, T)
    queries = first + tl.arange(0, BQ)
    i = first + row // S
    j = i - row % S
    valid = (i < N) & (j >= 0) & (row % S < W1)
    qo = ((b * N + i[:, None]) * HQ + bg % HQ) * D + d[None, :]
    ko = ((b * N + j[:, None]) * HK + g) * D + d[None, :]
    q = tl.load(Q + qo, i[:, None] < N, other=0).to(tl.float32)
    k1 = tl.load(K1 + ko, valid[:, None], other=0)
    v1 = tl.load(V1 + ko, valid[:, None], other=0)
    do = tl.load(DO + qo, i[:, None] < N, other=0)
    out = tl.load(OUT + qo, i[:, None] < N, other=0)
    lse = tl.load(LSE + (b * N + i) * HQ + bg % HQ, i < N, other=0)
    delta = tl.sum(do * out, 1)
    a = q * k1
    dov1 = do * v1
    da = tl.full((BQ * S, D), 0, tl.float32)
    dv1 = tl.full((BQ * S, D), 0, tl.float32)
    last = tl.minimum(first + BQ, N) - 1
    # Scan the union of these queries' windows, then enforce each query's
    # own causal/local bounds before sharing the long-axis gradient GEMMs.
    for start in range(tl.maximum(0, first - W2 + 1), last + 1, T):
        k = start + t
        off = ((b * N + k[:, None]) * HK + g) * D + d[None, :]
        k2 = tl.load(K2 + off, k[:, None] <= last, other=0)
        v2 = tl.load(V2 + off, k[:, None] <= last, other=0)
        scores = _compensated_dot(a, tl.trans(k2)) * (D**-0.5)
        pair_valid = valid[:, None] & (k[None, :] <= i[:, None]) & (k[None, :] > i[:, None] - W2)
        p = tl.where(pair_valid, tl.exp(scores - lse[:, None]), 0.0)
        dp = _compensated_dot(dov1, tl.trans(v2))
        ds = p * (dp - delta[:, None]) * (D**-0.5)
        ds = tl.where((i[:, None] == 0) | ((W1 == 1) & (W2 == 1)), 0.0, ds)
        da += _compensated_dot(ds, k2)
        dv1 += _compensated_dot(p, v2) * do
        dk2 = _compensated_dot(tl.trans(ds), a)
        dv2 = _compensated_dot(tl.trans(p), dov1)
        tl.atomic_add(DK2 + off, dk2, k[:, None] <= last, sem="relaxed")
        tl.atomic_add(DV2 + off, dv2, k[:, None] <= last, sem="relaxed")
    dq = tl.sum(tl.reshape(da * k1, (BQ, S, D)), 1)
    oo = ((b * N + queries[:, None]) * HQ + bg % HQ) * D + d[None, :]
    tl.store(DQ + oo, dq, queries[:, None] < N)
    tl.atomic_add(DK1 + ko, da * q, valid[:, None], sem="relaxed")
    tl.atomic_add(DV1 + ko, dv1, valid[:, None], sem="relaxed")


class PackedSimplicial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k1, k2, v1, v2, w1, w2):
        xs = [x.contiguous() for x in (q, k1, k2, v1, v2)]
        q, k1, k2, v1, v2 = xs
        b, n, hq, d = q.shape
        hk = k1.shape[2]
        if q.dtype not in (torch.float32, torch.bfloat16) or w1 > 16 or d != 128:
            raise ValueError("packed kernel requires FP32/BF16, head_dim=128, short window <=16")
        cfg = dict(N=n, HQ=hq, HK=hk, D=d, G=1, W1=w1, W2=w2, H=1, S=16, T=64)
        out = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty((b, n, hq), device=q.device, dtype=torch.float32)
        # Explicit FP32 forward buffers preserve the existing compiler path and
        # its rounding. Retain only native BF16 K/V tensors for backward.
        forward_inputs = [x.float() for x in xs]
        # Two stages are fastest while preserving the original forward
        # query/window partition and reduction order.
        _forward[(n, b * hq)](*forward_inputs, out, lse, **cfg, num_warps=4, num_stages=2)
        ctx.save_for_backward(*xs, out, lse)
        ctx.cfg = cfg
        return out

    @staticmethod
    def backward(ctx, do):
        q, k1, k2, v1, v2, out, lse = ctx.saved_tensors
        dq = torch.empty_like(q, dtype=torch.float32)
        grads = [torch.zeros_like(x, dtype=torch.float32) for x in (k1, k2, v1, v2)]
        # Two queries reduce repeated loads and atomics without the severe
        # register spills of larger tiles on B300. Three stages improve the
        # complete backward despite additional compiler spills.
        _backward[(triton.cdiv(q.shape[1], 2), q.shape[0] * q.shape[2])](
            q,
            k1,
            k2,
            v1,
            v2,
            out,
            lse,
            do.contiguous(),
            dq,
            *grads,
            **dict(ctx.cfg, T=32, BQ=2),
            num_warps=4,
            num_stages=3,
        )
        converted = tuple(
            g.to(x.dtype) for g, x in zip((dq, *grads), (q, k1, k2, v1, v2), strict=True)
        )
        return *converted, None, None


def packed_simplicial_attention(q, k1, k2, v1, v2, w1, w2):
    from archlab.architectures.simplicial_attention import validate_inputs

    validate_inputs(q, k1, k2, v1, v2, w1, w2, allow_scaled_query=True)
    if not q.is_cuda:
        raise ValueError("packed kernel requires CUDA")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("simplicial backward uses nondeterministic FP32 atomics")
    return PackedSimplicial.apply(q, k1, k2, v1, v2, w1, w2)


@triton.jit(do_not_specialize=["N1", "N2", "PARTS"])
def _decode_part(
    Q,
    K1,
    K2,
    V1,
    V2,
    PARTIAL,
    LSE,
    N1,
    N2,
    PARTS,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
):
    bh, part = tl.program_id(0), tl.program_id(1)
    b, h = bh // HQ, bh % HQ
    g = h // (HQ // HK)
    j, d, t = tl.arange(0, 16), tl.arange(0, D), tl.arange(0, T)
    q = tl.load(Q + bh * D + d)
    off1 = ((b * N1 + j[:, None]) * HK + g) * D + d[None, :]
    k1 = tl.load(K1 + off1, j[:, None] < N1, other=0)
    v1 = tl.load(V1 + off1, j[:, None] < N1, other=0)
    k = part * T + t
    off2 = ((b * N2 + k[:, None]) * HK + g) * D + d[None, :]
    k2 = tl.load(K2 + off2, k[:, None] < N2, other=0)
    v2 = tl.load(V2 + off2, k[:, None] < N2, other=0)
    scores = tl.dot(q[None, :] * k1, tl.trans(k2), input_precision="tf32x3") * (D**-0.5)
    scores = tl.where((j[:, None] < N1) & (k[None, :] < N2), scores, -float("inf"))
    maximum = tl.max(tl.max(scores, 1), 0)
    p = tl.exp(scores - maximum)
    z = tl.sum(tl.sum(p, 1), 0)
    value = tl.sum(tl.dot(p, v2, input_precision="tf32x3") * v1, 0) / z
    tl.store(PARTIAL + (bh * PARTS + part) * D + d, value)
    tl.store(LSE + bh * PARTS + part, maximum + tl.log(z))


@triton.jit(do_not_specialize=["PARTS"])
def _decode_merge(PARTIAL, LSE, OUT, PARTS, D: tl.constexpr, P: tl.constexpr):
    bh = tl.program_id(0)
    p, d = tl.arange(0, P), tl.arange(0, D)
    lse = tl.load(LSE + bh * PARTS + p, p < PARTS, other=-float("inf"))
    probability = tl.exp(lse - tl.max(lse, 0))
    probability /= tl.sum(probability, 0)
    value = tl.load(
        PARTIAL + (bh * PARTS + p[:, None]) * D + d[None, :], p[:, None] < PARTS, other=0
    )
    tl.store(OUT + bh * D + d, tl.sum(value * probability[:, None], 0))


def packed_simplicial_decode(q, k1, k2, v1, v2):
    """Split long keys across SMs; growing cache lengths are runtime scalars."""
    if (
        q.ndim != 4
        or q.shape[1] != 1
        or q.shape[-1] != 128
        or q.dtype != torch.float32
        or not q.is_cuda
    ):
        raise ValueError("native packed decode requires CUDA FP32 [B,1,H,128]")
    if (
        k1.shape != v1.shape
        or k2.shape != v2.shape
        or k1.shape[0] != q.shape[0]
        or k2.shape[0] != q.shape[0]
        or k1.shape[2:] != k2.shape[2:]
        or k1.shape[-1] != 128
        or q.shape[2] % k1.shape[2]
        or not 1 <= k1.shape[1] <= min(16, k2.shape[1])
    ):
        raise ValueError("incompatible native decode windows")
    if any(x.dtype != q.dtype or x.device != q.device for x in (k1, k2, v1, v2)):
        raise ValueError("native decode inputs must share dtype/device")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k1, k2, v1, v2)):
        raise ValueError("decode is inference-only")
    xs = [x.contiguous() for x in (q, k1, k2, v1, v2)]
    b, _, h, d = q.shape
    parts = triton.cdiv(k2.shape[1], 128)
    partial = torch.empty((b * h, parts, d), device=q.device, dtype=q.dtype)
    lse = torch.empty((b * h, parts), device=q.device, dtype=q.dtype)
    out = torch.empty_like(q)
    _decode_part[(b * h, parts)](
        *xs,
        partial,
        lse,
        k1.shape[1],
        k2.shape[1],
        parts,
        HQ=h,
        HK=k1.shape[2],
        D=d,
        T=128,
        num_warps=4,
        num_stages=1,
    )
    _decode_merge[(b * h,)](
        partial, lse, out, parts, D=d, P=triton.next_power_of_2(parts), num_warps=4
    )
    return out
