"""Shared BF16 GQA/joint-softmax attention for native Limite adapter geometry.

Derived from TileLang v0.1.8 FlashAttention/GQA scheduling examples at
41b25527cd672434f88eeea7e056d8d7c0d4faa4 (Copyright Tile-AI, MIT;
see docs/NOTICE.md). The short-axis extension normalizes all key pairs jointly.
BF16 tensor products use FP32 softmax, outputs and shared-gradient atomics.
This fresh-run precision contract deliberately differs from the historical
Triton FP32/TF32x3 path; checkpoint backends may not be silently changed.
"""

from functools import cache

import tilelang
import tilelang.language as T
import torch


@cache
def _unused_lengths(device):
    return torch.empty(3, device=device, dtype=torch.int32)


@tilelang.jit(out_idx=[5, 6], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _forward_kernel(B, N, HQ, HK, D, S, W1, W2, scale, BQ=4, BN=64, stages=1, decode=False, bounded_decode=False):
    N = T.dynamic("sequence_length") if N is None else N
    W2 = N if W2 == 0 else W2
    M = BQ * S
    N1 = T.dynamic("short_length") if decode else N
    N2 = T.dynamic("long_length") if decode else N

    @T.prim_func
    def main(
        Q: T.Tensor([B, N, HQ, D], T.bfloat16),
        K1: T.Tensor([B, N1, HK, D], T.bfloat16),
        K2: T.Tensor([B, N2, HK, D], T.bfloat16),
        V1: T.Tensor([B, N1, HK, D], T.bfloat16),
        V2: T.Tensor([B, N2, HK, D], T.bfloat16),
        OUT: T.Tensor([B, N, HQ, D], T.float32),
        LSE: T.Tensor([B, N, HQ], T.float32),
        LENGTHS: T.Tensor([3], T.int32),
    ):
        with T.Kernel(T.ceildiv(N, BQ), HQ, B, threads=128) as (bx, h, b):
            a = T.alloc_shared([M, D], T.bfloat16)
            vlo = T.alloc_fragment([M, D], T.float32)
            kv = T.alloc_shared([BN, D], T.bfloat16)
            vv = T.alloc_shared([BN, D], T.bfloat16)
            score = T.alloc_fragment([M, BN], T.float32)
            prob = T.alloc_fragment([M, BN], T.bfloat16)
            acc = T.alloc_fragment([M, D], T.float32)
            rowmax = T.alloc_fragment([M], T.float32)
            rowmax2 = T.reshape(rowmax, [BQ, S])
            rowz = T.alloc_fragment([M], T.float32)
            rowz2 = T.reshape(rowz, [BQ, S])
            mx = T.alloc_fragment([BQ], T.float32)
            old = T.alloc_fragment([BQ], T.float32)
            current_max = T.alloc_fragment([BQ], T.float32)
            z = T.alloc_fragment([BQ], T.float32)
            zz = T.alloc_fragment([BQ], T.float32)
            output = T.alloc_fragment([BQ, D], T.float32)
            result = T.reshape(vlo, [BQ, S, D])
            for r, d in T.Parallel(M, D):
                i = bx * BQ + r // S
                j = N1 - 1 - r % S if decode else i - r % S
                if S == 1:
                    a[r, d] = T.if_then_else(i < N, Q[b, i, h, d], 0)
                    vlo[r, d] = 1
                else:
                    a[r, d] = T.if_then_else(
                        i < N and j >= 0 and r % S < W1
                        and (r % S < LENGTHS[1] if bounded_decode else True),
                        Q[b, i, h, d].astype(T.float32)
                        * K1[b, j, h // (HQ // HK), d].astype(T.float32),
                        0,
                    )
                    vlo[r, d] = T.if_then_else(
                        i < N and j >= 0 and r % S < W1
                        and (r % S < LENGTHS[1] if bounded_decode else True), V1[b, j, h // (HQ // HK), d], 0
                    )
            T.clear(acc)
            T.clear(z)
            T.fill(mx, -T.infinity(T.float32))
            start = (LENGTHS[2] if bounded_decode else 0) if decode else T.max(0, bx * BQ - W2 + 1)
            end = (start + LENGTHS[0] if bounded_decode else N2) if decode else T.min(N, (bx + 1) * BQ)
            for tile in T.Pipelined(T.ceildiv(end - start, BN), num_stages=stages):
                T.copy(K2[b, start + tile * BN : start + (tile + 1) * BN, h // (HQ // HK), :], kv)
                T.copy(V2[b, start + tile * BN : start + (tile + 1) * BN, h // (HQ // HK), :], vv)
                T.clear(score)
                T.gemm(a, kv, score, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for r, k in T.Parallel(M, BN):
                    i = bx * BQ + r // S
                    key = start + tile * BN + k
                    score[r, k] = T.if_then_else(
                        i < N
                        and (N1 - 1 - r % S >= 0 if decode else i - r % S >= 0)
                        and r % S < W1
                        and (r % S < LENGTHS[1] if bounded_decode else True)
                        and (key < end if decode else key <= i and key > i - W2),
                        score[r, k] * scale * 1.4426950408889634,
                        -T.infinity(T.float32),
                    )
                T.reduce_max(score, rowmax, dim=1)
                T.reduce_max(rowmax2, current_max, dim=1)
                T.copy(mx, old)
                for i in T.Parallel(BQ):
                    mx[i] = T.if_then_else(bx * BQ + i < N, T.max(mx[i], current_max[i]), 0)
                for r, k in T.Parallel(M, BN):
                    score[r, k] = T.exp2(score[r, k] - mx[r // S])
                T.reduce_sum(score, rowz, dim=1)
                T.reduce_sum(rowz2, zz, dim=1)
                for i in T.Parallel(BQ):
                    z[i] = z[i] * T.exp2(old[i] - mx[i]) + zz[i]
                for r, d in T.Parallel(M, D):
                    acc[r, d] *= T.exp2(old[r // S] - mx[r // S])
                T.copy(score, prob)
                T.gemm(prob, vv, acc, policy=T.GemmWarpPolicy.FullRow)
            for r, d in T.Parallel(M, D):
                vlo[r, d] *= acc[r, d]
            T.reduce_sum(result, output, dim=1)
            for i, d in T.Parallel(BQ, D):
                if bx * BQ + i < N:
                    OUT[b, bx * BQ + i, h, d] = output[i, d] / z[i]
            for i in T.Parallel(BQ):
                if bx * BQ + i < N:
                    LSE[b, bx * BQ + i, h] = T.log2(z[i]) + mx[i]

    return main


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _backward_kernel(B, N, HQ, HK, D, S, W1, W2, scale, BQ=4, BN=32, stages=1):
    N = T.dynamic("sequence_length") if N is None else N
    W2 = N if W2 == 0 else W2
    M = BQ * S

    @T.prim_func
    def main(
        Q: T.Tensor([B, N, HQ, D], T.bfloat16),
        K1: T.Tensor([B, N, HK, D], T.bfloat16),
        K2: T.Tensor([B, N, HK, D], T.bfloat16),
        V1: T.Tensor([B, N, HK, D], T.bfloat16),
        V2: T.Tensor([B, N, HK, D], T.bfloat16),
        OUT: T.Tensor([B, N, HQ, D], T.float32),
        LSE: T.Tensor([B, N, HQ], T.float32),
        DO: T.Tensor([B, N, HQ, D], T.float32),
        DQ: T.Tensor([B, N, HQ, D], T.float32),
        DK1: T.Tensor([B, N, HK, D], T.float32),
        DK2: T.Tensor([B, N, HK, D], T.float32),
        DV1: T.Tensor([B, N, HK, D], T.float32),
        DV2: T.Tensor([B, N, HK, D], T.float32),
    ):
        with T.Kernel(T.ceildiv(N, BQ), HQ, B, threads=128) as (bx, h, b):
            a = T.alloc_shared([M, D], T.bfloat16)
            dov = T.alloc_shared([M, D], T.bfloat16)
            # Q, K1 and DO are inexpensive to reload after the key loop. Keeping
            # three expanded FP32 fragments live through all six GEMMs raises
            # register pressure in the native short-axis specialization.
            tmp = T.alloc_fragment([M, D], T.float32)
            delta = T.alloc_fragment([M], T.float32)
            lse = T.alloc_fragment([M], T.float32)
            kv = T.alloc_shared([BN, D], T.bfloat16)
            vv = T.alloc_shared([BN, D], T.bfloat16)
            p = T.alloc_fragment([M, BN], T.float32)
            dp = T.alloc_fragment([M, BN], T.float32)
            pc = T.alloc_shared([M, BN], T.bfloat16)
            ds = T.alloc_shared([M, BN], T.bfloat16)
            da = T.alloc_fragment([M, D], T.float32)
            if S != 1:
                dv1 = T.alloc_fragment([M, D], T.float32)
            dk2 = T.alloc_fragment([BN, D], T.float32)
            dv2 = T.alloc_fragment([BN, D], T.float32)
            part = T.reshape(tmp, [BQ, S, D])
            dq = T.alloc_fragment([BQ, D], T.float32)
            for r, d in T.Parallel(M, D):
                i = bx * BQ + r // S
                j = i - r % S
                tmp[r, d] = T.if_then_else(i < N, DO[b, i, h, d] * OUT[b, i, h, d], 0)
                if S == 1:
                    a[r, d] = T.if_then_else(i < N, Q[b, i, h, d], 0)
                    dov[r, d] = T.if_then_else(i < N, DO[b, i, h, d], 0)
                else:
                    a[r, d] = T.if_then_else(
                        i < N and j >= 0 and r % S < W1,
                        Q[b, i, h, d].astype(T.float32)
                        * K1[b, j, h // (HQ // HK), d].astype(T.float32),
                        0,
                    )
                    dov[r, d] = T.if_then_else(
                        i < N and j >= 0 and r % S < W1,
                        DO[b, i, h, d] * V1[b, j, h // (HQ // HK), d].astype(T.float32),
                        0,
                    )
            T.reduce_sum(tmp, delta, dim=1)
            for r in T.Parallel(M):
                lse[r] = T.if_then_else(bx * BQ + r // S < N, LSE[b, bx * BQ + r // S, h], 0)
            T.clear(da)
            if S != 1:
                T.clear(dv1)
            start = T.max(0, bx * BQ - W2 + 1)
            end = T.min(N, (bx + 1) * BQ)
            for tile in T.Pipelined(T.ceildiv(end - start, BN), num_stages=stages):
                T.copy(K2[b, start + tile * BN : start + (tile + 1) * BN, h // (HQ // HK), :], kv)
                T.copy(V2[b, start + tile * BN : start + (tile + 1) * BN, h // (HQ // HK), :], vv)
                T.clear(p)
                T.clear(dp)
                T.gemm(a, kv, p, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(dov, vv, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for r, k in T.Parallel(M, BN):
                    i = bx * BQ + r // S
                    key = start + tile * BN + k
                    p[r, k] = T.if_then_else(
                        i < N and i - r % S >= 0 and r % S < W1 and key <= i and key > i - W2,
                        T.exp2(p[r, k] * scale * 1.4426950408889634 - lse[r]),
                        0,
                    )
                    pc[r, k] = p[r, k]
                    ds[r, k] = T.if_then_else(
                        i == 0 or (W1 == 1 and W2 == 1), 0, p[r, k] * (dp[r, k] - delta[r]) * scale
                    )
                T.gemm(ds, kv, da, policy=T.GemmWarpPolicy.FullRow)
                if S != 1:
                    # DO is constant across long keys. Accumulate P @ V2 in
                    # FP32 and apply DO once when scattering the short gradient.
                    T.gemm(pc, vv, dv1, policy=T.GemmWarpPolicy.FullRow)
                T.clear(dk2)
                T.clear(dv2)
                T.gemm(ds, a, dk2, transpose_A=True)
                T.gemm(pc, dov, dv2, transpose_A=True)
                for k, d in T.Parallel(BN, D):
                    key = start + tile * BN + k
                    if key < end:
                        T.atomic_add(DK2[b, key, h // (HQ // HK), d], dk2[k, d])
                        T.atomic_add(DV2[b, key, h // (HQ // HK), d], dv2[k, d])
            for r, d in T.Parallel(M, D):
                i = bx * BQ + r // S
                j = i - r % S
                if S == 1:
                    tmp[r, d] = da[r, d]
                else:
                    tmp[r, d] = da[r, d] * T.if_then_else(
                        i < N and j >= 0 and r % S < W1,
                        K1[b, j, h // (HQ // HK), d].astype(T.float32),
                        0,
                    )
            T.reduce_sum(part, dq, dim=1)
            for i, d in T.Parallel(BQ, D):
                if bx * BQ + i < N:
                    DQ[b, bx * BQ + i, h, d] = dq[i, d]
            if S != 1:
                for r, d in T.Parallel(M, D):
                    i = bx * BQ + r // S
                    j = i - r % S
                    if i < N and j >= 0 and r % S < W1:
                        T.atomic_add(
                            DK1[b, j, h // (HQ // HK), d],
                            da[r, d] * Q[b, i, h, d].astype(T.float32),
                        )
                        T.atomic_add(DV1[b, j, h // (HQ // HK), d], dv1[r, d] * DO[b, i, h, d])

    return main


class _JointAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k1, k2, v1, v2, s, w1, w2, scale, dynamic_length):
        xs = [x.contiguous() for x in (q, k1, k2, v1, v2)]
        b, n, h, d = q.shape
        hk = k1.shape[2]
        bq = 4 if s == 16 else 64
        cfg = (b, None if dynamic_length else n, h, hk, d, s, w1, w2, scale, bq)
        out, lse = _forward_kernel(*cfg, BN=128 if s == 16 else 64)(*xs, _unused_lengths(q.device))
        ctx.save_for_backward(*xs, out, lse)
        ctx.cfg = cfg
        return out

    @staticmethod
    def backward(ctx, gradient_output):
        q, k1, k2, v1, v2, out, lse = ctx.saved_tensors
        dq = torch.empty_like(q, dtype=torch.float32)
        if ctx.cfg[5] == 1:
            # The ordinary-attention specialization never reads/writes the short
            # branch; aliases avoid redundant zeroing and allocation.
            dk = torch.zeros_like(k2, dtype=torch.float32)
            dv = torch.zeros_like(v2, dtype=torch.float32)
            gs = [dk, dk, dv, dv]
        else:
            gs = [torch.zeros_like(x, dtype=torch.float32) for x in (k1, k2, v1, v2)]
        _backward_kernel(*ctx.cfg)(
            q, k1, k2, v1, v2, out, lse, gradient_output.float().contiguous(), dq, *gs
        )
        if ctx.cfg[5] == 1:
            grads = [dq.to(q.dtype), None, gs[1].to(k2.dtype), None, gs[3].to(v2.dtype)]
        else:
            grads = [g.to(x.dtype) for g, x in zip([dq, *gs], [q, k1, k2, v1, v2], strict=True)]
        return *grads, None, None, None, None, None


def _validate(q, k, v, short):
    inputs = (q, k, v, *(short or ()))
    if any(x.ndim != 4 for x in inputs):
        raise ValueError("expected [batch, sequence, heads, head_dim]")
    if not q.is_cuda or any(x.device != q.device or x.dtype != torch.bfloat16 for x in inputs):
        raise ValueError("TileLang attention requires CUDA BF16 inputs")
    if q.shape[-1] != 128 or min(q.shape) < 1 or min(k.shape) < 1:
        raise ValueError("TileLang attention requires nonempty native head_dim=128")
    if (
        k.shape != v.shape
        or k.shape[0] != q.shape[0]
        or k.shape[-1] != q.shape[-1]
        or q.shape[2] % k.shape[2]
    ):
        raise ValueError("incompatible grouped-query geometry")
    if short and (
        short[0].shape != short[1].shape
        or short[0].shape[0] != q.shape[0]
        or short[0].shape[2:] != k.shape[2:]
    ):
        raise ValueError("incompatible short-axis geometry")


def normal_attention_forward(q, k, v, *, scaling, long_window, dynamic_length=False):
    """Return the shared ordinary-attention output and log2 normalizer.

    The shared entry point validates geometry before invoking this callback.
    Keep the existing query/key tiling and BF16 rounding for alternate backward
    schedules to compare the same forward computation.
    """
    b, n, h, d = q.shape
    hk = k.shape[2]
    length = None if dynamic_length else n
    window = 0 if dynamic_length and long_window >= n else long_window
    return _forward_kernel(b, length, h, hk, d, 1, 1, window, scaling, 64, BN=64)(q, k, k, v, v, _unused_lengths(q.device))


def tilelang_attention(
    q, k, v, *, scaling, long_window, short=None, short_window=1, normal_kernel="shared",
    dynamic_length=False,
):
    """Ordinary GQA or joint 2-simplicial softmax; no implicit fallback."""
    import math

    _validate(q, k, v, short)
    if not math.isfinite(scaling) or scaling <= 0 or long_window < 1 or not 1 <= short_window <= 16:
        raise ValueError("invalid attention scale/window")
    if k.shape[1] != q.shape[1] or (short and short[0].shape[1] != q.shape[1]):
        raise ValueError("training attention requires matching sequence lengths")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("TileLang attention backward uses FP32 atomics")
    if normal_kernel not in ("shared", "gqa"):
        raise ValueError("normal_kernel must be shared or gqa")
    if short is None and normal_kernel == "gqa":
        from functools import partial

        from archlab.architectures.tilelang_gqa import gqa_attention

        return gqa_attention(
            q, k, v, scaling=scaling, long_window=long_window,
            forward=partial(normal_attention_forward, dynamic_length=dynamic_length),
            dynamic_length=dynamic_length,
        )
    k1, v1 = short if short else (k, v)
    window = 0 if dynamic_length and long_window >= q.shape[1] else long_window
    return _JointAttention.apply(
        q, k1, k, v1, v, 16 if short else 1, min(short_window, long_window), window, scaling,
        dynamic_length,
    )


def tilelang_decode(q, k, v, *, scaling, short=None, lengths=None):
    """Cached one-token attention using the same TileLang softmax arithmetic."""
    _validate(q, k, v, short)
    if q.shape[1] != 1 or torch.is_grad_enabled():
        raise ValueError("cached TileLang decode requires one token without gradients")
    k1, v1 = short if short else (k, v)
    b, _, h, d = q.shape
    s = 16 if short else 1
    w1 = k1.shape[1] if short else 1
    if not 1 <= w1 <= s:
        raise ValueError("cached short axis exceeds 16 keys")
    cfg = (b, 1, h, k.shape[2], d, s, s, 1, scaling, 4 if short else 64)
    out, _ = _forward_kernel(*cfg, decode=True, bounded_decode=lengths is not None)(
        *(x.contiguous() for x in (q, k1, k, v1, v)),
        _unused_lengths(q.device) if lengths is None else lengths,
    )
    return out
