"""Key-owned GQA backward for the exact shared Limite forward.

Adapted from MIT-licensed Tile-AI example_gqa_bwd.py at TileLang revision
a35f8ddf45eba16c21211ec8822d56ce5363036f (Copyright Tile-AI; see docs/NOTICE.md).
The supplied forward callback owns the FP32 output and log2 LSE. Backward
retains BF16 Q/K/V/P/dS and tensor-core dO, raw-FP32-dO Delta, and FP32
shared-gradient accumulators. Sequence tails and one-key rows are masked.
"""

import tilelang
import tilelang.language as T
import torch

_FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}


@tilelang.jit(out_idx=[2], pass_configs=_FAST_MATH)
def _delta_kernel(B, N, HQ, block=32):
    N = T.dynamic("sequence_length") if N is None else N
    @T.prim_func
    def main(
        OUT: T.Tensor([B, N, HQ, 128], T.float32),
        DO: T.Tensor([B, N, HQ, 128], T.float32),
        Delta: T.Tensor([B, HQ, N], T.float32),
    ):
        with T.Kernel(HQ, T.ceildiv(N, block), B, threads=128) as (h, bx, b):
            out = T.alloc_fragment([block, 128], T.float32)
            do = T.alloc_fragment([block, 128], T.float32)
            product = T.alloc_fragment([block, 128], T.float32)
            delta = T.alloc_fragment([block], T.float32)
            T.copy(OUT[b, bx * block : (bx + 1) * block, h, :], out)
            T.copy(DO[b, bx * block : (bx + 1) * block, h, :], do)
            for i, d in T.Parallel(block, 128):
                product[i, d] = out[i, d] * do[i, d]
            T.reduce_sum(product, delta, dim=1)
            T.copy(delta, Delta[b, h, bx * block : (bx + 1) * block])

    return main


def _dq_layout(dq):
    return T.Layout(
        dq.shape,
        lambda b, i, h, d: [b, i // 8, h, d // 8, d % 2, 4 * (i % 8) + (d % 8) // 2],
    )


@tilelang.jit(pass_configs=_FAST_MATH)
def _restore_dq(B, N, HQ, NP):
    N = T.dynamic("sequence_length") if N is None else N
    NP = T.dynamic("gradient_length") if NP is None else NP
    @T.prim_func
    def main(
        DQ: T.Tensor([B, NP, HQ, 128], T.float32),
        OUT: T.Tensor([B, N, HQ, 128], T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, 64), HQ, B, threads=128) as (bx, h, b):
            T.annotate_layout({DQ: _dq_layout(DQ)})
            T.copy(DQ[b, bx * 64 : (bx + 1) * 64, h, :], OUT[b, bx * 64 : (bx + 1) * 64, h, :])

    return main


@tilelang.jit(pass_configs=_FAST_MATH)
def _key_owned_backward(B, N, HQ, HK, NP, W, scale, BK=128, BQ=32):
    runtime_length = N is None
    N = T.dynamic("sequence_length") if N is None else N
    NP = T.dynamic("gradient_length") if NP is None else NP
    W = N if W == 0 else W
    @T.prim_func
    def main(
        Q: T.Tensor([B, N, HQ, 128], T.bfloat16),
        K: T.Tensor([B, N, HK, 128], T.bfloat16),
        V: T.Tensor([B, N, HK, 128], T.bfloat16),
        DO: T.Tensor([B, N, HQ, 128], T.bfloat16),
        LSE: T.Tensor([B, N, HQ], T.float32),
        Delta: T.Tensor([B, HQ, N], T.float32),
        DQ: T.Tensor([B, NP, HQ, 128], T.float32),
        DK: T.Tensor([B, N, HK, 128], T.float32),
        DV: T.Tensor([B, N, HK, 128], T.float32),
    ):
        with T.Kernel(HQ, T.ceildiv(N, BK), B, threads=256) as (h, bx, b):
            k = T.alloc_shared([BK, 128], T.bfloat16)
            v = T.alloc_shared([BK, 128], T.bfloat16)
            q = T.alloc_shared([BQ, 128], T.bfloat16)
            do = T.alloc_shared([BQ, 128], T.bfloat16)
            score = T.alloc_fragment([BK, BQ], T.float32)
            dp = T.alloc_fragment([BK, BQ], T.float32)
            prob = T.alloc_fragment([BK, BQ], T.bfloat16)
            ds = T.alloc_fragment([BK, BQ], T.bfloat16)
            ds_shared = T.alloc_shared([BK, BQ], T.bfloat16)
            lse = T.alloc_shared([BQ], T.float32)
            delta = T.alloc_shared([BQ], T.float32)
            dv = T.alloc_fragment([BK, 128], T.float32)
            dk = T.alloc_fragment([BK, 128], T.float32)
            dq = T.alloc_fragment([BQ, 128], T.float32)
            dv_shared = T.alloc_shared([BK, 128], T.float32)
            dk_shared = T.alloc_shared([BK, 128], T.float32)
            T.annotate_layout({DQ: _dq_layout(DQ)})
            T.copy(K[b, bx * BK : (bx + 1) * BK, h // (HQ // HK), :], k)
            T.copy(V[b, bx * BK : (bx + 1) * BK, h // (HQ // HK), :], v)
            T.clear(dv)
            T.clear(dk)
            start = bx * BK // BQ
            end = T.min(T.ceildiv((bx + 1) * BK + W - 1, BQ), T.ceildiv(N, BQ))
            for tile in T.Pipelined(start, end, num_stages=2):
                T.copy(Q[b, tile * BQ : (tile + 1) * BQ, h, :], q)
                T.clear(score)
                T.gemm(k, q, score, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(LSE[b, tile * BQ : (tile + 1) * BQ, h], lse)
                for i, j in T.Parallel(BK, BQ):
                    score[i, j] = T.if_then_else(
                        bx * BK + i <= tile * BQ + j
                        and bx * BK + i > tile * BQ + j - W
                        and tile * BQ + j < N
                        and bx * BK + i < N,
                        T.exp2(score[i, j] * scale * 1.4426950408889634 - lse[j]),
                        0,
                    )
                T.copy(DO[b, tile * BQ : (tile + 1) * BQ, h, :], do)
                T.clear(dp)
                T.gemm(v, do, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(score, prob)
                T.gemm(prob, do, dv, policy=T.GemmWarpPolicy.FullRow)
                if runtime_length:
                    # Delta's head stride is N FP32 values. An arbitrary N
                    # need not satisfy TMA's 16-byte stride alignment.
                    for j in T.Parallel(BQ):
                        delta[j] = T.if_then_else(
                            tile * BQ + j < N, Delta[b, h, tile * BQ + j], 0,
                        )
                else:
                    T.copy(Delta[b, h, tile * BQ : (tile + 1) * BQ], delta)
                for i, j in T.Parallel(BK, BQ):
                    ds[i, j] = T.if_then_else(
                        tile * BQ + j == 0 or W == 1,
                        0,
                        score[i, j] * (dp[i, j] - delta[j]) * scale,
                    )
                T.gemm(ds, q, dk, policy=T.GemmWarpPolicy.FullRow)
                T.copy(ds, ds_shared)
                T.clear(dq)
                T.gemm(ds_shared, k, dq, transpose_A=True)
                for i, d in T.Parallel(BQ, 128):
                    if tile * BQ + i < N:
                        T.atomic_add(DQ[b, tile * BQ + i, h, d], dq[i, d])
            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.atomic_add(DV[b, bx * BK : (bx + 1) * BK, h // (HQ // HK), :], dv_shared)
            T.atomic_add(DK[b, bx * BK : (bx + 1) * BK, h // (HQ // HK), :], dk_shared)

    return main


class _GQAAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window, scale, forward, dynamic_length):
        q, k, v = [x.contiguous() for x in (q, k, v)]
        b, n, h, _ = q.shape
        hk = k.shape[2]
        out, lse = forward(q, k, v, scaling=scale, long_window=window)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.cfg = (b, n, h, hk, (n + 7) // 8 * 8, window, scale)
        ctx.dynamic_length = dynamic_length
        return out

    @staticmethod
    def backward(ctx, gradient_output):
        q, k, v, out, lse = ctx.saved_tensors
        b, n, h, _, padded_n, _, _ = ctx.cfg
        length = None if ctx.dynamic_length else n
        gradient_length = None if ctx.dynamic_length else padded_n
        do = gradient_output.float().contiguous()
        delta = _delta_kernel(b, length, h)(out, do)
        dq = torch.zeros((b, padded_n, h, 128), device=q.device, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        cfg = list(ctx.cfg)
        if ctx.dynamic_length:
            cfg[1], cfg[4] = None, None
            cfg[5] = 0 if cfg[5] >= n else cfg[5]
        _key_owned_backward(*cfg)(q, k, v, do.to(torch.bfloat16), lse, delta, dq, dk, dv)
        restored_dq = torch.empty_like(q)
        _restore_dq(b, length, h, gradient_length)(dq, restored_dq)
        return (
            restored_dq, dk.to(k.dtype), dv.to(v.dtype),
            None, None, None, None,
        )


def gqa_attention(q, k, v, *, scaling, long_window, forward, dynamic_length=False):
    """Use a native forward callback and the qualified GQA backward.

    Geometry is validated by the shared attention entry point. ``forward``
    returns FP32 output [B,N,HQ,128] and log2 LSE [B,N,HQ].
    """
    return _GQAAttention.apply(q, k, v, long_window, scaling, forward, dynamic_length)
