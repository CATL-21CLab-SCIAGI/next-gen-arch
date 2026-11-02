"""Triadic/GDN gated delta rule in TileLang (chunks of 64 tokens, D = V = 128).

The joint key is k2_t (x) k_t, so the state S is (E*D, V) with slice e on rows
e*D:(e+1)*D. Within a chunk, with chunk-local cumulative log decay gc and its
last row gL:

    M_ts  = sum_e k2_te k2_se exp(gc_te - gc_se),  Mp_ts = sum_e q2_te k2_se exp(...)
    A     = (I + StrictLower(beta_t <k_t, k_s> M_ts))^-1
    Pm    = Lower(<q_t, k_s> Mp_ts)
    W     = v - sum_e diag(a_e) K S_e                       a = k2 exp(gc)
    U     = A diag(beta) W
    o     = scale (sum_e diag(b_e) Q S_e + Pm U)            b = q2 exp(gc)
    S_e  <- exp(gL_e) S_e + K^T diag(c_e) U                 c = k2 exp(gL - gc)

The recurrence is separable over value columns: the two scans own a column
block of S and carry only the state-dependent products; row coefficients
scale the small value-side operands, so K and Q enter the tensor cores
unscaled. Every other kernel owns one chunk. Torch differentiates
the coefficients a, b, c, exp(gL) and the cumulative decay gc. Matrix products
take BF16 operands with FP32 accumulation; S, dS and the triangular solve stay
FP32. Per-token coefficients are head-major ([B, H, T, E]) so that one chunk's
rows are contiguous.
"""

import tilelang
import tilelang.language as T
import torch

CHUNK = 64
DIM = 128
_FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}


def _geometry(features):
    """Value-column block and prefetch depth of the scans, within shared memory.

    One MMA column tile per program shortens each chunk's serial chain."""
    return 8, (2 if features <= 8 else 1)


@tilelang.jit(pass_configs=_FAST_MATH)
def _prepare_kernel(B, H, E):
    N = T.dynamic("tokens")
    C = CHUNK

    @T.prim_func
    def main(
        Q: T.Tensor([B, N, H, DIM], T.bfloat16),
        K: T.Tensor([B, N, H, DIM], T.bfloat16),
        K2: T.Tensor([B, H, N, E], T.float32),
        Q2: T.Tensor([B, H, N, E], T.float32),
        GC: T.Tensor([B, H, N, E], T.float32),
        BETA: T.Tensor([B, H, N], T.float32),
        A_OUT: T.Tensor([B, H, N, C], T.bfloat16),
        PM_OUT: T.Tensor([B, H, N, C], T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, C), H, B, threads=128) as (n, h, b):
            rows = n * C
            k = T.alloc_shared([C, DIM], T.bfloat16)
            q = T.alloc_shared([C, DIM], T.bfloat16)
            k2 = T.alloc_shared([C, E], T.float32)
            q2 = T.alloc_shared([C, E], T.float32)
            gc = T.alloc_shared([C, E], T.float32)
            beta = T.alloc_shared([C], T.float32)
            p = T.alloc_fragment([C, C], T.float32)
            pq = T.alloc_fragment([C, C], T.float32)
            m = T.alloc_fragment([C, C], T.float32)
            mp = T.alloc_fragment([C, C], T.float32)
            lower = T.alloc_shared([C, C], T.float32)
            inverse = T.alloc_shared([C, C], T.bfloat16)
            column = T.alloc_local([C], T.float32)
            T.copy(K[b, rows : rows + C, h, :], k)
            T.copy(Q[b, rows : rows + C, h, :], q)
            T.copy(K2[b, h, rows : rows + C, :], k2)
            T.copy(Q2[b, h, rows : rows + C, :], q2)
            T.copy(GC[b, h, rows : rows + C, :], gc)
            T.copy(BETA[b, h, rows : rows + C], beta)
            T.gemm(k, k, p, transpose_B=True, clear_accum=True)
            T.gemm(q, k, pq, transpose_B=True, clear_accum=True)
            T.clear(m)
            T.clear(mp)
            for e in T.serial(E):
                for t, s in T.Parallel(C, C):
                    decay = T.exp(T.min(gc[t, e] - gc[s, e], 0))
                    m[t, s] += k2[t, e] * k2[s, e] * decay
                    mp[t, s] += q2[t, e] * k2[s, e] * decay
            for t, s in T.Parallel(C, C):
                lower[t, s] = T.if_then_else(t > s, beta[t] * p[t, s] * m[t, s], 0)
                pq[t, s] = T.if_then_else(t >= s, pq[t, s] * mp[t, s], 0)
            T.copy(pq, PM_OUT[b, h, rows : rows + C, :])
            # Forward substitution: thread j owns column j of (I + L)^-1 in registers.
            j = T.get_thread_binding()
            if j < C:
                for i in T.unroll(C):
                    column[i] = T.if_then_else(i == j, 1.0, 0.0)
                    for r in T.unroll(C):
                        if r < i:
                            column[i] -= lower[i, r] * column[r]
                for i in T.unroll(C):
                    inverse[i, j] = column[i]
            T.copy(inverse, A_OUT[b, h, rows : rows + C, :])

    return main


@tilelang.jit(pass_configs=_FAST_MATH)
def _forward_scan_kernel(B, H, E, VB, stages):
    N, NC = T.dynamic("tokens, chunks")
    C, R = CHUNK, E * DIM

    @T.prim_func
    def main(
        K: T.Tensor([B, N, H, DIM], T.bfloat16),
        V: T.Tensor([B, N, H, DIM], T.bfloat16),
        CA: T.Tensor([B, H, N, E], T.float32),
        CC: T.Tensor([B, H, N, E], T.float32),
        BETA: T.Tensor([B, H, N], T.float32),
        DECAY: T.Tensor([B, H, NC, E], T.float32),
        A: T.Tensor([B, H, N, C], T.bfloat16),
        STATES: T.Tensor([B, NC, H, R, DIM], T.bfloat16),
        W_OUT: T.Tensor([B, N, H, DIM], T.bfloat16),
        U_OUT: T.Tensor([B, N, H, DIM], T.bfloat16),
    ):
        with T.Kernel(DIM // VB, H, B, threads=128) as (vb, h, b):
            state = T.alloc_shared([R, VB], T.float32)
            mirror = T.alloc_shared([R, VB], T.bfloat16)
            k = T.alloc_shared([C, DIM], T.bfloat16)
            v = T.alloc_shared([C, VB], T.bfloat16)
            a = T.alloc_shared([C, C], T.bfloat16)
            ca = T.alloc_shared([C, E], T.float32)
            cc = T.alloc_shared([C, E], T.float32)
            beta = T.alloc_shared([C], T.float32)
            decay = T.alloc_shared([E], T.float32)
            x = T.alloc_shared([C, VB], T.bfloat16)
            writes = T.alloc_shared([E * C, VB], T.bfloat16)
            w = T.alloc_fragment([C, VB], T.float32)
            y = T.alloc_fragment([C, VB], T.float32)
            u = T.alloc_fragment([C, VB], T.float32)
            update = T.alloc_fragment([DIM, VB], T.float32)
            T.clear(state)
            for n in T.Pipelined(NC, num_stages=stages):
                rows = n * C
                T.copy(K[b, rows : rows + C, h, :], k)
                T.copy(V[b, rows : rows + C, h, vb * VB : (vb + 1) * VB], v)
                T.copy(A[b, h, rows : rows + C, :], a)
                T.copy(CA[b, h, rows : rows + C, :], ca)
                T.copy(CC[b, h, rows : rows + C, :], cc)
                T.copy(BETA[b, h, rows : rows + C], beta)
                T.copy(DECAY[b, h, n, :], decay)
                T.copy(state, mirror)
                T.copy(mirror, STATES[b, n, h, :, vb * VB : (vb + 1) * VB])
                # Row coefficients scale the small value-side operand, never K.
                T.copy(v, w)
                for e in T.serial(E):
                    T.gemm(k, mirror[e * DIM : (e + 1) * DIM, :], y, clear_accum=True)
                    for t, j in T.Parallel(C, VB):
                        w[t, j] -= ca[t, e] * y[t, j]
                T.copy(w, W_OUT[b, rows : rows + C, h, vb * VB : (vb + 1) * VB])
                for t, j in T.Parallel(C, VB):
                    x[t, j] = beta[t] * w[t, j]
                T.gemm(a, x, u, clear_accum=True)
                T.copy(u, U_OUT[b, rows : rows + C, h, vb * VB : (vb + 1) * VB])
                for e, t, j in T.Parallel(E, C, VB):
                    writes[e * C + t, j] = cc[t, e] * u[t, j]
                for e in T.serial(E):
                    T.gemm(k, writes[e * C : (e + 1) * C, :], update, transpose_A=True,
                           clear_accum=True)
                    for d, j in T.Parallel(DIM, VB):
                        state[e * DIM + d, j] = decay[e] * state[e * DIM + d, j] + update[d, j]

    return main


@tilelang.jit(pass_configs=_FAST_MATH)
def _output_kernel(B, H, E, scale):
    N, NC = T.dynamic("tokens, chunks")
    C, R = CHUNK, E * DIM

    @T.prim_func
    def main(
        Q: T.Tensor([B, N, H, DIM], T.bfloat16),
        CB: T.Tensor([B, H, N, E], T.float32),
        PM: T.Tensor([B, H, N, C], T.bfloat16),
        U: T.Tensor([B, N, H, DIM], T.bfloat16),
        STATES: T.Tensor([B, NC, H, R, DIM], T.bfloat16),
        OUT: T.Tensor([B, N, H, DIM], T.bfloat16),
    ):
        with T.Kernel(NC, H, B, threads=128) as (n, h, b):
            rows = n * C
            q = T.alloc_shared([C, DIM], T.bfloat16)
            read = T.alloc_shared([C, DIM], T.bfloat16)
            s = T.alloc_shared([DIM, DIM], T.bfloat16)
            pm = T.alloc_shared([C, C], T.bfloat16)
            u = T.alloc_shared([C, DIM], T.bfloat16)
            cb = T.alloc_shared([C, E], T.float32)
            out = T.alloc_fragment([C, DIM], T.float32)
            T.copy(Q[b, rows : rows + C, h, :], q)
            T.copy(PM[b, h, rows : rows + C, :], pm)
            T.copy(U[b, rows : rows + C, h, :], u)
            T.copy(CB[b, h, rows : rows + C, :], cb)
            T.gemm(pm, u, out, clear_accum=True)
            for e in T.Pipelined(E, num_stages=2):
                T.copy(STATES[b, n, h, e * DIM : (e + 1) * DIM, :], s)
                for t, d in T.Parallel(C, DIM):
                    read[t, d] = cb[t, e] * q[t, d]
                T.gemm(read, s, out)
            for t, d in T.Parallel(C, DIM):
                out[t, d] *= scale
            T.copy(out, OUT[b, rows : rows + C, h, :])

    return main


@tilelang.jit(pass_configs=_FAST_MATH)
def _backward_scan_kernel(B, H, E, VB, stages, scale):
    N, NC = T.dynamic("tokens, chunks")
    C, R = CHUNK, E * DIM

    @T.prim_func
    def main(
        Q: T.Tensor([B, N, H, DIM], T.bfloat16),
        K: T.Tensor([B, N, H, DIM], T.bfloat16),
        DO: T.Tensor([B, N, H, DIM], T.bfloat16),
        CA: T.Tensor([B, H, N, E], T.float32),
        CB: T.Tensor([B, H, N, E], T.float32),
        CC: T.Tensor([B, H, N, E], T.float32),
        BETA: T.Tensor([B, H, N], T.float32),
        DECAY: T.Tensor([B, H, NC, E], T.float32),
        A: T.Tensor([B, H, N, C], T.bfloat16),
        PM: T.Tensor([B, H, N, C], T.bfloat16),
        DSTATES: T.Tensor([B, NC, H, R, DIM], T.bfloat16),
        DU: T.Tensor([B, N, H, DIM], T.float32),
        DV: T.Tensor([B, N, H, DIM], T.float32),
    ):
        with T.Kernel(DIM // VB, H, B, threads=128) as (vb, h, b):
            dstate = T.alloc_shared([R, VB], T.float32)
            mirror = T.alloc_shared([R, VB], T.bfloat16)
            q = T.alloc_shared([C, DIM], T.bfloat16)
            k = T.alloc_shared([C, DIM], T.bfloat16)
            do = T.alloc_shared([C, VB], T.bfloat16)
            a = T.alloc_shared([C, C], T.bfloat16)
            pm = T.alloc_shared([C, C], T.bfloat16)
            ca = T.alloc_shared([C, E], T.float32)
            cb = T.alloc_shared([C, E], T.float32)
            cc = T.alloc_shared([C, E], T.float32)
            beta = T.alloc_shared([C], T.float32)
            decay = T.alloc_shared([E], T.float32)
            dus = T.alloc_shared([C, VB], T.bfloat16)
            rhs = T.alloc_shared([E * 2 * C, VB], T.bfloat16)
            du = T.alloc_fragment([C, VB], T.float32)
            y = T.alloc_fragment([C, VB], T.float32)
            dx = T.alloc_fragment([C, VB], T.float32)
            update = T.alloc_fragment([DIM, VB], T.float32)
            T.clear(dstate)
            for step in T.Pipelined(NC, num_stages=stages):
                n = NC - 1 - step
                rows = n * C
                T.copy(Q[b, rows : rows + C, h, :], q)
                T.copy(K[b, rows : rows + C, h, :], k)
                T.copy(DO[b, rows : rows + C, h, vb * VB : (vb + 1) * VB], do)
                T.copy(A[b, h, rows : rows + C, :], a)
                T.copy(PM[b, h, rows : rows + C, :], pm)
                T.copy(CA[b, h, rows : rows + C, :], ca)
                T.copy(CB[b, h, rows : rows + C, :], cb)
                T.copy(CC[b, h, rows : rows + C, :], cc)
                T.copy(BETA[b, h, rows : rows + C], beta)
                T.copy(DECAY[b, h, n, :], decay)
                T.copy(dstate, mirror)
                T.copy(mirror, DSTATES[b, n, h, :, vb * VB : (vb + 1) * VB])
                # dU = scale Pm^T dO + sum_e diag(c_e) K dS_e'.
                T.gemm(pm, do, du, transpose_A=True, clear_accum=True)
                for t, j in T.Parallel(C, VB):
                    du[t, j] *= scale
                for e in T.serial(E):
                    T.gemm(k, mirror[e * DIM : (e + 1) * DIM, :], y, clear_accum=True)
                    for t, j in T.Parallel(C, VB):
                        du[t, j] += cc[t, e] * y[t, j]
                T.copy(du, DU[b, rows : rows + C, h, vb * VB : (vb + 1) * VB])
                T.copy(du, dus)
                # dW = diag(beta) A^T dU, which is also dv.
                T.gemm(a, dus, dx, transpose_A=True, clear_accum=True)
                for t, j in T.Parallel(C, VB):
                    dx[t, j] *= beta[t]
                T.copy(dx, DV[b, rows : rows + C, h, vb * VB : (vb + 1) * VB])
                # dS_e = exp(gL_e) dS_e' + scale Q^T diag(b_e) dO - K^T diag(a_e) dW.
                for e, t, j in T.Parallel(E, C, VB):
                    rhs[e * 2 * C + t, j] = scale * cb[t, e] * do[t, j]
                    rhs[e * 2 * C + C + t, j] = -ca[t, e] * dx[t, j]
                for e in T.serial(E):
                    T.gemm(q, rhs[e * 2 * C : e * 2 * C + C, :], update, transpose_A=True,
                           clear_accum=True)
                    T.gemm(k, rhs[e * 2 * C + C : (e + 1) * 2 * C, :], update, transpose_A=True)
                    for d, j in T.Parallel(DIM, VB):
                        dstate[e * DIM + d, j] = decay[e] * dstate[e * DIM + d, j] + update[d, j]

    return main


@tilelang.jit(pass_configs=_FAST_MATH)
def _inter_chunk_kernel(B, H, E, scale):
    N, NC = T.dynamic("tokens, chunks")
    C, R = CHUNK, E * DIM

    @T.prim_func
    def main(
        Q: T.Tensor([B, N, H, DIM], T.bfloat16),
        K: T.Tensor([B, N, H, DIM], T.bfloat16),
        DO: T.Tensor([B, N, H, DIM], T.bfloat16),
        DV: T.Tensor([B, N, H, DIM], T.float32),
        U: T.Tensor([B, N, H, DIM], T.bfloat16),
        CA: T.Tensor([B, H, N, E], T.float32),
        CB: T.Tensor([B, H, N, E], T.float32),
        CC: T.Tensor([B, H, N, E], T.float32),
        STATES: T.Tensor([B, NC, H, R, DIM], T.bfloat16),
        DSTATES: T.Tensor([B, NC, H, R, DIM], T.bfloat16),
        DQ: T.Tensor([B, N, H, DIM], T.float32),
        DK: T.Tensor([B, N, H, DIM], T.float32),
        DCA: T.Tensor([B, H, N, E], T.float32),
        DCB: T.Tensor([B, H, N, E], T.float32),
        DCC: T.Tensor([B, H, N, E], T.float32),
        DDECAY: T.Tensor([B, H, NC, E], T.float32),
    ):
        with T.Kernel(NC, H, B, threads=256) as (n, h, b):
            rows = n * C
            k = T.alloc_shared([C, DIM], T.bfloat16)
            q = T.alloc_shared([C, DIM], T.bfloat16)
            do = T.alloc_shared([C, DIM], T.bfloat16)
            dv = T.alloc_shared([C, DIM], T.bfloat16)
            u = T.alloc_shared([C, DIM], T.bfloat16)
            s = T.alloc_shared([DIM, DIM], T.bfloat16)
            ds = T.alloc_shared([DIM, DIM], T.bfloat16)
            ca = T.alloc_shared([C, E], T.float32)
            cb = T.alloc_shared([C, E], T.float32)
            cc = T.alloc_shared([C, E], T.float32)
            y = T.alloc_fragment([C, DIM], T.float32)
            product = T.alloc_fragment([C, DIM], T.float32)
            row = T.alloc_fragment([C], T.float32)
            row2 = T.reshape(row, [1, C])
            total = T.alloc_fragment([1], T.float32)
            overlap = T.alloc_fragment([1], T.float32)
            dq = T.alloc_fragment([C, DIM], T.float32)
            dk = T.alloc_fragment([C, DIM], T.float32)
            T.copy(K[b, rows : rows + C, h, :], k)
            T.copy(Q[b, rows : rows + C, h, :], q)
            T.copy(DO[b, rows : rows + C, h, :], do)
            T.copy(DV[b, rows : rows + C, h, :], dv)
            T.copy(U[b, rows : rows + C, h, :], u)
            T.copy(CA[b, h, rows : rows + C, :], ca)
            T.copy(CB[b, h, rows : rows + C, :], cb)
            T.copy(CC[b, h, rows : rows + C, :], cc)
            T.clear(dq)
            T.clear(dk)
            for e in T.serial(E):
                T.copy(STATES[b, n, h, e * DIM : (e + 1) * DIM, :], s)
                T.copy(DSTATES[b, n, h, e * DIM : (e + 1) * DIM, :], ds)
                # dO S_e^T feeds q and the read coefficient b.
                T.gemm(do, s, y, transpose_B=True, clear_accum=True)
                for t, d in T.Parallel(C, DIM):
                    dq[t, d] += scale * cb[t, e] * y[t, d]
                    product[t, d] = q[t, d] * y[t, d]
                T.reduce_sum(product, row, dim=1)
                for t in T.Parallel(C):
                    DCB[b, h, rows + t, e] = scale * row[t]
                # dW S_e^T feeds k and the erase coefficient a (W = v - a K S).
                T.gemm(dv, s, y, transpose_B=True, clear_accum=True)
                for t, d in T.Parallel(C, DIM):
                    dk[t, d] -= ca[t, e] * y[t, d]
                    product[t, d] = k[t, d] * y[t, d]
                T.reduce_sum(product, row, dim=1)
                for t in T.Parallel(C):
                    DCA[b, h, rows + t, e] = -row[t]
                # U dS_e'^T feeds k and the write coefficient c.
                T.gemm(u, ds, y, transpose_B=True, clear_accum=True)
                for t, d in T.Parallel(C, DIM):
                    dk[t, d] += cc[t, e] * y[t, d]
                    product[t, d] = k[t, d] * y[t, d]
                T.reduce_sum(product, row, dim=1)
                for t in T.Parallel(C):
                    DCC[b, h, rows + t, e] = row[t]
                # The decay multiplies S_e: its gradient is <S_e, dS_e'>.
                T.clear(overlap)
                for half in T.serial(DIM // C):
                    for t, d in T.Parallel(C, DIM):
                        product[t, d] = (s[half * C + t, d].astype(T.float32)
                                         * ds[half * C + t, d].astype(T.float32))
                    T.reduce_sum(product, row, dim=1)
                    T.reduce_sum(row2, total, dim=1)
                    overlap[0] += total[0]
                DDECAY[b, h, n, e] = overlap[0]
            T.copy(dq, DQ[b, rows : rows + C, h, :])
            T.copy(dk, DK[b, rows : rows + C, h, :])

    return main


@tilelang.jit(pass_configs=_FAST_MATH)
def _intra_chunk_kernel(B, H, E, scale):
    N = T.dynamic("tokens")
    C = CHUNK

    @T.prim_func
    def main(
        Q: T.Tensor([B, N, H, DIM], T.bfloat16),
        K: T.Tensor([B, N, H, DIM], T.bfloat16),
        DO: T.Tensor([B, N, H, DIM], T.bfloat16),
        U: T.Tensor([B, N, H, DIM], T.bfloat16),
        W: T.Tensor([B, N, H, DIM], T.bfloat16),
        DU: T.Tensor([B, N, H, DIM], T.float32),
        A: T.Tensor([B, H, N, C], T.bfloat16),
        K2: T.Tensor([B, H, N, E], T.float32),
        Q2: T.Tensor([B, H, N, E], T.float32),
        GC: T.Tensor([B, H, N, E], T.float32),
        BETA: T.Tensor([B, H, N], T.float32),
        DQ: T.Tensor([B, N, H, DIM], T.float32),
        DK: T.Tensor([B, N, H, DIM], T.float32),
        DK2: T.Tensor([B, H, N, E], T.float32),
        DQ2: T.Tensor([B, H, N, E], T.float32),
        DGC: T.Tensor([B, H, N, E], T.float32),
        DBETA: T.Tensor([B, H, N], T.float32),
    ):
        with T.Kernel(T.ceildiv(N, C), H, B, threads=256) as (n, h, b):
            rows = n * C
            k = T.alloc_shared([C, DIM], T.bfloat16)
            q = T.alloc_shared([C, DIM], T.bfloat16)
            do = T.alloc_shared([C, DIM], T.bfloat16)
            u = T.alloc_shared([C, DIM], T.bfloat16)
            x = T.alloc_shared([C, DIM], T.bfloat16)
            du = T.alloc_shared([C, DIM], T.bfloat16)
            a = T.alloc_shared([C, C], T.bfloat16)
            square = T.alloc_shared([C, C], T.bfloat16)
            p = T.alloc_shared([C, C], T.float32)
            pq = T.alloc_shared([C, C], T.float32)
            dl = T.alloc_shared([C, C], T.float32)
            dpm = T.alloc_shared([C, C], T.float32)
            k2 = T.alloc_shared([C, E], T.float32)
            q2 = T.alloc_shared([C, E], T.float32)
            gc = T.alloc_shared([C, E], T.float32)
            beta = T.alloc_shared([C], T.float32)
            dbeta = T.alloc_shared([C], T.float32)
            acc = T.alloc_fragment([C, C], T.float32)
            m = T.alloc_fragment([C, C], T.float32)
            mp = T.alloc_fragment([C, C], T.float32)
            wide = T.alloc_fragment([C, DIM], T.float32)
            dq = T.alloc_fragment([C, DIM], T.float32)
            dk = T.alloc_fragment([C, DIM], T.float32)
            wide_row = T.alloc_fragment([C], T.float32)
            row = T.alloc_fragment([C], T.float32)
            row2 = T.alloc_fragment([C], T.float32)
            col = T.alloc_fragment([C], T.float32)
            col2 = T.alloc_fragment([C], T.float32)
            T.copy(K[b, rows : rows + C, h, :], k)
            T.copy(Q[b, rows : rows + C, h, :], q)
            T.copy(DO[b, rows : rows + C, h, :], do)
            T.copy(U[b, rows : rows + C, h, :], u)
            T.copy(W[b, rows : rows + C, h, :], x)
            T.copy(DU[b, rows : rows + C, h, :], du)
            T.copy(A[b, h, rows : rows + C, :], a)
            T.copy(K2[b, h, rows : rows + C, :], k2)
            T.copy(Q2[b, h, rows : rows + C, :], q2)
            T.copy(GC[b, h, rows : rows + C, :], gc)
            T.copy(BETA[b, h, rows : rows + C], beta)
            T.gemm(k, k, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, p)
            T.gemm(q, k, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, pq)
            # Output read: o += scale Pm U with Pm = Lower(Pq Mp).
            T.gemm(do, u, acc, transpose_B=True, clear_accum=True)
            for t, s in T.Parallel(C, C):
                dpm[t, s] = T.if_then_else(t >= s, scale * acc[t, s], 0)
            # U = A diag(beta) W: dX = A^T dU and dbeta = rowsum(W dX).
            T.gemm(a, du, wide, transpose_A=True, clear_accum=True)
            for t, d in T.Parallel(C, DIM):
                wide[t, d] = x[t, d] * wide[t, d]
            T.reduce_sum(wide, wide_row, dim=1)
            for t in T.Parallel(C):
                dbeta[t] = wide_row[t]
            for t, d in T.Parallel(C, DIM):
                x[t, d] = beta[t] * x[t, d]
            # dA = dU X^T; A = (I + L)^-1 gives dL = -A^T dA A^T.
            T.gemm(du, x, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, square)
            T.gemm(a, square, acc, transpose_A=True, clear_accum=True)
            T.copy(acc, square)
            T.gemm(square, a, acc, transpose_B=True, clear_accum=True)
            for t, s in T.Parallel(C, C):
                dl[t, s] = T.if_then_else(t > s, -acc[t, s], 0)
            T.clear(m)
            T.clear(mp)
            for e in T.serial(E):
                for t, s in T.Parallel(C, C):
                    decay = T.exp(T.min(gc[t, e] - gc[s, e], 0))
                    m[t, s] += k2[t, e] * k2[s, e] * decay
                    mp[t, s] += q2[t, e] * k2[s, e] * decay
            # L = beta_t P M with P = K K^T.
            for t, s in T.Parallel(C, C):
                acc[t, s] = dl[t, s] * p[t, s] * m[t, s]
            T.reduce_sum(acc, row, dim=1)
            for t in T.Parallel(C):
                DBETA[b, h, rows + t] = dbeta[t] + row[t]
            T.copy(DQ[b, rows : rows + C, h, :], dq)
            T.copy(DK[b, rows : rows + C, h, :], dk)
            for t, s in T.Parallel(C, C):
                square[t, s] = dpm[t, s] * mp[t, s]
            T.gemm(square, k, dq)
            T.gemm(square, q, dk, transpose_A=True)
            for t, s in T.Parallel(C, C):
                square[t, s] = dl[t, s] * beta[t] * m[t, s]
            T.gemm(square, k, dk)
            T.gemm(square, k, dk, transpose_A=True)
            T.copy(dq, DQ[b, rows : rows + C, h, :])
            T.copy(dk, DK[b, rows : rows + C, h, :])
            # dM = dL beta_t P and dMp = dPm Pq reach k2, q2 and gc through each slice's decay.
            for t, s in T.Parallel(C, C):
                dl[t, s] = dl[t, s] * beta[t] * p[t, s]
                dpm[t, s] = dpm[t, s] * pq[t, s]
            for e in T.serial(E):
                for t, s in T.Parallel(C, C):
                    acc[t, s] = dl[t, s] * T.exp(T.min(gc[t, e] - gc[s, e], 0)) * k2[s, e]
                T.reduce_sum(acc, row, dim=1)
                for t, s in T.Parallel(C, C):
                    acc[t, s] = dpm[t, s] * T.exp(T.min(gc[t, e] - gc[s, e], 0)) * k2[s, e]
                T.reduce_sum(acc, row2, dim=1)
                # Column sums over t, filled transposed so every reduction is along rows.
                for s, t in T.Parallel(C, C):
                    acc[s, t] = dl[t, s] * T.exp(T.min(gc[t, e] - gc[s, e], 0)) * k2[t, e]
                T.reduce_sum(acc, col, dim=1)
                for s, t in T.Parallel(C, C):
                    acc[s, t] = dpm[t, s] * T.exp(T.min(gc[t, e] - gc[s, e], 0)) * q2[t, e]
                T.reduce_sum(acc, col2, dim=1)
                for t in T.Parallel(C):
                    DK2[b, h, rows + t, e] = row[t] + col[t] + col2[t]
                    DQ2[b, h, rows + t, e] = row2[t]
                    DGC[b, h, rows + t, e] = (
                        k2[t, e] * (row[t] - col[t] - col2[t]) + q2[t, e] * row2[t]
                    )

    return main


class _ChunkedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k2, q2, gc, beta, ca, cb, cc, decay, scale):
        batch, tokens, heads, _ = q.shape
        features = k2.shape[-1]
        block, stages = _geometry(features)
        a = torch.empty(batch, heads, tokens, CHUNK, device=q.device, dtype=torch.bfloat16)
        pm = torch.empty_like(a)
        _prepare_kernel(batch, heads, features)(q, k, k2, q2, gc, beta, a, pm)
        states = torch.empty(batch, tokens // CHUNK, heads, features * DIM, DIM,
                             device=q.device, dtype=torch.bfloat16)
        w, u, out = (torch.empty_like(v) for _ in range(3))
        _forward_scan_kernel(batch, heads, features, block, stages)(
            k, v, ca, cc, beta, decay, a, states, w, u)
        _output_kernel(batch, heads, features, scale)(q, cb, pm, u, states, out)
        ctx.save_for_backward(q, k, k2, q2, gc, beta, ca, cb, cc, decay, a, pm, states, w, u)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, gradient):
        q, k, k2, q2, gc, beta, ca, cb, cc, decay, a, pm, states, w, u = ctx.saved_tensors
        batch, _, heads, _ = q.shape
        features, scale = k2.shape[-1], ctx.scale
        block, stages = _geometry(features)
        do = gradient.to(torch.bfloat16).contiguous()
        dstates = torch.empty_like(states)
        du = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        dv = torch.empty_like(du)
        _backward_scan_kernel(batch, heads, features, block, stages, scale)(
            q, k, do, ca, cb, cc, beta, decay, a, pm, dstates, du, dv)
        dq, dk = torch.empty_like(du), torch.empty_like(du)
        dca, dcb, dcc = (torch.empty_like(ca) for _ in range(3))
        ddecay = torch.empty_like(decay)
        _inter_chunk_kernel(batch, heads, features, scale)(
            q, k, do, dv, u, ca, cb, cc, states, dstates, dq, dk, dca, dcb, dcc, ddecay)
        dk2, dq2, dgc = (torch.empty_like(k2) for _ in range(3))
        dbeta = torch.empty_like(beta)
        _intra_chunk_kernel(batch, heads, features, scale)(
            q, k, do, u, w, du, a, k2, q2, gc, beta, dq, dk, dk2, dq2, dgc, dbeta)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(q.dtype), dk2, dq2, dgc, dbeta,
                dca, dcb, dcc, ddecay, None)


def tilelang_triadic_gdn(q, k, v, k2, q2, g, beta, *, scale):
    """Rows [B, T, H, *] are independent documents; T need not be a chunk multiple.

    q/k/v BF16 with D = V = 128; k2/q2/g FP32 [B, T, H, E]; beta FP32 [B, T, H].
    Returns BF16 [B, T, H, 128]. The zero tail added to reach a chunk multiple
    follows every real token, so causality keeps it out of the output.
    """
    batch, length, heads, dim = q.shape
    features = k2.shape[-1]
    if dim != DIM or v.shape[-1] != DIM:
        raise ValueError("TileLang delta kernels require D = V = 128")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("TileLang delta q/k/v must be BF16")
    if any(t.dtype != torch.float32 for t in (k2, q2, g, beta)):
        raise ValueError("TileLang delta second axes, gates and beta must be FP32")
    if not q.is_cuda:
        raise ValueError("TileLang delta kernels require CUDA")
    pad = -length % CHUNK
    q, k, v = (torch.nn.functional.pad(t, (0, 0, 0, 0, 0, pad)).contiguous() for t in (q, k, v))
    k2, q2, g = (torch.nn.functional.pad(t, (0, 0, 0, 0, 0, pad)).transpose(1, 2)
                 for t in (k2, q2, g))
    beta = torch.nn.functional.pad(beta, (0, 0, 0, pad)).transpose(1, 2).contiguous()
    tokens = length + pad
    chunks = tokens // CHUNK
    gc = g.reshape(batch, heads, chunks, CHUNK, features).cumsum(3)
    last = gc[:, :, :, -1:]
    decay = last[:, :, :, 0].exp().contiguous()
    shape = (batch, heads, tokens, features)
    ca = (k2 * gc.exp().view(shape)).contiguous()
    cb = (q2 * gc.exp().view(shape)).contiguous()
    cc = (k2 * (last - gc).exp().view(shape)).contiguous()
    out = _ChunkedDelta.apply(q, k, v, k2.contiguous(), q2.contiguous(), gc.view(shape).contiguous(),
                              beta, ca, cb, cc, decay, float(scale))
    return out[:, :length]
