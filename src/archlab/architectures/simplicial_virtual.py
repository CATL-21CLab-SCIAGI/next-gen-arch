"""Exact global 2-simplicial attention as container cuDNN attention over virtual queries.

Row (i, t) is the virtual query q_i * k1_{i-t}; it attends query i's causal long
keys as one head of an ordinary GQA layer with HQ * w query heads. Its log
normalizer weights the joint softmax over t. Backward passes cuDNN the joint
softmax delta through a corrected output O' = o + lambda * dO with
dO . O' = weight * (do . out), so every input gradient is the joint-softmax
gradient. TileLang kernels own the memory-bound combine and gradient gathers.
Layout is [batch, sequence, heads, 128]; virtual rows are [batch, sequence, HQ*w, 128].
"""

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

_forward = torch.ops.aten._scaled_dot_product_cudnn_attention
_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward
_FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
D = 128
BI = 16


@tilelang.jit(out_idx=[3, 4], pass_configs=_FAST_MATH)
def _combine(B, HQ, HK, W):
    N = T.dynamic("sequence_length")
    WP = 1 << (W - 1).bit_length()

    @T.prim_func
    def main(
        O: T.Tensor([B, N, HQ * W, D], T.bfloat16),
        LSE: T.Tensor([B, HQ * W, N], T.float32),
        V1: T.Tensor([B, N, HK, D], T.bfloat16),
        OUT: T.Tensor([B, N, HQ, D], T.float32),
        WEIGHT: T.Tensor([B, N, HQ, W], T.float32),
    ):
        with T.Kernel(T.ceildiv(N, BI), HQ, B, threads=128) as (bx, h, b):
            logit = T.alloc_fragment([BI, WP], T.float32)
            mx = T.alloc_fragment([BI], T.float32)
            z = T.alloc_fragment([BI], T.float32)
            weight = T.alloc_shared([BI, WP], T.float32)
            acc = T.alloc_fragment([BI, D], T.float32)
            for i, t in T.Parallel(BI, WP):
                row = bx * BI + i
                logit[i, t] = T.if_then_else(
                    row < N and t <= row and t < W, LSE[b, h * W + t, row],
                    T.if_then_else(row >= N and t == 0, 0, -T.infinity(T.float32)),
                )
            T.reduce_max(logit, mx, dim=1)
            for i, t in T.Parallel(BI, WP):
                logit[i, t] = T.exp(logit[i, t] - mx[i])
            T.reduce_sum(logit, z, dim=1)
            for i, t in T.Parallel(BI, WP):
                logit[i, t] = logit[i, t] / z[i]
            T.copy(logit, weight)
            T.clear(acc)
            for t in T.serial(W):
                for i, d in T.Parallel(BI, D):
                    row = bx * BI + i
                    if row < N and t <= row:
                        acc[i, d] += (weight[i, t] * V1[b, row - t, h // (HQ // HK), d].astype(T.float32)
                                      * O[b, row, h * W + t, d].astype(T.float32))
            for i, d in T.Parallel(BI, D):
                if bx * BI + i < N:
                    OUT[b, bx * BI + i, h, d] = acc[i, d]
            for i, t in T.Parallel(BI, W):
                if bx * BI + i < N:
                    WEIGHT[b, bx * BI + i, h, t] = weight[i, t]

    return main


@tilelang.jit(out_idx=[5, 6], pass_configs=_FAST_MATH)
def _prepare(B, HQ, HK, W):
    N = T.dynamic("sequence_length")

    @T.prim_func
    def main(
        GRAD: T.Tensor([B, N, HQ, D], T.bfloat16),
        OUT: T.Tensor([B, N, HQ, D], T.float32),
        WEIGHT: T.Tensor([B, N, HQ, W], T.float32),
        O: T.Tensor([B, N, HQ * W, D], T.bfloat16),
        V1: T.Tensor([B, N, HK, D], T.bfloat16),
        DO: T.Tensor([B, N, HQ * W, D], T.bfloat16),
        CORRECTED: T.Tensor([B, N, HQ * W, D], T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, BI), HQ, B, threads=128) as (bx, h, b):
            grad = T.alloc_fragment([BI, D], T.float32)
            product = T.alloc_fragment([BI, D], T.float32)
            square = T.alloc_fragment([BI, D], T.float32)
            do = T.alloc_fragment([BI, D], T.float32)
            o = T.alloc_fragment([BI, D], T.float32)
            delta = T.alloc_fragment([BI], T.float32)
            dot = T.alloc_fragment([BI], T.float32)
            norm = T.alloc_fragment([BI], T.float32)
            lam = T.alloc_fragment([BI], T.float32)
            weight = T.alloc_shared([BI, W], T.float32)
            for i, t in T.Parallel(BI, W):
                row = bx * BI + i
                weight[i, t] = T.if_then_else(row < N, WEIGHT[b, row, h, t], 0)
            for i, d in T.Parallel(BI, D):
                row = bx * BI + i
                grad[i, d] = T.if_then_else(row < N, GRAD[b, row, h, d].astype(T.float32), 0)
                product[i, d] = T.if_then_else(row < N, grad[i, d] * OUT[b, row, h, d], 0)
            T.reduce_sum(product, delta, dim=1)
            for t in T.serial(W):
                for i, d in T.Parallel(BI, D):
                    row = bx * BI + i
                    valid = row < N and t <= row
                    do[i, d] = T.if_then_else(
                        valid,
                        (weight[i, t] * grad[i, d] * V1[b, row - t, h // (HQ // HK), d].astype(T.float32))
                        .astype(T.bfloat16).astype(T.float32),
                        0,
                    )
                    o[i, d] = T.if_then_else(valid, O[b, row, h * W + t, d].astype(T.float32), 0)
                    product[i, d] = do[i, d] * o[i, d]
                    square[i, d] = do[i, d] * do[i, d]
                T.reduce_sum(product, dot, dim=1)
                T.reduce_sum(square, norm, dim=1)
                for i in T.Parallel(BI):
                    lam[i] = T.if_then_else(norm[i] > 0, (weight[i, t] * delta[i] - dot[i]) / norm[i], 0)
                for i, d in T.Parallel(BI, D):
                    row = bx * BI + i
                    if row < N:
                        DO[b, row, h * W + t, d] = do[i, d]
                        CORRECTED[b, row, h * W + t, d] = o[i, d] + lam[i] * do[i, d]

    return main


@tilelang.jit(out_idx=[2], pass_configs=_FAST_MATH)
def _query_gradient(B, HQ, HK, W):
    N = T.dynamic("sequence_length")

    @T.prim_func
    def main(
        DA: T.Tensor([B, N, HQ * W, D], T.bfloat16),
        K1: T.Tensor([B, N, HK, D], T.bfloat16),
        DQ: T.Tensor([B, N, HQ, D], T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, BI), HQ, B, threads=128) as (bx, h, b):
            acc = T.alloc_fragment([BI, D], T.float32)
            T.clear(acc)
            for t in T.serial(W):
                for i, d in T.Parallel(BI, D):
                    row = bx * BI + i
                    if row < N and t <= row:
                        acc[i, d] += (DA[b, row, h * W + t, d].astype(T.float32)
                                      * K1[b, row - t, h // (HQ // HK), d].astype(T.float32))
            for i, d in T.Parallel(BI, D):
                if bx * BI + i < N:
                    DQ[b, bx * BI + i, h, d] = acc[i, d]

    return main


@tilelang.jit(out_idx=[5, 6], pass_configs=_FAST_MATH)
def _key_gradient(B, HQ, HK, W):
    """dK1/dV1 gather every (head, slot) row that read key j; no atomics."""
    N = T.dynamic("sequence_length")
    G = HQ // HK

    @T.prim_func
    def main(
        DA: T.Tensor([B, N, HQ * W, D], T.bfloat16),
        Q: T.Tensor([B, N, HQ, D], T.bfloat16),
        GRAD: T.Tensor([B, N, HQ, D], T.bfloat16),
        WEIGHT: T.Tensor([B, N, HQ, W], T.float32),
        O: T.Tensor([B, N, HQ * W, D], T.bfloat16),
        DK1: T.Tensor([B, N, HK, D], T.bfloat16),
        DV1: T.Tensor([B, N, HK, D], T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, BI), HK, B, threads=128) as (bx, hk, b):
            dk = T.alloc_fragment([BI, D], T.float32)
            dv = T.alloc_fragment([BI, D], T.float32)
            T.clear(dk)
            T.clear(dv)
            for g in T.serial(G):
                for t in T.serial(W):
                    for j, d in T.Parallel(BI, D):
                        row = bx * BI + j + t
                        h = hk * G + g
                        if row < N:
                            dk[j, d] += (DA[b, row, h * W + t, d].astype(T.float32)
                                         * Q[b, row, h, d].astype(T.float32))
                            dv[j, d] += (WEIGHT[b, row, h, t] * GRAD[b, row, h, d].astype(T.float32)
                                         * O[b, row, h * W + t, d].astype(T.float32))
            for j, d in T.Parallel(BI, D):
                if bx * BI + j < N:
                    DK1[b, bx * BI + j, hk, d] = dk[j, d]
                    DV1[b, bx * BI + j, hk, d] = dv[j, d]

    return main


def _virtual(q, k1, w):
    """Virtual queries as cuDNN [B, HQ*w, N, D] views of contiguous [B, N, HQ*w, D] rows."""
    b, n, hq, d = q.shape
    hk = k1.shape[2]
    shifted = F.pad(k1, (0, 0, 0, 0, w - 1, 0)).unfold(1, w, 1).flip(-1).transpose(-1, -2)
    a = q.view(b, n, hk, hq // hk, 1, d) * shifted[:, :, :, None]
    return a.reshape(b, n, hq * w, d).transpose(1, 2)


def _rows(x):
    """cuDNN [B, H, N, D] result -> contiguous [B, N, H, D] rows."""
    return x.transpose(1, 2).contiguous()


class _Band:
    """cuDNN causal attention, or a sliding window as query chunks over overlapping keys.

    Chunk c holds queries [cC, cC + C) and keys [cC - window + 1, cC + C); an additive
    mask keeps each query's window and drops keys before position 0.
    """

    def __init__(self, b, n, window, chunk, device, dtype):
        self.b, self.n, self.window = b, n, window
        self.causal = window >= n
        self.bias = None
        if self.causal:
            return
        self.chunk, self.chunks = chunk, -(-n // chunk)
        self.span = chunk + window - 1
        rows = torch.arange(chunk, device=device)[None, :, None]
        keys = torch.arange(self.span, device=device)[None, None]
        position = torch.arange(self.chunks, device=device)[:, None, None] * chunk - window + 1 + keys
        allowed = (keys >= rows) & (keys <= rows + window - 1) & (position >= 0)
        bias = torch.zeros(allowed.shape, device=device, dtype=dtype).masked_fill(~allowed, float("-inf"))
        self.bias = bias[:, None].repeat(b, 1, 1, 1)

    def queries(self, x):
        """[B, H, N, D] -> [B*chunks, H, C, D]."""
        if self.causal:
            return x
        h, d = x.shape[1], x.shape[3]
        x = F.pad(x, (0, 0, 0, self.chunks * self.chunk - self.n))
        return x.view(self.b, h, self.chunks, self.chunk, d).transpose(1, 2).reshape(-1, h, self.chunk, d)

    def keys(self, x):
        """[B, H, N, D] -> [B*chunks, H, span, D]."""
        if self.causal:
            return x
        h, d = x.shape[1], x.shape[3]
        x = F.pad(x, (0, 0, self.window - 1, self.chunks * self.chunk - self.n))
        x = x.unfold(2, self.span, self.chunk)
        return x.permute(0, 2, 1, 4, 3).reshape(-1, h, self.span, d)

    def fold_keys(self, x):
        if self.causal:
            return x
        h, d = x.shape[1], x.shape[3]
        x = x.view(self.b, self.chunks, h, self.span, d)
        out = x.new_zeros(self.b, h, self.chunks * self.chunk + self.window - 1, d, dtype=torch.float32)
        for c in range(self.chunks):
            out[:, :, c * self.chunk : c * self.chunk + self.span] += x[:, c]
        return out[:, :, self.window - 1 : self.window - 1 + self.n].to(x.dtype)

    def rows(self, x):
        """Inverse of ``queries``."""
        if self.causal:
            return x
        h, d = x.shape[1], x.shape[3]
        x = x.view(self.b, self.chunks, h, self.chunk, d).transpose(1, 2)
        return x.reshape(self.b, h, self.chunks * self.chunk, d)[:, :, : self.n]

    def forward(self, q, k, v, scale):
        out, lse, *meta = _forward(self.queries(q), self.keys(k), self.keys(v), self.bias, True, 0.0,
                                   self.causal, False, scale=scale)
        self.meta = meta[:6]
        return self.rows(out), self.rows(lse)

    def backward(self, do, q, k, v, out, lse, scale):
        cq, ck, mq, mk, seed, offset = self.meta
        dq, dk, dv = _backward(
            self.queries(do), self.queries(q), self.keys(k), self.keys(v), self.queries(out),
            self.queries(lse).contiguous(), seed, offset, self.bias, cq, ck, mq, mk, 0.0,
            self.causal, scale=scale,
        )
        return self.rows(dq), self.fold_keys(dk), self.fold_keys(dv)


class _VirtualSimplicial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k1, k2, v1, v2, w, window, chunk, scale):
        b, n, hq, _ = q.shape
        hk = k1.shape[2]
        band = _Band(b, n, window, chunk, q.device, q.dtype)
        o, lse = band.forward(_virtual(q, k1, w), k2.transpose(1, 2), v2.transpose(1, 2), scale)
        o, lse = _rows(o), lse.squeeze(-1).contiguous()
        out, weight = _combine(b, hq, hk, w)(o, lse, v1)
        ctx.save_for_backward(q, k1, k2, v1, v2, o, lse, weight, out)
        ctx.band, ctx.cfg = band, (w, scale)
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, grad):
        q, k1, k2, v1, v2, o, lse, weight, out = ctx.saved_tensors
        band, (w, scale) = ctx.band, ctx.cfg
        b, _, hq, _ = q.shape
        hk = k1.shape[2]
        grad = grad.to(q.dtype).contiguous()
        do, corrected = _prepare(b, hq, hk, w)(grad, out, weight, o, v1)
        da, dk2, dv2 = band.backward(
            do.transpose(1, 2), _virtual(q, k1, w), k2.transpose(1, 2), v2.transpose(1, 2),
            corrected.transpose(1, 2), lse.unsqueeze(-1), scale,
        )
        da = _rows(da)
        dq = _query_gradient(b, hq, hk, w)(da, k1)
        dk1, dv1 = _key_gradient(b, hq, hk, w)(da, q, grad, weight, o)
        return dq, dk1, dk2.transpose(1, 2), dv1, dv2.transpose(1, 2), None, None, None, None


def virtual_simplicial(q, k1, k2, v1, v2, *, short_window, long_window, scale, chunk=512):
    """Joint causal 2-simplicial softmax with ``reference_simplicial`` semantics.

    Requires CUDA BF16 head_dim=128 inputs and 1 <= short_window <= long_window.
    ``chunk`` sets the sliding-window query chunk; it changes scheduling, not semantics.
    """
    if not 1 <= short_window <= min(long_window, q.shape[1]) or chunk < 1:
        raise ValueError("require 1 <= short_window <= min(long_window, sequence length)")
    if not q.is_cuda or any(x.dtype != torch.bfloat16 for x in (q, k1, k2, v1, v2)) or q.shape[-1] != D:
        raise ValueError("virtual simplicial attention requires CUDA BF16 head_dim=128 inputs")
    q, k1, k2, v1, v2 = (x.contiguous() for x in (q, k1, k2, v1, v2))
    return _VirtualSimplicial.apply(q, k1, k2, v1, v2, short_window, long_window, chunk, scale)
