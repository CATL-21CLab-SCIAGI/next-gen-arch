"""The existing simplicial forward with private-query gradient accumulation.

Each program owns one query/KV-group's gradient windows. Query contributions
are then summed in stable temporal order, avoiding cross-query FP32 atomics.
The original frozen speedrun kernel and API remain unchanged.
"""

import torch
import triton
import triton.language as tl

from archlab.architectures.ordered_reduction import ordered_index_sum
from archlab.architectures.simplicial_attention import validate_inputs
from archlab.architectures.simplicial_kernels import SimplicialFunction


@triton.jit
def _private_backward(Q, K1, K2, V1, V2, OUT, LSE, DO, DQ, DK1, DK2, DV1, DV2, START,
                      N: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                      G: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr,
                      H: tl.constexpr, T: tl.constexpr, CHUNK: tl.constexpr):
    local_i, bg = tl.program_id(0), tl.program_id(1)
    i = START + local_i
    b, g = bg // HK, bg % HK
    h, d, t = tl.arange(0, H), tl.arange(0, D), tl.arange(0, T)
    qo = ((b * N + i) * HQ + g * G + h[:, None]) * D + d[None, :]
    q = tl.load(Q + qo, h[:, None] < G, other=0)
    do = tl.load(DO + qo, h[:, None] < G, other=0)
    out = tl.load(OUT + qo, h[:, None] < G, other=0).to(tl.float32)
    delta = tl.sum(do.to(tl.float32) * out, 1)
    lo = (b * N + i) * HQ + g * G + h
    lse = tl.load(LSE + lo, h < G, other=0)
    dq = tl.full((H, D), 0, tl.float32)
    for j in range(tl.maximum(0, i - W1 + 1), i + 1):
        ko = ((b * N + j) * HK + g) * D + d
        k1 = tl.load(K1 + ko)
        v1 = tl.load(V1 + ko)
        a = (q.to(tl.float32) * k1[None, :].to(tl.float32)).to(q.dtype)
        dov1 = (do.to(tl.float32) * v1[None, :].to(tl.float32)).to(do.dtype)
        dk1 = tl.full((H, D), 0, tl.float32)
        dv1 = tl.full((H, D), 0, tl.float32)
        for start in range(tl.maximum(0, i - W2 + 1), i + 1, T):
            k = start + t
            off = ((b * N + k[:, None]) * HK + g) * D + d[None, :]
            k2 = tl.load(K2 + off, k[:, None] <= i, other=0)
            v2 = tl.load(V2 + off, k[:, None] <= i, other=0)
            s = tl.dot(a, tl.trans(k2), input_precision="tf32x3") * (D ** -0.5)
            p = tl.exp(s - lse[:, None])
            p = tl.where((k[None, :] <= i) & (h[:, None] < G), p, 0)
            dp = tl.dot(dov1, tl.trans(v2), input_precision="tf32x3")
            ds = (p * (dp - delta[:, None]) * (D ** -0.5)).to(q.dtype)
            ds = tl.where((i == 0) | ((W1 == 1) & (W2 == 1)), 0, ds)
            da = tl.dot(ds, k2, input_precision="tf32x3")
            dq += da * k1[None, :].to(tl.float32)
            dk1 += da * q.to(tl.float32)
            dk2 = tl.dot(tl.trans(ds), a, input_precision="tf32x3")
            pv2 = tl.dot(p.to(v2.dtype), v2, input_precision="tf32x3")
            dv1 += pv2 * do.to(tl.float32)
            dv2 = tl.dot(tl.trans(p.to(do.dtype)), dov1, input_precision="tf32x3")
            private = (((b * CHUNK + local_i) * HK + g) * W2 + (i - k[:, None])) * D + d[None, :]
            # The same program exclusively owns this query's windows. The
            # per-element additions follow its serial short-window loop.
            tl.atomic_add(DK2 + private, dk2, k[:, None] <= i, sem="relaxed")
            tl.atomic_add(DV2 + private, dv2, k[:, None] <= i, sem="relaxed")
        private = (((b * CHUNK + local_i) * HK + g) * W1 + (i - j)) * D + d
        tl.store(DK1 + private, tl.sum(dk1, 0))
        tl.store(DV1 + private, tl.sum(dv1, 0))
    tl.store(DQ + qo, dq, h[:, None] < G)


def _reduce_window(destination, partial, first, count, window):
    query = torch.arange(first, first + count, device=destination.device)
    keys = query[:, None] - torch.arange(window, device=destination.device)[None, :]
    valid = keys.flatten() >= 0
    indices = keys.flatten()[valid].long()
    for b in range(destination.shape[0]):
        values = partial[b, :count].permute(0, 2, 1, 3).reshape(count * window, -1)
        ordered_index_sum(destination[b].flatten(1), indices, values[valid])


class DeterministicSimplicialFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k1, k2, v1, v2, short_window, long_window):
        return SimplicialFunction.forward(ctx, q, k1, k2, v1, v2, short_window, long_window)

    @staticmethod
    def backward(ctx, grad_output):
        q, k1, k2, v1, v2, output, lse = ctx.saved_tensors
        batch, sequence = q.shape[:2]
        groups, dim = k1.shape[2:]
        chunk = 128
        dq = torch.empty_like(q)
        gradients = [torch.zeros_like(value, dtype=torch.float32) for value in (k1, k2, v1, v2)]
        windows = [ctx.config["W1"], ctx.config["W2"]] * 2
        do = grad_output.contiguous()
        for first in range(0, sequence, chunk):
            count = min(chunk, sequence - first)
            private = [torch.zeros((batch, chunk, groups, window, dim), device=q.device, dtype=torch.float32)
                       for window in windows]
            _private_backward[(count, batch * groups)](q, k1, k2, v1, v2, output, lse, do, dq,
                                                       *private, first, CHUNK=chunk, **ctx.config)
            for target, partial, window in zip(gradients, private, windows, strict=True):
                _reduce_window(target, partial, first, count, window)
        return dq, *(value.to(q.dtype) for value in gradients), None, None


def deterministic_simplicial_attention(q, k1, k2, v1, v2, short_window, long_window):
    validate_inputs(q, k1, k2, v1, v2, short_window, long_window)
    if q.device.type != "cuda" or q.dtype != torch.float32 or q.shape[-1] not in (16, 32, 64, 128, 256):
        raise ValueError("deterministic simplicial core requires supported CUDA FP32 head geometry")
    return DeterministicSimplicialFunction.apply(q, k1, k2, v1, v2, short_window, long_window)
