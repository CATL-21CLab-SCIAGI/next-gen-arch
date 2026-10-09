"""Bound native expert pointwise temporaries while preserving BF16 GEMM outputs.

The separate full-size gate/up GEMMs remain with the caller. For CUDA BF16
inputs, FP32 activation arithmetic and its first-order backward are tiled by
token row. Other device/dtype combinations use the original PyTorch reference.
"""

import math

import torch
from torch.nn import functional as F


class _NativeSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, value, probabilities, limit, chunk_rows):
        result = torch.empty_like(gate)
        for first in range(0, gate.shape[0], chunk_rows):
            last = first + chunk_rows
            g = gate[first:last].float().clamp(max=limit)
            v = value[first:last].float().clamp(min=-limit, max=limit)
            result[first:last].copy_(F.silu(g) * v * probabilities[first:last])
        ctx.save_for_backward(gate, value, probabilities)
        ctx.limit, ctx.chunk_rows = limit, chunk_rows
        return result

    @staticmethod
    def backward(ctx, gradient):
        gate, value, probabilities = ctx.saved_tensors
        dg = torch.empty_like(gate) if ctx.needs_input_grad[0] else None
        dv = torch.empty_like(value) if ctx.needs_input_grad[1] else None
        dp = torch.empty_like(probabilities) if ctx.needs_input_grad[2] else None
        for first in range(0, gate.shape[0], ctx.chunk_rows):
            last = first + ctx.chunk_rows
            raw_g, raw_v = gate[first:last].float(), value[first:last].float()
            g = raw_g.clamp(max=ctx.limit)
            v = raw_v.clamp(min=-ctx.limit, max=ctx.limit)
            dy = gradient[first:last].float()
            activation = F.silu(g)
            da = dy * probabilities[first:last]
            if dg is not None:
                derivative = torch.ops.aten.silu_backward.default(da * v, g)
                dg[first:last].copy_(torch.where(raw_g <= ctx.limit, derivative, 0))
            if dv is not None:
                derivative = da * activation
                dv[first:last].copy_(
                    torch.where((raw_v >= -ctx.limit) & (raw_v <= ctx.limit), derivative, 0)
                )
            if dp is not None:
                terms = dy * (activation * v)
                rows = terms.shape[0]
                # Keep a large row-reduction launch for a short final tile when
                # the original operation covered many rows.
                if gate.shape[0] > ctx.chunk_rows and rows < ctx.chunk_rows:
                    padded = terms.new_zeros((ctx.chunk_rows, terms.shape[1]))
                    padded[:rows].copy_(terms)
                    reduced = padded.sum(-1, keepdim=True)[:rows]
                else:
                    reduced = terms.sum(-1, keepdim=True)
                dp[first:last].copy_(reduced)
        return dg, dv, dp, None, None


def native_swiglu(gate, value, probabilities, *, limit, chunk_rows=1024):
    if (
        gate.ndim != 2
        or value.shape != gate.shape
        or value.dtype != gate.dtype
        or probabilities.shape != (gate.shape[0], 1)
        or probabilities.dtype != torch.float32
        or not gate.is_floating_point()
        or not gate.device == value.device == probabilities.device
        or not math.isfinite(limit)
        or limit <= 0
        or type(chunk_rows) is not int
        or chunk_rows < 1
    ):
        raise ValueError(
            "expected matching gate/value rows, FP32 row probabilities and positive limits"
        )
    if gate.device.type != "cuda" or gate.dtype != torch.bfloat16:
        return (
            F.silu(gate.float().clamp(max=limit))
            * value.float().clamp(min=-limit, max=limit)
            * probabilities
        ).to(gate.dtype)
    return _NativeSwiGLU.apply(gate, value, probabilities, limit, chunk_rows)
