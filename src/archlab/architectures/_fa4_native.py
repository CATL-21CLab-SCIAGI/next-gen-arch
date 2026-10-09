"""Lazy CuTe binding for the official FA4 backward; no upstream code copies.

Official FlashAttention BSD-3-Clause source is pinned by the public leaf's
runtime contract. Its function bytecode remains unchanged in a local namespace;
the callback and output boundary below are project-owned numerical adaptations.
"""

from contextvars import ContextVar
from functools import cache

import cutlass.cute as cute
import torch
from cutlass import Float32
from flash_attn.cute.interface import (
    _bwd_postprocess_convert,
    _bwd_preprocess,
    _flash_attn_bwd,
)

from archlab.architectures.limite_bindings import bind_native_forward
from archlab.architectures.tilelang_gqa import _delta_kernel

_boundaries = ContextVar("archlab_fa4_native_boundaries")


def _exact_preprocess(out, dout, dpsum, lse, lse_log2, *args, **kwargs):
    result = _bwd_preprocess(out, dout, dpsum, lse, lse_log2, *args, **kwargs)
    delta, original_lse_log2 = _boundaries.get()
    dpsum[..., : delta.shape[-1]].copy_(delta)
    lse_log2[..., : original_lse_log2.shape[-1]].copy_(original_lse_log2)
    return result


def _unscaled_postprocess(accum, output, scale, *args, **kwargs):
    # Score scale is already included before BF16 dS; dV has no score scale.
    return _bwd_postprocess_convert(accum, output, 1.0, *args, **kwargs)


@cute.jit
def _native_score_gradient(
    grad, score, batch_idx, head_idx, *, q_idx, kv_idx, seqlen_info, aux_tensors
):
    return cute.where(q_idx == 0, Float32(0.0), grad * Float32(0.1))


@cache
def _upstream_backward():
    return bind_native_forward(
        _flash_attn_bwd,
        (("_bwd_preprocess", _exact_preprocess), ("_bwd_postprocess_convert", _unscaled_postprocess)),
        "archlab_fa4_exact_delta_lse_scaled_ds_q0_v3",
    )


class NativeBF16Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window, scale, forward):
        q, k, v = [value.contiguous() for value in (q, k, v)]
        out, lse_log2 = forward(q, k, v, scaling=scale, long_window=window)
        ctx.save_for_backward(q, k, v, out, lse_log2)
        ctx.settings = (window, scale)
        return out.to(v.dtype)

    @staticmethod
    def backward(ctx, gradient_output):
        q, k, v, out, lse_log2 = ctx.saved_tensors
        window, scale = ctx.settings
        do = gradient_output.float().contiguous()
        b, n, h, _ = q.shape
        delta = _delta_kernel(b, n, h)(out, do)
        original_lse = lse_log2.transpose(1, 2).contiguous()
        token = _boundaries.set((delta, original_lse))
        try:
            dq, dk, dv = _upstream_backward()(
                q,
                k,
                v,
                out.to(q.dtype),
                do.to(q.dtype),
                original_lse * 0.6931471805599453,
                softmax_scale=scale,
                causal=True,
                window_size_left=window - 1 if window < n else None,
                window_size_right=0 if window < n else None,
                score_mod_bwd=_native_score_gradient,
            )
        finally:
            _boundaries.reset(token)
        dq[:, 0].zero_()
        return dq, dk, dv, None, None, None
