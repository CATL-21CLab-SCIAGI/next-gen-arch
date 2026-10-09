"""Single-process qualification bridge for the *native* frozen V4.1 model.

This is not a distributed trainer. Forward quantization/GEMMs remain the pinned
official kernels. Input gradients use the explicit straight-through estimator
(STE): dInput = dOutput @ dequantized frozen weight. This is a training surrogate,
not the mathematical derivative of rounding. No base-parameter gradients exist.

Only the imported reference module is adapted; no PyTorch/NeMo/TileLang package
is patched. Distributed construction is rejected until independently qualified.
"""

from __future__ import annotations

import types

import torch
from torch import nn
from torch.nn import functional as F

from archlab.architectures.deepseek_v41_math import dequantize_frozen_weight, hc_split_sinkhorn


class FrozenQuantizedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale, native_linear):
        if weight.requires_grad or scale.requires_grad:
            raise ValueError("the quantized base and its scales must be frozen")
        ctx.save_for_backward(weight, scale)
        ctx.input_dtype = x.dtype
        return native_linear(x, weight)

    @staticmethod
    def backward(ctx, dy):
        weight, scale = ctx.saved_tensors
        # Weight is frozen: retaining the large input activation is unnecessary.
        decoded = dequantize_frozen_weight(weight, scale)
        dx = torch.matmul(dy.to(decoded.dtype), decoded).to(ctx.input_dtype)
        return dx, None, None, None


class _FrozenNativeMHC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixes, scale, base, streams, iterations, eps, native):
        if scale.requires_grad or base.requires_grad:
            raise ValueError("base mHC coefficients must be frozen")
        ctx.save_for_backward(mixes, scale, base)
        ctx.geometry = streams, iterations, eps
        return native(mixes, scale, base, streams, iterations, eps)

    @staticmethod
    def backward(ctx, dpre, dpost, dcomb):
        mixes, scale, base = ctx.saved_tensors
        with torch.enable_grad():
            x = mixes.detach().requires_grad_()
            outputs = hc_split_sinkhorn(x, scale, base, *ctx.geometry)
            dx, = torch.autograd.grad(outputs, x, (dpre, dpost, dcomb))
        return dx, None, None, None, None, None, None


class _FrozenNativeSparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, sink, indices, softmax_scale, native):
        output = native(q, kv, sink, indices, softmax_scale)
        ctx.save_for_backward(q, kv, sink, indices, output)
        ctx.softmax_scale = softmax_scale
        return output.clone()  # native attention applies inverse RoPE in place

    @staticmethod
    def backward(ctx, dy):
        from nemo_automodel.components.models.deepseek_v4.kernels import (
            tilelang_sparse_mla_bwd,
            tilelang_sparse_mla_fwd,
        )

        q, kv, sink, indices, output = ctx.saved_tensors
        indices = F.pad(indices, (0, -indices.shape[-1] % 64), value=-1).contiguous()
        # Reuse NeMo's LSE computation/backward, but never substitute its
        # differently rounded forward result for the native backbone output.
        temporary, lse = tilelang_sparse_mla_fwd.sparse_mqa_fwd_interface(
            q.contiguous(), kv.contiguous(), sink, indices, sm_scale=ctx.softmax_scale)
        del temporary
        dq, dkv, _ = tilelang_sparse_mla_bwd.sparse_mqa_bwd_interface(
            q.contiguous(), kv.contiguous(), sink, output.contiguous(), dy.contiguous(),
            indices, lse, sm_scale=ctx.softmax_scale)
        return dq, dkv, None, None, None, None


class _RoundedIdentityGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original, rounded):
        # Do not use x + (rounded-x).detach(): that adds BF16 rounding beyond
        # the native quantizer and can change subsequent expert selections.
        return rounded.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def _ste_inplace_quantizer(native, x, *args, inplace=False, **kwargs):
    if not inplace:
        return native(x, *args, inplace=False, **kwargs)
    with torch.no_grad():
        rounded = x.detach().clone()
        native(rounded, *args, inplace=True, **kwargs)
    # The native caller intentionally ignores the return value. Preserve its
    # in-place API, but give the replaced value an identity input derivative.
    x.copy_(_RoundedIdentityGradient.apply(x, rounded))
    return x


def install_frozen_backward(reference):
    if reference.dist.is_initialized() and reference.dist.get_world_size() != 1:
        raise RuntimeError("this bridge has not qualified distributed input-gradient collectives")
    if getattr(reference, "_archlab_backward_installed", False):
        raise RuntimeError("backward bridge already installed")
    native_linear = reference.linear
    native_fp8, native_fp4 = reference.act_quant, reference.fp4_act_quant
    native_sparse, native_mhc = reference.sparse_attn, reference.hc_split_sinkhorn

    def linear(x, weight, bias=None):
        if bias is not None or weight.requires_grad:
            raise ValueError("only frozen bias-free base projections are supported")
        if weight.dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2) and x.requires_grad:
            return FrozenQuantizedLinear.apply(x, weight, weight.scale, native_linear)
        return native_linear(x, weight, bias)

    def fp8(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        return _ste_inplace_quantizer(native_fp8, x, block_size, scale_fmt, scale_dtype, inplace=inplace)

    def fp4(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        return _ste_inplace_quantizer(native_fp4, x, block_size, inplace=inplace, scale_dtype=scale_dtype)

    def sparse(q, kv, sink, indices, softmax_scale):
        if not torch.is_grad_enabled() or not (q.requires_grad or kv.requires_grad):
            return native_sparse(q, kv, sink, indices, softmax_scale)
        return _FrozenNativeSparse.apply(q, kv, sink, indices, softmax_scale, native_sparse)

    def mhc(mixes, scale, base, streams=4, iterations=20, eps=1e-6):
        if not torch.is_grad_enabled() or not mixes.requires_grad:
            return native_mhc(mixes, scale, base, streams, iterations, eps)
        return _FrozenNativeMHC.apply(mixes, scale, base, streams, iterations, eps, native_mhc)

    def gate(self, x, image_mask=None):
        scores = native_linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        indices = (scores + bias).topk(self.topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, indices

    reference.linear = linear
    reference.act_quant, reference.fp4_act_quant = fp8, fp4
    reference.hc_split_sinkhorn = mhc
    reference.sparse_attn = sparse
    reference.Gate.forward = gate
    # The index selection is discrete and has no task derivative. Its latent
    # input is NOT detached on the shared-KV value path.
    reference.Indexer.forward = torch.no_grad()(reference.Indexer.forward)
    reference._archlab_backward_installed = True


def _block_with_adapter(self, x, start_pos, pre_mix, image_mask, *attn_args):
    residual = x
    attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
    attended = self.attn(self.attn_norm(self.hc_pre(x, pre_mix)), start_pos, *attn_args)
    x = self.hc_post(attended, residual, attn_post, attn_comb)
    x = self.simplicial_adapter(x)
    residual = x
    ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
    ffn = self.ffn(self.ffn_norm(self.hc_pre(x, attn_pre)), image_mask)
    return self.hc_post(ffn, residual, ffn_post, ffn_comb), ffn_pre


def attach_adapters(model, adapters: dict[int, nn.Module]):
    """Keys are zero-based block numbers; the original pre-mix flow is retained."""
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("freeze the complete base before attaching adapters")
    if any(type(i) is not int or not 0 <= i < len(model.layers) for i in adapters):
        raise ValueError("adapter block outside backbone")
    if any(hasattr(model.layers[i], "simplicial_adapter") for i in adapters):
        raise ValueError("adapter already attached")
    for i, adapter in adapters.items():
        layer = model.layers[i]
        layer.simplicial_adapter = adapter
        layer.forward = types.MethodType(_block_with_adapter, layer)


def training_hidden(reference, model, input_ids):
    """Text-only, independent full-prefill segments; deliberately no decode API.

    Never feed detached cross-segment caches into a later training segment.
    Full-context/chunk selection is the caller's explicit experiment contract.
    MTP/vision stay inactive, rather than contributing extra objectives.
    """
    if reference.world_size != 1:
        raise RuntimeError("single-process qualification only")
    if not getattr(reference, "_archlab_backward_installed", False):
        raise RuntimeError("install and qualify the backward bridge first")
    if input_ids.ndim != 2 or input_ids.shape[1] > model.max_seq_len:
        raise ValueError("input must be a bounded two-dimensional token segment")
    reference.shared_attn = reference.SharedAttentionRuntime()
    # Rebind, don't mutate tensors still needed by an earlier backward. RoPE
    # and frozen persistent buffers remain untouched.
    for module in model.modules():
        for name, buffer in list(module.named_buffers(recurse=False)):
            if ("cache" in name or name in ("kv_state", "score_state")) and name in module._non_persistent_buffers_set:
                setattr(module, name, torch.zeros_like(buffer, requires_grad=False))
    hashes = model.engram_hash(input_ids, 0, None) if model.engram_hash is not None else None
    h = model.embed(input_ids).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre = reference.make_identity_pre_mix(h, model.hc_mult)
    for layer in model.layers:
        if layer.engram is not None:
            h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :], None)
        h, pre = layer(h, 0, pre, None)
    return model.norm(model.layers[-1].hc_pre(h, pre))
