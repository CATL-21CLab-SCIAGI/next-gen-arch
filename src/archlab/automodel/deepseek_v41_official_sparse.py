# ruff: noqa: I001  # Preserve the checkpoint-qualified import order.
"""Deterministic shared-KV reduction around the official sparse MLA kernels.

The official forward is unchanged. Backward reuses its exact per-query kernel
with private compact KV slots, then reduces shared-key contributions in a fixed
order. No queries race to update the same FP32 accumulator before BF16 casting.
"""

from types import FunctionType, MethodType

import torch
from torch.nn import functional as F
from archlab.architectures.ordered_reduction import ordered_index_sum as _ordered_index_sum


def _pad_queries(value, size, fill=0):
    if value.shape[0] == size:
        return value.contiguous()
    return torch.cat(
        (value, value.new_full((size - value.shape[0], *value.shape[1:]), fill))
    ).contiguous()


class _DeterministicSparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, sinks, indices, scale, head_chunk, query_chunk):
        from nemo_automodel.components.models.deepseek_v4.kernels.sparse_attention import (
            DeepSeekV4SparseAttentionHeadChunked,
        )

        ctx.query_chunk = query_chunk
        return DeepSeekV4SparseAttentionHeadChunked.forward(
            ctx, q, kv, sinks, indices, head_chunk, scale, True
        )

    @staticmethod
    def backward(ctx, grad_output):
        from nemo_automodel.components.models.deepseek_v4.kernels import (
            tilelang_sparse_mla_bwd as kernels,
        )

        q, kv, sinks, indices, output, lse = ctx.saved_tensors
        batch, sequence, heads, dim = q.shape
        slots = (indices.shape[-1] + 31) // 32 * 32
        indices = F.pad(indices, (0, slots - indices.shape[-1]), value=-1).contiguous()
        do = grad_output.contiguous()
        dq = torch.empty_like(q)
        dkv = torch.zeros_like(kv, dtype=torch.float32)
        chunk, max_heads = ctx.query_chunk, ctx.max_heads_per_kernel
        deltas = []
        for first_head in range(0, heads, max_heads):
            last_head = min(first_head + max_heads, heads)
            deltas.append(
                kernels.preprocess(batch, sequence, last_head - first_head, dim)(
                    output[:, :, first_head:last_head].contiguous(),
                    do[:, :, first_head:last_head].contiguous(),
                )
            )
        private_indices = torch.arange(slots, device=q.device, dtype=torch.int32).view(1, 1, slots)
        private_indices = private_indices.expand(chunk, 1, slots).contiguous()
        for b in range(batch):
            for first in range(0, sequence, chunk):
                last = min(first + chunk, sequence)
                count = last - first
                original = _pad_queries(indices[b, first:last], chunk, -1)
                valid = (original >= 0) & (original < kv.shape[1])
                private_kv = kv[b, original.clamp(0, kv.shape[1] - 1)].contiguous()
                valid_mask = valid.to(torch.int32).unsqueeze(1).contiguous()
                combined = torch.zeros((chunk, slots, dim), device=q.device, dtype=torch.float32)
                for h, first_head in enumerate(range(0, heads, max_heads)):
                    last_head = min(first_head + max_heads, heads)
                    kernel = kernels.bwd(
                        chunk, 1, slots, last_head - first_head, dim, slots, ctx.sm_scale
                    )
                    query = _pad_queries(q[b, first:last, first_head:last_head], chunk).unsqueeze(1)
                    dout = _pad_queries(do[b, first:last, first_head:last_head], chunk).unsqueeze(1)
                    normalizer = _pad_queries(
                        lse[b, first:last, first_head:last_head], chunk, float("inf")
                    ).unsqueeze(1)
                    delta = _pad_queries(deltas[h][b, first:last], chunk).unsqueeze(1)
                    partial = torch.zeros_like(combined)
                    unused_sink = torch.zeros(
                        last_head - first_head, device=q.device, dtype=torch.float32
                    )
                    grad_q = kernel(
                        query,
                        private_kv,
                        dout,
                        sinks[first_head:last_head].contiguous(),
                        private_indices,
                        valid_mask,
                        normalizer,
                        delta,
                        partial,
                        unused_sink,
                    )
                    dq[b, first:last, first_head:last_head].copy_(grad_q[:count, 0])
                    combined.add_(partial)
                if ctx.needs_input_grad[1]:
                    flat_valid = valid.reshape(-1)
                    _ordered_index_sum(
                        dkv[b],
                        original.reshape(-1)[flat_valid].long(),
                        combined.reshape(-1, dim)[flat_valid],
                    )
        dsink = None
        if ctx.needs_input_grad[2]:
            # The kernel carries base-2 LSE. Sink logits are already scaled.
            delta = (output.float() * do.float()).sum(-1)
            probability = torch.exp2(sinks.view(1, 1, -1) * 1.4426950408889634 - lse)
            dsink = -(delta * probability).sum((0, 1))
        return (
            dq if ctx.needs_input_grad[0] else None,
            dkv.to(kv.dtype) if ctx.needs_input_grad[1] else None,
            dsink,
            None,
            None,
            None,
            None,
        )


def deterministic_sparse_attention(
    q, kv, sinks, indices, scale, *, backend, reference_rounding=False
):
    from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sparse_attention

    if backend != "tilelang" or not reference_rounding:
        raise ValueError(
            "this precision boundary is specific to native-rounding V4.1 TileLang attention"
        )
    if not torch.is_grad_enabled() or not (
        q.requires_grad or kv.requires_grad or sinks.requires_grad
    ):
        return dsv4_sparse_attention(
            q, kv, sinks, indices, scale, backend=backend, reference_rounding=reference_rounding
        )
    original_heads = q.shape[2]
    if original_heads < 16:
        q = torch.cat((q, q.new_zeros(*q.shape[:2], 16 - original_heads, q.shape[3])), dim=2)
        sinks = torch.cat((sinks, sinks.new_zeros(16 - original_heads)))
    output = _DeterministicSparse.apply(
        q.contiguous(),
        kv.contiguous(),
        sinks.float().contiguous(),
        indices.to(torch.int32).contiguous(),
        scale,
        16 if q.shape[-1] >= 256 else 64,
        128,
    )
    return output[:, :, :original_heads]


def install_official_deterministic_sparse(model):
    """Bind a private sparse dispatcher to each unchanged official forward body."""
    from nemo_automodel.components.models.deepseek_v41.attention import DeepseekV41Attention

    selected = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, DeepseekV41Attention)
    ]
    if not selected or getattr(model, "_archlab_deterministic_sparse_installed", False):
        raise ValueError("install deterministic sparse attention once on an official V4.1 backbone")
    for name, module in selected:
        fn = module.forward.__func__
        if (
            module.backend.attn != "tilelang"
            or "dsv4_sparse_attention" not in fn.__code__.co_names
            or "forward" in module.__dict__
        ):
            raise ValueError(f"{name}: expected the original TileLang V4.1 attention forward")
        if any(p.requires_grad for p in module.parameters()):
            raise ValueError("freeze the official base before sparse precision adaptation")
    parameters = dict(model.named_parameters())
    for _, module in selected:
        fn = module.forward.__func__
        namespace = dict(fn.__globals__)
        namespace["dsv4_sparse_attention"] = deterministic_sparse_attention
        bound = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
        bound.__kwdefaults__, bound.__annotations__ = fn.__kwdefaults__, fn.__annotations__
        module.forward = MethodType(bound, module)
    after = dict(model.named_parameters())
    if after.keys() != parameters.keys() or any(
        after[name] is not p for name, p in parameters.items()
    ):
        raise RuntimeError("sparse precision adaptation changed a parameter")
    model._archlab_deterministic_sparse_installed = True
    return {
        "implementation": "official-sparse-kernels-with-private-query-KV-reduction-v1",
        "modules": [name for name, _ in selected],
        "forward": "unchanged-official-native-rounding",
        "query_chunk": 128,
        "kv_gradient_dtype": "float32",
        "reduction": "stable-key-sort-segment-sum",
        "upstream_globals_unchanged": True,
        "original_parameters_preserved": True,
    }
