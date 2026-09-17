"""PyTorch-first frozen-base execution: one-time BF16 decoding, no hot-path TL.

The imported native architecture still owns routing, masks, shared KV, RoPE,
Engram and single-pass mHC. Only numerical leaf implementations are replaced.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from archlab.architectures.deepseek_v41_math import dequantize_frozen_weight, hc_split_sinkhorn
from archlab.architectures.deepseek_v41_torch import (
    query_chunked_sparse_attention,
    rounded_activation,
)
from archlab.automodel.deepseek_v41_autograd import install_frozen_backward


def bf16_memory_plan(model, *, reserve_gib=48):
    """Conservative peak: final resident + largest old tensor + working reserve."""
    current = final = largest_old = 0
    for name, p in model.named_parameters():
        before = p.numel() * p.element_size()
        current += before
        if name.endswith(".scale"):
            after = 0
        elif p.dtype == torch.float4_e2m1fn_x2:
            after = p.numel() * 4
        elif p.dtype == torch.float8_e4m3fn:
            after = p.numel() * 2
        else:
            after = before
        final += after
        if after > before:
            largest_old = max(largest_old, before)
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    free, total = torch.cuda.mem_get_info()
    # Driver/NCCL/non-PyTorch allocations are not available to the model.
    # PyTorch's unused reserved blocks are reclaimable, not driver/NCCL use.
    outside_model = total - free - torch.cuda.memory_reserved()
    required = final + buffers + largest_old + int(reserve_gib * 2**30) + max(0, outside_model)
    return {"quantized_parameter_gib": current / 2**30, "bf16_parameter_gib": final / 2**30,
            "buffers_gib": buffers / 2**30, "largest_old_tensor_gib": largest_old / 2**30,
            "reserved_workspace_gib": reserve_gib, "required_peak_gib": required / 2**30,
            "device_total_gib": total / 2**30, "device_free_gib": free / 2**30,
            "fits": required < total}


@torch.no_grad()
def dequantize_base_once(model, *, reserve_gib=48, chunk_rows=1024):
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("decode before attaching trainable adapters")
    plan = bf16_memory_plan(model, reserve_gib=reserve_gib)
    if not plan["fits"]:
        raise MemoryError(f"BF16 frozen base violates per-GPU reserve: {plan}")
    if chunk_rows < 32 or chunk_rows % 32:
        raise ValueError("decode chunks must align to FP8 block rows")
    converted = []
    for name, module in model.named_modules():
        old = getattr(module, "weight", None)
        if not isinstance(old, nn.Parameter) or old.dtype not in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2):
            continue
        scale = module.scale
        is_table = name.endswith(".engram.embed")
        columns = old.shape[1] * (2 if old.dtype == torch.float4_e2m1fn_x2 else 1)
        target = torch.empty(old.shape[0], columns, device=old.device, dtype=torch.bfloat16)
        for start in range(0, old.shape[0], chunk_rows):
            end = min(start + chunk_rows, old.shape[0])
            if is_table:
                block = old[start:end].float() * scale[start:end].float().repeat_interleave(32, -1)
                target[start:end].copy_(block)
            else:
                factors = scale[start:end] if old.dtype == torch.float4_e2m1fn_x2 else scale[start // 32:(end + 31) // 32]
                target[start:end].copy_(dequantize_frozen_weight(old[start:end], factors))
        module.weight = nn.Parameter(target, requires_grad=False)
        module.weight._archlab_quantized_source = not is_table
        module.register_parameter("scale", None)
        converted.append({"module": name, "from": str(old.dtype), "logical_shape": list(target.shape)})
        del old, scale, target
    torch.cuda.empty_cache()
    plan["converted_modules"] = len(converted)
    plan["resident_after_gib"] = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    plan["conversion_shapes"] = converted
    return plan


def install_pytorch_leaves(reference, *, query_chunk=32, activation_mode="native"):
    if activation_mode not in ("native", "bf16"):
        raise ValueError("choose native-style rounding or the measured BF16 activation ablation")
    if not getattr(reference, "_archlab_backward_installed", False):
        # Reuse its functional router and index-selection no_grad boundary.
        # Replace every kernel-bearing leaf below before any model execution.
        install_frozen_backward(reference)
    reference._archlab_activation_mode = activation_mode

    def linear(x, weight, bias=None):
        if weight.dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn) or weight.requires_grad:
            raise ValueError("PyTorch base linears require frozen, once-dequantized weights")
        if getattr(weight, "_archlab_quantized_source", False) and reference._archlab_activation_mode == "native":
            x = rounded_activation(x)
        return F.linear(x, weight, bias)

    def fp8(x, block_size=32, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        if not inplace:
            raise ValueError("no packed activation GEMMs are allowed in the PyTorch path")
        if reference._archlab_activation_mode == "native":
            x.copy_(rounded_activation(x, block_size=block_size))
        return x

    def fp4(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        if not inplace:
            raise ValueError("only the native dequantized-inplace FP4 interface is supported")
        if reference._archlab_activation_mode == "native":
            x.copy_(rounded_activation(x, bits=4, block_size=block_size,
                                       e4m3_scale=scale_dtype == torch.float8_e4m3fn))
        return x

    reference.linear = linear
    reference.act_quant, reference.fp4_act_quant = fp8, fp4
    reference.hc_split_sinkhorn = hc_split_sinkhorn
    reference.sparse_attn = partial(query_chunked_sparse_attention, query_chunk=query_chunk,
                                    native_rounding=True)
    original_embed = reference.ParallelEngramEmbedding.forward

    def embedding(self, ids):
        if hasattr(self, "vocab_start_idx") and self.weight.dtype == torch.bfloat16:
            if reference.world_size != 1:
                raise ValueError("use the explicitly qualified EP embedding for distributed execution")
            return F.embedding(ids, self.weight)
        return original_embed(self, ids)

    reference.ParallelEngramEmbedding.forward = embedding
    install_chunked_indexer(reference, query_chunk=max(32, query_chunk))


def install_chunked_indexer(reference, *, query_chunk):
    """Native prefill indexer equations/topk, bounded in the query dimension.

    The selection is discrete; shared compressed KV is NOT detached. The
    published candidate mask and position-sorted selections are unchanged.
    """
    @torch.no_grad()
    def forward(self, x, qr, latent, start_pos, offset):
        if start_pos != 0:
            raise ValueError("training indexer supports independent full prefills only")
        bsz, length, _ = x.shape
        ratio, rd = self.compress_ratio, self.rope_head_dim
        shared = reference.shared_attn
        if self.owns_k and latent is not None:
            k = self.k_norm(self.wk(latent))
            reference.apply_rotary_emb(k[..., -rd:], self.freqs_cis[:length - length % ratio:ratio])
            reference.fp4_act_quant(k, reference.fp4_block_size, True)
            self.k_cache[:bsz, :k.shape[1]] = k
            shared.index_k = self.k_cache
        keys = shared.index_k[:bsz, :length // ratio]
        outputs, candidates = [], []
        for start in range(0, length, query_chunk):
            end = min(length, start + query_chunk)
            q = self.wq_b(qr[:, start:end]).unflatten(-1, (self.n_local_heads, self.index_head_dim))
            reference.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start:end])
            reference.fp4_act_quant(q, reference.fp4_block_size, True)
            weights = self.weights_proj(x[:, start:end]) * (self.softmax_scale * self.n_heads**-.5)
            scores = torch.einsum("bshd,btd->bsht", q, keys)
            scores = (scores.relu() * weights.unsqueeze(-1)).sum(2)
            visible = (torch.arange(start + 1, end + 1, device=x.device) // ratio).unsqueeze(-1)
            scores.masked_fill_(torch.arange(length // ratio, device=x.device) >= visible, -torch.inf)
            if self.is_candidate_source:
                candidates.append(reference.select_candidate_blocks(scores, visible, self.candidate_topk_blocks,
                                                                     self.candidate_block_size))
            elif self.uses_candidates:
                scores.masked_fill_(~shared.candidates[:, start:end], -torch.inf)
            top = scores.topk(min(self.index_topk, length // ratio), dim=-1, sorted=False).indices.sort(-1).values
            outputs.append(torch.where(top < visible, top + offset, -1).int())
        if self.is_candidate_source:
            shared.candidates = torch.cat(candidates, 1)
        return torch.cat(outputs, 1)
    reference.Indexer.forward = forward
