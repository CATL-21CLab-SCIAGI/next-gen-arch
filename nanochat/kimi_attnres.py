"""Kimi K3-style Block Attention Residuals on the nanochat backbone.

This module changes only the depth-wise residual pathway. Token mixing, MLPs,
value embeddings, RoPE, token smear, and their initializers remain nanochat's.
The implementation follows the released Kimi K3 reference ordering: one
depth-attention read before attention, one before the MLP, and one final read.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.common import COMPUTE_DTYPE, get_dist_info, print0
from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.optim import DistMuonAdamW, MuonAdamW


@dataclass
class KimiAttnResConfig(GPTConfig):
    arch_family: str = "kimi_attnres"
    # K3 uses 12 transformer layers/block at depth 93, i.e. about eight
    # blocks. Two layers/block preserves that granularity at d14.
    attn_res_block_size: int = 2
    attn_res_recompute: bool = True
    attn_res_variant: str = "kimi_k3_block_attnres"
    # H=1 is the original Kimi read. H=8 is Multi-Head Attention Residuals
    # (MHAR), which reshapes the same width-sized query without adding params.
    attn_res_heads: int = 1


class AttnResRead(nn.Module):
    """Single-pseudo-query softmax attention over residual sources."""

    def __init__(self, width: int, heads: int = 1, eps: float = 1e-6):
        super().__init__()
        if heads <= 0 or width % heads:
            raise ValueError("AttnRes width must be divisible by a positive head count")
        self.heads = int(heads)
        self.query = nn.Parameter(torch.zeros(width))
        self.norm_weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, *sources: torch.Tensor) -> torch.Tensor:
        if not sources:
            raise ValueError("AttnRes requires at least one residual source")
        shape = sources[0].shape
        if any(source.shape != shape for source in sources):
            raise ValueError("all AttnRes sources must share a shape")

        # Match K3's released reference: scores and aggregation are FP32, then
        # cast back to the residual-stream dtype. The source-wise loop avoids a
        # large [sources, batch, time, width] FP32 materialization.
        score_weight = self.norm_weight.float() * self.query.float()
        if self.heads > 1:
            head_dim = shape[-1] // self.heads
            score_weight = score_weight.view(self.heads, head_dim)
            scores = []
            values = []
            for source in sources:
                value = source.float()
                inv_rms = torch.rsqrt(value.square().mean(dim=-1) + self.eps)
                normalized = value * inv_rms.unsqueeze(-1)
                normalized = normalized.view(*shape[:-1], self.heads, head_dim)
                scores.append((normalized * score_weight).sum(dim=-1))
                values.append(value.view(*shape[:-1], self.heads, head_dim))
            probabilities = torch.softmax(torch.stack(scores, dim=-2), dim=-2)
            output = torch.zeros_like(values[0])
            for source_idx, value in enumerate(values):
                output = output + probabilities[..., source_idx, :, None] * value
            return output.reshape(shape).to(sources[0].dtype)

        scores = []
        for source in sources:
            value = source.float()
            inv_rms = torch.rsqrt(value.square().mean(dim=-1) + self.eps)
            scores.append((value * inv_rms.unsqueeze(-1) * score_weight).sum(dim=-1))
        probabilities = torch.softmax(torch.stack(scores, dim=-1), dim=-1)

        output = torch.zeros_like(sources[0], dtype=torch.float32)
        for source_idx, source in enumerate(sources):
            output = output + probabilities[..., source_idx, None] * source.float()
        return output.to(sources[0].dtype)


class KimiAttnRes(GPT):
    """nanochat d14 with the Kimi K3 Block AttnRes residual operator."""

    def __init__(self, config: KimiAttnResConfig, pad_vocab_size_to: int = 64):
        if config.attn_res_block_size <= 0:
            raise ValueError("attn_res_block_size must be positive")
        if config.attn_res_variant not in {"kimi_k3_block_attnres", "multi_head_attnres"}:
            raise ValueError("unsupported attention-residual variant")
        if config.attn_res_heads <= 0 or config.n_embd % config.attn_res_heads:
            raise ValueError("attention-residual heads must divide model width")
        if config.attn_res_variant == "kimi_k3_block_attnres" and config.attn_res_heads != 1:
            raise ValueError("Kimi K3 Block AttnRes uses one routing head")
        if config.attn_res_variant == "multi_head_attnres" and config.attn_res_heads <= 1:
            raise ValueError("MHAR requires more than one routing head")
        super().__init__(config, pad_vocab_size_to=pad_vocab_size_to)

        # These nanochat residual controls are alternative depth-mixing rules.
        # Keep non-persistent buffers so GPT.init_weights consumes the identical
        # shared RNG stream, but remove them from the trainable/state contract.
        del self.resid_lambdas
        del self.x0_lambdas
        del self.backout_lambda
        self.register_buffer("resid_lambdas", torch.empty(config.n_layer), persistent=False)
        self.register_buffer("x0_lambdas", torch.empty(config.n_layer), persistent=False)
        self.register_buffer("backout_lambda", torch.empty(1), persistent=False)

        self.attention_residual_reads = nn.ModuleList(
            [AttnResRead(config.n_embd, heads=config.attn_res_heads) for _ in range(config.n_layer)]
        )
        self.mlp_residual_reads = nn.ModuleList(
            [AttnResRead(config.n_embd, heads=config.attn_res_heads) for _ in range(config.n_layer)]
        )
        self.output_residual_read = AttnResRead(config.n_embd, heads=config.attn_res_heads)

    @torch.no_grad()
    def init_weights(self):
        # Shared tensors consume exactly the baseline initialization stream.
        GPT.init_weights(self)
        self.backout_lambda.zero_()
        for read in self._attn_res_reads():
            read.query.zero_()
            read.norm_weight.fill_(1.0)

    def _attn_res_reads(self):
        yield from self.attention_residual_reads
        yield from self.mlp_residual_reads
        yield self.output_residual_read

    def _apply_read(self, read: AttnResRead, sources: list[torch.Tensor]) -> torch.Tensor:
        if self.training and self.config.attn_res_recompute and torch.is_grad_enabled():
            return checkpoint(read, *sources, use_reentrant=False)
        return read(*sources)

    def residual_source_counts(self) -> dict[str, list[int] | int]:
        """Return the exact static source map used by this depth/block layout."""
        completed = 0
        pre_attention = []
        pre_mlp = []
        for layer_idx in range(self.config.n_layer):
            pre_attention.append(completed + 1)
            if layer_idx % self.config.attn_res_block_size == 0:
                completed += 1
            pre_mlp.append(completed + 1)
        return {
            "pre_attention": pre_attention,
            "pre_mlp": pre_mlp,
            "final": completed + 1,
        }

    def get_architecture_state(self) -> dict:
        counts = self.residual_source_counts()
        return {
            "family": "kimi_attnres",
            "variant": self.config.attn_res_variant,
            "routing_heads": self.config.attn_res_heads,
            "routing_head_dim": self.config.n_embd // self.config.attn_res_heads,
            "block_size_transformer_layers": self.config.attn_res_block_size,
            "completed_transformer_blocks": (self.config.n_layer + self.config.attn_res_block_size - 1)
            // self.config.attn_res_block_size,
            "final_source_count_including_embedding": counts["final"],
            "pre_attention_source_counts": counts["pre_attention"],
            "pre_mlp_source_counts": counts["pre_mlp"],
            "pseudo_query_initialization": "zeros",
            "score_normalization": "learned_rmsnorm_fp32",
            "aggregation_dtype": "float32",
            "activation_recomputation": self.config.attn_res_recompute,
            "standard_resid_lambdas": "replaced",
            "x0_lambdas": "replaced_by_embedding_source",
            "backout": "replaced_by_final_attnres_read",
        }

    def _attn_res_parameter_count(self) -> int:
        return sum(p.numel() for read in self._attn_res_reads() for p in read.parameters())

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        smear = self.smear_gate.weight.numel() + self.smear_lambda.numel()
        attn_res = self._attn_res_parameter_count()
        total = wte + value_embeds + lm_head + transformer_matrices + smear + attn_res
        assert total == sum(p.numel() for p in self.parameters()), "parameter count mismatch"
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "smear": smear,
            "attn_res": attn_res,
            "total": total,
        }

    def estimate_flops(self):
        # Forward+backward convention matches GPT: 6 FLOPs per matrix weight,
        # plus 12*h*q*context for attention. Each AttnRes source contributes
        # two d-dimensional dot-like reductions (score and weighted read), or
        # 12*d FLOPs including backward; norm/softmax elementwise FLOPs omitted.
        matrix_params = sum(p.numel() for p in self.transformer.h.parameters())
        matrix_params += sum(p.numel() for p in self.lm_head.parameters())
        h = self.config.n_head
        q = self.config.n_embd // h
        seq_len = self.config.sequence_len
        attention_flops = 0
        for window_size in self.window_sizes:
            effective_seq = min(window_size[0], seq_len) if window_size[0] >= 0 else seq_len
            attention_flops += 12 * h * q * effective_seq
        counts = self.residual_source_counts()
        source_reads = sum(counts["pre_attention"]) + sum(counts["pre_mlp"]) + counts["final"]
        attn_res_flops = 12 * self.config.n_embd * source_reads
        return 6 * matrix_params + attention_flops + attn_res_flops

    def estimate_executed_flops(self):
        return self.estimate_flops()

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        scalar_lr=0.5,
    ):
        model_dim = self.config.n_embd
        ddp, _, _, _ = get_dist_info()
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        smear_params = [self.smear_gate.weight, self.smear_lambda]
        attn_res_params = [p for read in self._attn_res_reads() for p in read.parameters()]
        grouped = matrix_params + value_embeds_params + embedding_params + lm_head_params + smear_params + attn_res_params
        assert len(grouped) == len(list(self.parameters()))

        scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {scale:.6f}")
        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=value_embeds_params, lr=embedding_lr * scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            dict(
                kind="adamw",
                params=attn_res_params,
                lr=scalar_lr * 0.01,
                betas=(0.8, 0.95),
                eps=1e-10,
                weight_decay=0.0,
                attn_res=True,
            ),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            param_groups.append(dict(
                kind="muon",
                params=[p for p in matrix_params if p.shape == shape],
                lr=matrix_lr,
                momentum=0.95,
                ns_steps=5,
                beta2=0.9,
                weight_decay=weight_decay,
            ))
        optimizer = (DistMuonAdamW if ddp else MuonAdamW)(param_groups)
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        _, time = idx.size()
        if time > self.cos.size(1):
            raise ValueError(f"sequence length {time} exceeds rotary cache {self.cos.size(1)}")
        if idx.device != self.cos.device:
            raise ValueError("rotary embeddings and tokens are on different devices")
        if self.cos.dtype != COMPUTE_DTYPE:
            raise ValueError(f"rotary embeddings must use {COMPUTE_DTYPE}, got {self.cos.dtype}")
        offset = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, offset : offset + time], self.sin[:, offset : offset + time]

        hidden = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        if kv_cache is None:
            if time <= 1:
                raise ValueError("training forward requires more than one token")
            gate = self.smear_lambda.to(hidden.dtype) * torch.sigmoid(
                self.smear_gate(hidden[:, 1:, : self.smear_gate_channels])
            )
            hidden = torch.cat([hidden[:, :1], hidden[:, 1:] + gate * hidden[:, :-1]], dim=1)
        else:
            previous = kv_cache.prev_embedding
            kv_cache.prev_embedding = hidden[:, -1:, :]
            if time > 1:
                gate = self.smear_lambda.to(hidden.dtype) * torch.sigmoid(
                    self.smear_gate(hidden[:, 1:, : self.smear_gate_channels])
                )
                hidden = torch.cat([hidden[:, :1], hidden[:, 1:] + gate * hidden[:, :-1]], dim=1)
            elif previous is not None:
                gate = self.smear_lambda.to(hidden.dtype) * torch.sigmoid(
                    self.smear_gate(hidden[:, :, : self.smear_gate_channels])
                )
                hidden = hidden + gate * previous

        completed_blocks: list[torch.Tensor] = []
        partial_block = hidden
        for layer_idx, block in enumerate(self.transformer.h):
            layer_input = self._apply_read(
                self.attention_residual_reads[layer_idx],
                completed_blocks + [partial_block],
            )
            if layer_idx % self.config.attn_res_block_size == 0:
                completed_blocks.append(partial_block)
                partial_block = None

            value_embedding = (
                self.value_embeds[str(layer_idx)](idx).to(layer_input.dtype)
                if str(layer_idx) in self.value_embeds
                else None
            )
            attention_output = block.attn(
                norm(layer_input), value_embedding, cos_sin, self.window_sizes[layer_idx], kv_cache
            )
            partial_block = attention_output if partial_block is None else partial_block + attention_output

            mlp_input = self._apply_read(
                self.mlp_residual_reads[layer_idx],
                completed_blocks + [partial_block],
            )
            partial_block = partial_block + block.mlp(norm(mlp_input))

        hidden = self._apply_read(self.output_residual_read, completed_blocks + [partial_block])
        hidden = norm(hidden)
        logits = self.lm_head(hidden)[..., : self.config.vocab_size].float()
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)
        if targets is None:
            return logits
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
