"""Controlled DeepSeek Sparse Attention ablation on the nanochat backbone.

This module preserves nanochat's dense MHA projections and adds the lightning
indexer from DeepSeek-V3.2.  The selected-token semantics are exact, while the
attention computation deliberately uses a dense masked SDPA fallback because
the public FlashMLA sparse-prefill operator has no merged training backward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.gpt import CausalSelfAttention, GPT, GPTConfig, Linear, apply_rotary_emb
from nanochat.optim import DistMuonAdamW, MuonAdamW


DSA_TOP_K = 32
DSA_INDEX_HEADS = 4
DSA_INDEX_HEAD_DIM = 128
DSA_INDEX_ROPE_DIM = 64
DSA_DENSE_WARMUP_STEPS = 40
DSA_QUERY_CHUNK_SIZE = 128
DSA_WARMUP_INDEXER_LR = 1e-3
DSA_SPARSE_INDEXER_LR = 7.3e-6
DSA_BACKEND = "sdpa_masked"


@dataclass
class DeepSeekDSAConfig(GPTConfig):
    arch_family: str = "deepseek_dsa"
    dsa_top_k: int = DSA_TOP_K
    dsa_index_heads: int = DSA_INDEX_HEADS
    dsa_index_head_dim: int = DSA_INDEX_HEAD_DIM
    dsa_index_rope_dim: int = DSA_INDEX_ROPE_DIM
    dsa_dense_warmup_steps: int = DSA_DENSE_WARMUP_STEPS
    dsa_query_chunk_size: int = DSA_QUERY_CHUNK_SIZE
    dsa_backend: str = DSA_BACKEND
    dsa_warmup_indexer_lr: float = DSA_WARMUP_INDEXER_LR
    dsa_sparse_indexer_lr: float = DSA_SPARSE_INDEXER_LR


class TransposedLinear(nn.Module):
    """Linear projection stored as [in, out] for eight-rank AdamW sharding."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.to(dtype=x.dtype))


def _apply_partial_noninterleaved_rope(
    x: torch.Tensor,
    cos_sin: tuple[torch.Tensor, torch.Tensor],
    rope_dim: int,
) -> torch.Tensor:
    """Apply half-split (non-interleaved) RoPE to the leading channels only."""
    if rope_dim == 0:
        return x
    if rope_dim % 2 or rope_dim > x.size(-1):
        raise ValueError(f"Invalid indexer RoPE dimension {rope_dim} for width {x.size(-1)}")
    cos, sin = cos_sin
    rotated = apply_rotary_emb(x[..., :rope_dim], cos[..., : rope_dim // 2], sin[..., : rope_dim // 2])
    return torch.cat((rotated, x[..., rope_dim:]), dim=-1)


def _safe_kl(target: torch.Tensor, log_prediction: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean KL(target || prediction), avoiding 0 * -inf at masked positions."""
    target = target.masked_fill(~valid, 0.0)
    log_prediction = log_prediction.masked_fill(~valid, 0.0)
    log_target = target.clamp_min(torch.finfo(target.dtype).tiny).log()
    return (target * (log_target - log_prediction)).sum(dim=-1).mean()


class LightningIndexer(nn.Module):
    """Weighted-ReLU MQA indexer from the DeepSeek-V3.2 DSA operator."""

    def __init__(self, config: DeepSeekDSAConfig):
        super().__init__()
        self.hidden_size = config.n_embd
        self.n_heads = config.dsa_index_heads
        self.head_dim = config.dsa_index_head_dim
        self.rope_dim = config.dsa_index_rope_dim
        self.top_k = config.dsa_top_k
        self.query_chunk_size = config.dsa_query_chunk_size
        if self.head_dim <= 0:
            raise ValueError("DSA indexer head dimension must be positive")
        if self.rope_dim < 0 or self.rope_dim > self.head_dim or self.rope_dim % 2:
            raise ValueError("DSA indexer RoPE dimension must be even and fit the indexer head")
        if self.top_k <= 0:
            raise ValueError("DSA top-k must be positive")
        if self.query_chunk_size <= 0:
            raise ValueError("DSA query chunk size must be positive")

        self.q_proj = Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = TransposedLinear(self.hidden_size, self.n_heads)
        self.softmax_scale = self.head_dim**-0.5

    def project(
        self,
        hidden_states: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detached = hidden_states.detach()
        batch, seq_len, _ = detached.shape
        q = self.q_proj(detached).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(detached)
        # Indexer norms intentionally remain FP32 optimizer parameters. Cast the
        # affine tensors at the operation boundary so eager and compiled BF16
        # execution have identical dtype semantics.
        k = F.layer_norm(
            k,
            (self.head_dim,),
            self.k_norm.weight.to(k.dtype),
            self.k_norm.bias.to(k.dtype),
            self.k_norm.eps,
        ).unsqueeze(2)
        q = _apply_partial_noninterleaved_rope(q, cos_sin, self.rope_dim)
        k = _apply_partial_noninterleaved_rope(k, cos_sin, self.rope_dim).squeeze(2)
        weights = self.weights_proj(detached).float() * self.n_heads**-0.5
        return q, k, weights

    @staticmethod
    def _main_dense_target(
        main_q: torch.Tensor,
        main_k: torch.Tensor,
        start: int,
        end: int,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        head_dim = main_q.size(-1)
        logits = torch.einsum(
            "bqhd,bkhd->bhqk",
            main_q[:, start:end].detach(),
            main_k.detach(),
        ).float() * head_dim**-0.5
        logits = logits.masked_fill(~valid[:, None], float("-inf"))
        return logits.softmax(dim=-1).mean(dim=1)

    @staticmethod
    def _main_sparse_target(
        main_q: torch.Tensor,
        main_k: torch.Tensor,
        indices: torch.Tensor,
        valid: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        batch, _, n_heads, head_dim = main_k.shape
        query_len, selected = indices.shape[1:]
        offsets = torch.arange(batch, device=indices.device)[:, None, None] * main_k.size(1)
        flat_indices = (indices.long() + offsets).reshape(-1)
        selected_k = main_k.detach().reshape(batch * main_k.size(1), n_heads, head_dim)[flat_indices]
        selected_k = selected_k.view(batch, query_len, selected, n_heads, head_dim).permute(0, 3, 1, 2, 4)
        query = main_q[:, start:end].detach().permute(0, 2, 1, 3).unsqueeze(3)
        logits = (query * selected_k).sum(dim=-1).float() * head_dim**-0.5
        logits = logits.masked_fill(~valid[:, None], float("-inf"))
        return logits.softmax(dim=-1).mean(dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        main_q: torch.Tensor,
        main_k: torch.Tensor,
        *,
        dense_warmup: bool,
        compute_kl: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden_states.shape
        q, k, weights = self.project(hidden_states, cos_sin)
        all_indices: list[torch.Tensor] = []
        kl_terms: list[torch.Tensor] = []
        selected_mass_terms: list[torch.Tensor] = []
        key_positions = torch.arange(seq_len, device=hidden_states.device)

        for start in range(0, seq_len, self.query_chunk_size):
            end = min(start + self.query_chunk_size, seq_len)
            query_positions = torch.arange(start, end, device=hidden_states.device)
            causal = key_positions[None, :] <= query_positions[:, None]
            causal = causal.unsqueeze(0).expand(batch, -1, -1)

            per_head = torch.einsum("bqhd,bkd->bqhk", q[:, start:end], k)
            index_scores = (
                F.relu(per_head.float()) * weights[:, start:end, :, None]
            ).sum(dim=2) * self.softmax_scale
            index_scores = index_scores.masked_fill(~causal, float("-inf"))
            selected_values, indices = index_scores.topk(min(self.top_k, seq_len), dim=-1)
            selected_valid = indices <= query_positions[None, :, None]
            all_indices.append(indices)

            if not compute_kl:
                continue

            if dense_warmup:
                target = self._main_dense_target(main_q, main_k, start, end, causal)
                log_prediction = index_scores.log_softmax(dim=-1)
                kl_terms.append(_safe_kl(target, log_prediction, causal))
                selected_mass = target.gather(-1, indices).masked_fill(~selected_valid, 0.0).sum(dim=-1).mean()
                selected_mass_terms.append(selected_mass)
            else:
                target = self._main_sparse_target(main_q, main_k, indices, selected_valid, start, end)
                log_prediction = selected_values.log_softmax(dim=-1)
                kl_terms.append(_safe_kl(target, log_prediction, selected_valid))

        indices = torch.cat(all_indices, dim=1)
        zero = hidden_states.new_zeros((), dtype=torch.float32)
        kl = torch.stack(kl_terms).mean() if kl_terms else zero
        selected_mass = torch.stack(selected_mass_terms).mean() if selected_mass_terms else zero
        return indices, kl, selected_mass


class DSACausalSelfAttention(CausalSelfAttention):
    def __init__(self, config: DeepSeekDSAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer = LightningIndexer(config)
        self.sparse_enabled = False
        self.last_indexer_kl: torch.Tensor | None = None
        self.last_selected_mass: torch.Tensor | None = None

    def set_sparse_enabled(self, enabled: bool) -> None:
        self.sparse_enabled = bool(enabled)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError("DSA KV-cache inference is outside this controlled training ablation")
        if window_size[0] < x.size(1):
            raise ValueError("Controlled DSA requires full-context attention")
        batch, seq_len, _ = x.shape
        q = self.c_q(x).view(batch, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(batch, seq_len, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(batch, seq_len, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (self.head_dim,)), F.rms_norm(k, (self.head_dim,))
        q, k = q * 1.2, k * 1.2

        compute_indexer = self.sparse_enabled or self.training
        if compute_indexer:
            indices, kl, selected_mass = self.indexer(
                x,
                cos_sin,
                q,
                k,
                dense_warmup=not self.sparse_enabled,
                compute_kl=self.training,
            )
        else:
            indices = None
            kl = x.new_zeros((), dtype=torch.float32)
            selected_mass = x.new_zeros((), dtype=torch.float32)
        self.last_indexer_kl = kl
        self.last_selected_mass = selected_mass

        q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.sparse_enabled:
            allowed = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool, device=x.device)
            allowed.scatter_(2, indices.detach().long(), True)
            causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril_()
            allowed = allowed & causal.unsqueeze(0)
            y = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=allowed[:, None], dropout_p=0.0)
        else:
            y = F.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.n_embd)
        return self.c_proj(y)


class DeepSeekDSA(GPT):
    def __init__(self, config: DeepSeekDSAConfig, pad_vocab_size_to: int = 64):
        if config.dsa_backend != DSA_BACKEND:
            raise ValueError(f"Controlled DSA requires backend {DSA_BACKEND!r}")
        if config.n_kv_head != config.n_head:
            raise ValueError("Controlled DSA preserves nanochat MHA and does not enable GQA/MLA")
        super().__init__(config, pad_vocab_size_to=pad_vocab_size_to)
        for layer_idx, block in enumerate(self.transformer.h):
            block.attn = DSACausalSelfAttention(config, layer_idx)
        self.training_step = 0
        self._last_training_metrics: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def init_weights(self):
        # This consumes exactly the baseline RNG stream for every shared tensor.
        GPT.init_weights(self)
        device = self.transformer.wte.weight.device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        indexer_seed = torch.initial_seed() ^ 0xD5A
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(indexer_seed)
            bound = math.sqrt(3) * self.config.n_embd**-0.5
            for block in self.transformer.h:
                indexer = block.attn.indexer
                torch.nn.init.uniform_(indexer.q_proj.weight, -bound, bound)
                torch.nn.init.uniform_(indexer.k_proj.weight, -bound, bound)
                torch.nn.init.uniform_(indexer.weights_proj.weight, -bound, bound)
                indexer.k_norm.weight.fill_(1.0)
                indexer.k_norm.bias.zero_()

    def set_training_step(self, step: int) -> None:
        self.training_step = int(step)
        sparse = self.training_step >= self.config.dsa_dense_warmup_steps
        for block in self.transformer.h:
            block.attn.set_sparse_enabled(sparse)

    def get_architecture_state(self) -> dict:
        return {
            "training_step": self.training_step,
            "phase": "sparse" if self.training_step >= self.config.dsa_dense_warmup_steps else "dense_warmup",
            "backend": self.config.dsa_backend,
            "top_k": self.config.dsa_top_k,
        }

    def consume_training_metrics(self) -> dict[str, torch.Tensor]:
        metrics, self._last_training_metrics = self._last_training_metrics, {}
        return metrics

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        lm_loss_or_logits = super().forward(idx, targets, kv_cache, loss_reduction)
        if targets is None or not self.training:
            return lm_loss_or_logits
        indexer_kls = torch.stack([block.attn.last_indexer_kl for block in self.transformer.h])
        selected_masses = torch.stack([block.attn.last_selected_mass for block in self.transformer.h])
        indexer_kl = indexer_kls.mean()
        self._last_training_metrics = {
            "train/lm_loss": lm_loss_or_logits.detach(),
            "dsa/indexer_kl": indexer_kl.detach(),
            "dsa/selected_mass": selected_masses.mean().detach(),
            # Guard compiled training on the two-state phase, not the integer
            # training step (which would otherwise create one graph per step).
            "dsa/sparse_phase": lm_loss_or_logits.new_tensor(
                float(self.transformer.h[0].attn.sparse_enabled)
            ),
        }
        return lm_loss_or_logits + indexer_kl

    def _indexer_parameter_count(self) -> int:
        return sum(p.numel() for block in self.transformer.h for p in block.attn.indexer.parameters())

    def num_scaling_params(self):
        counts = super().num_scaling_params()
        indexer = self._indexer_parameter_count()
        counts["indexer"] = indexer
        # Keep scaling-law schedules tied to the shared baseline backbone.
        counts["transformer_matrices"] -= indexer
        return counts

    def _flop_components(self) -> tuple[int, int, int]:
        indexer_all_params = self._indexer_parameter_count()
        baseline_dense = GPT.estimate_flops(self) - 6 * indexer_all_params
        seq_len = self.config.sequence_len
        main_dense_attention = 12 * self.config.n_head * self.head_dim * seq_len * self.config.n_layer
        main_sparse_attention = 12 * self.config.n_head * self.head_dim * self.config.dsa_top_k * self.config.n_layer
        indexer_projection = 6 * sum(
            p.numel()
            for block in self.transformer.h
            for p in block.attn.indexer.parameters()
            if p.ndim == 2
        )
        indexer_scores = 6 * self.config.dsa_index_heads * self.config.dsa_index_head_dim * seq_len * self.config.n_layer
        indexer = indexer_projection + indexer_scores
        non_attention = baseline_dense - main_dense_attention
        return non_attention, main_sparse_attention, indexer

    @property
    def head_dim(self) -> int:
        return self.config.n_embd // self.config.n_head

    def estimate_flops(self):
        non_attention, sparse_attention, indexer = self._flop_components()
        return non_attention + sparse_attention + indexer

    def estimate_executed_flops(self):
        non_attention, _, indexer = self._flop_components()
        dense_attention = 12 * self.config.n_head * self.head_dim * self.config.sequence_len * self.config.n_layer
        return non_attention + dense_attention + indexer

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
        indexer_params = [p for block in self.transformer.h for p in block.attn.indexer.parameters()]
        indexer_ids = {id(p) for p in indexer_params}
        shared_transformer_params = [p for p in self.transformer.h.parameters() if id(p) not in indexer_ids]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        grouped = (
            shared_transformer_params + indexer_params + value_embeds_params + embedding_params
            + lm_head_params + resid_params + x0_params + smear_params
        )
        assert len(grouped) == len(list(self.parameters()))

        scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {scale:.6f}")
        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=value_embeds_params, lr=embedding_lr * scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind="adamw", params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            dict(
                kind="adamw",
                params=indexer_params,
                lr=self.config.dsa_warmup_indexer_lr,
                initial_lr=self.config.dsa_warmup_indexer_lr,
                dsa_indexer=True,
                dsa_warmup_lr=self.config.dsa_warmup_indexer_lr,
                dsa_sparse_lr=self.config.dsa_sparse_indexer_lr,
                betas=(0.8, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),
        ]
        for shape in sorted({p.shape for p in shared_transformer_params}):
            param_groups.append(dict(
                kind="muon",
                params=[p for p in shared_transformer_params if p.shape == shape],
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
