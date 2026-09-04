"""Full and quarter-shape text backbones derived from ``Qwen/Qwen3.8-27B``.

The released checkpoint is multimodal, but FineWeb-Edu pretraining exercises only
its dense Qwen3.5-family text backbone. The full configuration preserves its released
geometry; the historical quarter variant divides capacity dimensions by four while
retaining vocabulary identity, context reach, layer cadence, and convolution extent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from archlab.architectures.qwen38_flash_next import (
    GatedDeltaAttention,
    RMSNorm,
    _linear,
)

SOURCE_MODEL = "Qwen/Qwen3.8-27B"
SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCE_CONFIG_SHA256 = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"


@dataclass(frozen=True)
class Qwen38DenseConfig:
    # Token IDs and positional reach are identities, not capacity dimensions.
    vocab_size: int = 248_320
    sequence_len: int = 2_048
    max_position_embeddings: int = 262_144
    eos_token_id: int = 248_044

    # Qwen3.8-27B text geometry divided by four.
    num_hidden_layers: int = 16
    hidden_size: int = 1_280
    intermediate_size: int = 4_352
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25

    # One gated full-attention layer follows every three Gated DeltaNet layers.
    full_attention_interval: int = 4
    attention_heads: int = 6
    attention_kv_heads: int = 1
    attention_head_dim: int = 64

    linear_qk_heads: int = 4
    linear_v_heads: int = 12
    linear_key_dim: int = 32
    linear_value_dim: int = 32
    # Kernel extent is an operator property and remains the released value.
    linear_conv_kernel: int = 4

    # The released training config retains one multi-token-prediction layer.
    mtp_layers: int = 1
    mtp_loss_weight: float = 0.1
    mtp_fusion: bool = False
    tie_word_embeddings: bool = False
    arch_family: str = "qwen38_27b_quarter_text"

    @classmethod
    def for_scale(cls, scale: str, **overrides) -> Qwen38DenseConfig:
        """Build either the historical quarter geometry or released full text geometry."""
        if scale == "quarter":
            return replace(cls(), **overrides)
        if scale != "full":
            raise ValueError(f"unsupported Qwen3.8-27B scale: {scale}")
        values = {
            "num_hidden_layers": 64,
            "hidden_size": 5_120,
            "intermediate_size": 17_408,
            "attention_heads": 24,
            "attention_kv_heads": 4,
            "attention_head_dim": 256,
            "linear_qk_heads": 16,
            "linear_v_heads": 48,
            "linear_key_dim": 128,
            "linear_value_dim": 128,
            "mtp_fusion": True,
            "arch_family": "qwen38_27b_text",
        }
        values.update(overrides)
        return replace(cls(), **values)

    def __post_init__(self) -> None:
        positive = (
            self.vocab_size,
            self.sequence_len,
            self.max_position_embeddings,
            self.num_hidden_layers,
            self.hidden_size,
            self.intermediate_size,
            self.full_attention_interval,
            self.attention_heads,
            self.attention_kv_heads,
            self.attention_head_dim,
            self.linear_qk_heads,
            self.linear_v_heads,
            self.linear_key_dim,
            self.linear_value_dim,
            self.linear_conv_kernel,
            self.mtp_layers,
        )
        if min(positive) < 1:
            raise ValueError("Qwen3.8-27B dimensions must be positive")
        if self.attention_heads % self.attention_kv_heads:
            raise ValueError("attention KV heads must divide query heads")
        if self.linear_v_heads % self.linear_qk_heads:
            raise ValueError("linear QK heads must divide value heads")
        rotary_dim = int(self.attention_head_dim * self.partial_rotary_factor)
        if rotary_dim < 2 or rotary_dim % 2:
            raise ValueError("partial RoPE must retain a positive even dimension")
        if self.mtp_layers != 1:
            raise ValueError("the source and quarter contracts retain one MTP layer")

    def to_dict(self) -> dict:
        return asdict(self)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    rotary, remainder = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = rotary * cos + _rotate_half(rotary) * sin
    return torch.cat((rotated, remainder), dim=-1)


class Qwen38DenseAttention(nn.Module):
    """Qwen3.8 gated GQA with per-head Q/K norm and partial RoPE."""

    def __init__(self, config: Qwen38DenseConfig, *, runtime_backend: str):
        super().__init__()
        self.q_heads = config.attention_heads
        self.kv_heads = config.attention_kv_heads
        self.head_dim = config.attention_head_dim
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        q_width = self.q_heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        # The released implementation fuses the sigmoid output gate into Q.
        self.q_gate = _linear(
            config.hidden_size,
            2 * q_width,
            runtime_backend=runtime_backend,
        )
        self.k = _linear(config.hidden_size, kv_width, runtime_backend=runtime_backend)
        self.v = _linear(config.hidden_size, kv_width, runtime_backend=runtime_backend)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.out = _linear(q_width, config.hidden_size, runtime_backend=runtime_backend)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q_width = self.q_heads * self.head_dim
        query, gate = (
            self.q_gate(x).view(batch, seq_len, self.q_heads, 2 * self.head_dim).chunk(2, dim=-1)
        )
        key = self.k(x).view(batch, seq_len, self.kv_heads, self.head_dim)
        value = self.v(x).view(batch, seq_len, self.kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query = _apply_partial_rope(query, cos, sin, self.rotary_dim)
        key = _apply_partial_rope(key, cos, sin, self.rotary_dim)
        if self.q_heads != self.kv_heads:
            repeat = self.q_heads // self.kv_heads
            key = key.repeat_interleave(repeat, dim=2)
            value = value.repeat_interleave(repeat, dim=2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=True,
        ).transpose(1, 2)
        attended = attended * torch.sigmoid(gate)
        return self.out(attended.reshape(batch, seq_len, q_width))


class Qwen38DenseMLP(nn.Module):
    def __init__(self, config: Qwen38DenseConfig, *, runtime_backend: str):
        super().__init__()
        self.gate = _linear(
            config.hidden_size,
            config.intermediate_size,
            runtime_backend=runtime_backend,
        )
        self.up = _linear(
            config.hidden_size,
            config.intermediate_size,
            runtime_backend=runtime_backend,
        )
        self.down = _linear(
            config.intermediate_size,
            config.hidden_size,
            runtime_backend=runtime_backend,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Qwen38DenseBlock(nn.Module):
    def __init__(
        self,
        config: Qwen38DenseConfig,
        layer_idx: int,
        *,
        runtime_backend: str,
        gdn_kernel: str,
        force_full_attention: bool = False,
    ):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        use_full = force_full_attention or (layer_idx + 1) % config.full_attention_interval == 0
        self.attention_kind = "full_attention" if use_full else "gdn"
        self.attention = (
            Qwen38DenseAttention(config, runtime_backend=runtime_backend)
            if use_full
            else GatedDeltaAttention(
                config,
                runtime_backend=runtime_backend,
                gdn_kernel=gdn_kernel,
            )
        )
        self.mlp = Qwen38DenseMLP(config, runtime_backend=runtime_backend)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        branch = self.input_norm(x)
        if self.attention_kind == "full_attention":
            branch = self.attention(branch, cos, sin)
        else:
            branch = self.attention(branch)
        x = x + branch
        return x + self.mlp(self.mlp_norm(x))


class Qwen38MTP(nn.Module):
    """Released Qwen3.8 MTP fusion followed by one full-attention block."""

    def __init__(
        self,
        config: Qwen38DenseConfig,
        *,
        runtime_backend: str,
        gdn_kernel: str,
    ):
        super().__init__()
        self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.fc = _linear(
            2 * config.hidden_size,
            config.hidden_size,
            runtime_backend=runtime_backend,
        )
        self.block = Qwen38DenseBlock(
            config,
            config.num_hidden_layers,
            runtime_backend=runtime_backend,
            gdn_kernel=gdn_kernel,
            force_full_attention=True,
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        shifted_embedding: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat(
            (
                self.pre_fc_norm_hidden(hidden),
                self.pre_fc_norm_embedding(shifted_embedding),
            ),
            dim=-1,
        )
        return self.norm(self.block(self.fc(fused), cos, sin))


class Qwen38Dense(nn.Module):
    def __init__(
        self,
        config: Qwen38DenseConfig,
        *,
        runtime_backend: str = "native",
        gdn_kernel: str = "fla",
    ):
        super().__init__()
        self.config = config
        self.runtime_backend = runtime_backend
        self.gdn_kernel = gdn_kernel
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen38DenseBlock(
                    config,
                    layer_idx,
                    runtime_backend=runtime_backend,
                    gdn_kernel=gdn_kernel,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = _linear(
            config.hidden_size,
            config.vocab_size,
            runtime_backend=runtime_backend,
        )
        if config.mtp_fusion:
            self.mtp = Qwen38MTP(
                config,
                runtime_backend=runtime_backend,
                gdn_kernel=gdn_kernel,
            )
        else:
            # Retain the historical quarter-run state-dict layout. Full-scale runs
            # use the released normalized concatenation and projection above.
            self.mtp_block = Qwen38DenseBlock(
                config,
                config.num_hidden_layers,
                runtime_backend=runtime_backend,
                gdn_kernel=gdn_kernel,
                force_full_attention=True,
            )
        self._configure_optimizer_parameters()
        self.last_loss_metrics: dict[str, torch.Tensor] = {}

        rotary_dim = int(config.attention_head_dim * config.partial_rotary_factor)
        positions = torch.arange(config.sequence_len, dtype=torch.float32)
        channels = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
        inverse_frequency = 1.0 / (config.rope_theta ** (channels / rotary_dim))
        angles = torch.outer(positions, inverse_frequency)
        embeddings = torch.cat((angles, angles), dim=-1)
        self.register_buffer("rope_cos", embeddings.cos(), persistent=False)
        self.register_buffer("rope_sin", embeddings.sin(), persistent=False)

    @staticmethod
    def _tag_optimizer_module(
        module: nn.Module,
        *,
        optimizer: str,
        category: str,
    ) -> None:
        for parameter in module.parameters():
            parameter.archlab_optimizer = optimizer
            parameter.archlab_optimizer_category = category
            parameter.is_embedding_or_output_parameter = optimizer != "muon"

    def _configure_optimizer_parameters(self) -> None:
        self._tag_optimizer_module(
            self,
            optimizer="adamw",
            category="scalar_norm_or_non_linear",
        )
        self._tag_optimizer_module(
            self.token_embedding,
            optimizer="adamw",
            category="input_embedding",
        )
        self._tag_optimizer_module(
            self.lm_head,
            optimizer="adamw",
            category="output_head",
        )
        mtp_block = self.mtp.block if self.config.mtp_fusion else self.mtp_block
        for block in (*self.layers, mtp_block):
            attention = block.attention
            if isinstance(attention, GatedDeltaAttention):
                self._tag_optimizer_module(
                    attention.qkv,
                    optimizer="muon",
                    category="gdn_qkv",
                )
                self._tag_optimizer_module(
                    attention.out,
                    optimizer="muon",
                    category="gdn_output",
                )
                self._tag_optimizer_module(
                    attention.z,
                    optimizer="adamw",
                    category="gdn_output_gate",
                )
                self._tag_optimizer_module(
                    attention.ba,
                    optimizer="adamw",
                    category="gdn_decay_beta",
                )
                self._tag_optimizer_module(
                    attention.conv,
                    optimizer="adamw",
                    category="gdn_depthwise_convolution",
                )
            else:
                self._tag_optimizer_module(
                    attention.q_gate,
                    optimizer="muon",
                    category="attention_query_and_gate",
                )
                for name in ("k", "v", "out"):
                    self._tag_optimizer_module(
                        getattr(attention, name),
                        optimizer="muon",
                        category=f"attention_{name}",
                    )
            for name in ("gate", "up", "down"):
                self._tag_optimizer_module(
                    getattr(block.mlp, name),
                    optimizer="muon",
                    category=f"dense_mlp_{name}",
                )
        if self.config.mtp_fusion:
            self._tag_optimizer_module(
                self.mtp.fc,
                optimizer="muon",
                category="mtp_fusion",
            )

    def optimizer_contract(self, *, require_two_dimensional_muon: bool = False) -> dict:
        optimizers: dict[str, dict[str, int]] = {}
        categories: dict[str, dict[str, int | str]] = {}
        missing: list[str] = []
        invalid: list[str] = []
        for name, parameter in self.named_parameters():
            optimizer = getattr(parameter, "archlab_optimizer", None)
            category = getattr(parameter, "archlab_optimizer_category", None)
            if optimizer is None or category is None:
                missing.append(name)
                continue
            if optimizer == "muon" and require_two_dimensional_muon and parameter.ndim != 2:
                invalid.append(
                    f"{name}: expected 2D Muon parameter, found {tuple(parameter.shape)}"
                )
            matrix_count = 1 if optimizer == "muon" else 0
            bucket = optimizers.setdefault(
                optimizer,
                {"tensors": 0, "parameters": 0, "logical_matrices": 0},
            )
            bucket["tensors"] += 1
            bucket["parameters"] += parameter.numel()
            bucket["logical_matrices"] += matrix_count
            category_bucket = categories.setdefault(
                category,
                {
                    "optimizer": optimizer,
                    "tensors": 0,
                    "parameters": 0,
                    "logical_matrices": 0,
                },
            )
            category_bucket["tensors"] = int(category_bucket["tensors"]) + 1
            category_bucket["parameters"] = int(category_bucket["parameters"]) + parameter.numel()
            category_bucket["logical_matrices"] = (
                int(category_bucket["logical_matrices"]) + matrix_count
            )
        if missing or invalid:
            details = "; ".join([*(f"untagged: {name}" for name in missing), *invalid])
            raise RuntimeError(f"invalid Qwen3.8-27B optimizer partition: {details}")
        return {
            "optimizers": optimizers,
            "categories": categories,
            "all_trainable_parameters_assigned_once": True,
            "frozen_parameters": 0,
        }

    @torch.no_grad()
    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, RMSNorm) and not module.weight.is_meta:
                module.weight.fill_(1.0)
            if isinstance(module, nn.Embedding) and not module.weight.is_meta:
                nn.init.normal_(module.weight, mean=0.0, std=std)
        for name, parameter in self.named_parameters():
            if parameter.is_meta or parameter.ndim < 2 or "embedding" in name:
                continue
            nn.init.normal_(parameter, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, GatedDeltaAttention) and not module.a_log.is_meta:
                module.a_log.copy_(torch.empty_like(module.a_log).uniform_(0.01, 16.0).log())
                module.dt_bias.fill_(1.0)

    def get_device(self) -> torch.device:
        return self.token_embedding.weight.device

    def _token_cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.device.type == "cuda" and self.runtime_backend.startswith("te_"):
            from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

            return parallel_cross_entropy(
                logits,
                labels,
                reduce_loss=False,
                ignore_idx=-1,
            )
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-1,
            reduction="none",
        ).view_as(labels)

    def num_scaling_params(self) -> dict[str, int]:
        embeddings = self.token_embedding.weight.numel() + self.lm_head.weight.numel()
        mtp_module = self.mtp if self.config.mtp_fusion else self.mtp_block
        mtp = sum(parameter.numel() for parameter in mtp_module.parameters())
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "embeddings_and_head": embeddings,
            "text_backbone": total - embeddings - mtp,
            "mtp": mtp,
            "total": total,
        }

    def estimate_flops(self) -> float:
        return 6.0 * float(self.num_scaling_params()["total"])

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        loss_reduction: str = "mean",
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.rope_cos.size(0):
            raise ValueError("sequence exceeds the configured rotary cache")
        seq_len = input_ids.size(1)
        compute_dtype = self.rope_cos.dtype
        if self.get_device().type == "cuda":
            compute_dtype = torch.bfloat16
        x = self.token_embedding(input_ids).to(compute_dtype)
        cos = self.rope_cos[:seq_len].to(device=x.device, dtype=x.dtype).view(1, seq_len, 1, -1)
        sin = self.rope_sin[:seq_len].to(device=x.device, dtype=x.dtype).view(1, seq_len, 1, -1)
        for layer in self.layers:
            x = layer(x, cos, sin)
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)[..., : self.config.vocab_size].float()
        if labels is None:
            return logits

        cross_entropy_losses = self._token_cross_entropy(logits, labels)
        mtp_contribution = torch.zeros_like(cross_entropy_losses)
        if seq_len > 1:
            shifted_embedding = self.token_embedding(input_ids[:, 1:]).to(hidden.dtype)
            if self.config.mtp_fusion:
                mtp_hidden = self.mtp(
                    hidden[:, :-1],
                    shifted_embedding,
                    cos[:, :-1],
                    sin[:, :-1],
                )
                mtp_logits = self.lm_head(mtp_hidden).float()
            else:
                mtp_hidden = self.mtp_block(
                    hidden[:, :-1] + shifted_embedding,
                    cos[:, :-1],
                    sin[:, :-1],
                )
                mtp_logits = self.lm_head(self.final_norm(mtp_hidden)).float()
            mtp_losses = self._token_cross_entropy(mtp_logits, labels[:, 1:])
            mtp_contribution[:, :-1] = self.config.mtp_loss_weight * mtp_losses
        valid = labels.ne(-1)
        losses = cross_entropy_losses + mtp_contribution
        valid_count = valid.sum().clamp_min(1)
        self.last_loss_metrics = {
            "cross entropy": (cross_entropy_losses * valid).sum().detach() / valid_count,
            "mtp loss": (mtp_contribution * valid).sum().detach() / valid_count,
        }
        if loss_reduction == "none":
            return losses
        if loss_reduction == "sum":
            return losses.sum()
        if loss_reduction == "mean":
            return losses.sum() / valid_count
        raise ValueError(f"unsupported loss reduction: {loss_reduction}")
