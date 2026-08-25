"""Controlled state-of-the-art architecture pool on the nanochat backbone.

Every arm changes one named mechanism while preserving nanochat's tokenizer,
attention/MLP widths, residual controls, value embeddings, and initialization
for all shared parameters.  The exceptions intrinsic to the cited mechanisms
are recorded in :meth:`get_architecture_state` (notably BoV replacing deep V
projections and Differential Attention's two attention maps).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from next_gen_arch.arch.base import (
    GPT,
    MLP,
    Block,
    CausalSelfAttention,
    GPTConfig,
    Linear,
    apply_qk_features,
    apply_rotary_emb,
    language_model_loss,
    norm,
)

SOTA_VARIANTS = frozenset(
    {
        "baseline",
        "gated_attention",
        "exclusive_attention",
        "differential_attention",
        "xielu",
        "dynamic_tanh",
        "peri_ln",
        "canon_abcd",
        "bank_of_values",
    }
)


@dataclass
class SotaPoolConfig(GPTConfig):
    arch_family: str = "sota_pool"
    sota_variant: str = "baseline"
    sota_extra_lr: float = 0.005
    canon_kernel_size: int = 4
    bov_target_fraction: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        if self.sota_variant not in SOTA_VARIANTS:
            raise ValueError(f"unknown sota_variant={self.sota_variant!r}")
        if self.sota_extra_lr <= 0:
            raise ValueError("sota_extra_lr must be positive")
        if self.canon_kernel_size <= 0:
            raise ValueError("canon_kernel_size must be positive")
        if not 0 < self.bov_target_fraction < 1:
            raise ValueError("bov_target_fraction must be in (0, 1)")
        if self.sota_variant == "differential_attention" and self.n_embd // self.n_head % 2:
            raise ValueError("Differential Attention requires an even head dimension")


class DynamicTanh(nn.Module):
    """Official DyT LLaMA form: ``weight * tanh(alpha * x)``."""

    def __init__(self, width: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1))
        self.weight = nn.Parameter(torch.empty(width))
        self.alpha_init = float(alpha_init)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.alpha.fill_(self.alpha_init)
            self.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight.to(x.dtype) * torch.tanh(self.alpha.to(x.dtype) * x)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class XIELU(nn.Module):
    """Reference xIELU equation with trainable positive/negative curvature."""

    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        if alpha_n_init <= beta:
            raise ValueError("alpha_n_init must exceed beta")
        self.alpha_p = nn.Parameter(torch.empty(1))
        self.alpha_n = nn.Parameter(torch.empty(1))
        self.alpha_p_init = float(alpha_p_init)
        self.alpha_n_init = float(alpha_n_init)
        self.beta = float(beta)
        self.eps = float(eps)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.alpha_p.fill_(_inverse_softplus(self.alpha_p_init))
            self.alpha_n.fill_(_inverse_softplus(self.alpha_n_init - self.beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = F.softplus(self.alpha_p.float()).to(x.dtype)
        alpha_n = (self.beta + F.softplus(self.alpha_n.float())).to(x.dtype)
        negative_x = torch.minimum(x, x.new_tensor(-self.eps))
        positive = alpha_p * x.square() + self.beta * x
        negative = alpha_n * torch.expm1(negative_x) - alpha_n * x + self.beta * x
        return torch.where(x > 0, positive, negative)


class CausalDepthwiseConv(nn.Module):
    """Canon layer: residual causal depthwise convolution along tokens."""

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))

    def reset_parameters(self) -> None:
        # Match nn.Conv1d's default Kaiming-uniform initialization, which is the
        # initialization explicitly retained by the Canon release.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        mixed = F.conv1d(
            x.transpose(1, 2),
            self.weight.to(x.dtype),
            padding=self.kernel_size - 1,
            groups=self.channels,
        )[..., :length]
        return x + mixed.transpose(1, 2)


def _bov_start_layer(config: SotaPoolConfig) -> int:
    count = max(1, int(config.n_layer * config.bov_target_fraction))
    return config.n_layer - count


class SotaAttention(CausalSelfAttention):
    def __init__(self, config: SotaPoolConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.variant = config.sota_variant
        self.is_bov_target = self.variant == "bank_of_values" and layer_idx >= _bov_start_layer(
            config
        )
        if self.variant == "gated_attention":
            self.output_gate = Linear(config.n_embd, config.n_head, bias=False)
        if self.variant == "differential_attention":
            half_dim = self.head_dim // 2
            self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
            self.lambda_q1 = nn.Parameter(torch.empty(half_dim))
            self.lambda_k1 = nn.Parameter(torch.empty(half_dim))
            self.lambda_q2 = nn.Parameter(torch.empty(half_dim))
            self.lambda_k2 = nn.Parameter(torch.empty(half_dim))
            self.diff_subln_weight = nn.Parameter(torch.empty(self.head_dim))
        if self.variant == "canon_abcd":
            self.canon_b = CausalDepthwiseConv(
                3 * config.n_embd, kernel_size=config.canon_kernel_size
            )
        if self.is_bov_target:
            padded_vocab = ((config.vocab_size + 63) // 64) * 64
            self.value_table = nn.Embedding(padded_vocab, config.n_embd)
            self.gamma_v = nn.Parameter(torch.empty(1))

    def reset_sota_parameters(self) -> None:
        n_embd = self.config.n_embd
        bound = math.sqrt(3.0) * n_embd**-0.5
        if hasattr(self, "output_gate"):
            nn.init.uniform_(self.output_gate.weight, -bound, bound)
        if self.variant == "differential_attention":
            for parameter in (self.lambda_q1, self.lambda_k1, self.lambda_q2, self.lambda_k2):
                nn.init.normal_(parameter, mean=0.0, std=0.1)
            self.diff_subln_weight.fill_(1.0)
        if hasattr(self, "canon_b"):
            self.canon_b.reset_parameters()
        if self.is_bov_target:
            self.gamma_v.fill_(1.0)

    def _project_qkv(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        kv_source: torch.Tensor | None = None,
    ):
        batch, time, _ = x.shape
        source = x if kv_source is None else kv_source
        q = self.c_q(x).view(batch, time, self.n_head, self.head_dim)
        k = self.c_k(source).view(batch, time, self.n_kv_head, self.head_dim)
        if self.is_bov_target:
            if ve is None:
                raise ValueError("BoV target layer requires a value-table lookup")
            v = self.gamma_v.to(x.dtype) * ve.view(batch, time, self.n_kv_head, self.head_dim)
        else:
            v = self.c_v(source).view(batch, time, self.n_kv_head, self.head_dim)
            if ve is not None:
                ve = ve.view(batch, time, self.n_kv_head, self.head_dim)
                gate = 3 * torch.sigmoid(self.ve_gate(source[..., : self.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve
        return q, k, v

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size,
        kv_cache,
    ) -> torch.Tensor:
        if kv_cache is None:
            return self.runtime.attention.flash_attn_func(
                q, k, v, causal=True, window_size=window_size
            )
        k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
        y = self.runtime.attention.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=kv_cache.cache_seqlens,
            causal=True,
            window_size=window_size,
        )
        if self.layer_idx == kv_cache.n_layers - 1:
            kv_cache.advance(q.size(1))
        return y

    def _differential_attention(self, q, k, v, cos_sin, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError(
                "Differential Attention KV-cache inference is not used in this campaign"
            )
        half = self.head_dim // 2
        batch, time = q.shape[:2]
        q = q.view(batch, time, self.n_head, 2, half)
        k = k.view(batch, time, self.n_kv_head, 2, half)
        cos, sin = cos_sin
        cos_half, sin_half = cos[..., : half // 2], sin[..., : half // 2]
        q1 = apply_rotary_emb(q[..., 0, :], cos_half, sin_half)
        q2 = apply_rotary_emb(q[..., 1, :], cos_half, sin_half)
        k1 = apply_rotary_emb(k[..., 0, :], cos_half, sin_half)
        k2 = apply_rotary_emb(k[..., 1, :], cos_half, sin_half)
        q1, q2, k1, k2 = norm(q1), norm(q2), norm(k1), norm(k2)
        q1, q2, k1, k2 = q1 * 1.2, q2 * 1.2, k1 * 1.2, k2 * 1.2

        def sdpa(qh, kh):
            return F.scaled_dot_product_attention(
                qh.transpose(1, 2),
                kh.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=True,
            ).transpose(1, 2)

        y1, y2 = sdpa(q1, k1), sdpa(q2, k2)
        lambda_1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum().float())
        lambda_2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum().float())
        lambda_full = (lambda_1 - lambda_2 + self.lambda_init).to(y1.dtype)
        y = y1 - lambda_full * y2
        y = norm(y) * self.diff_subln_weight.to(y.dtype)
        return y * (1.0 - self.lambda_init)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, kv_source=None):
        batch, time, _ = x.shape
        q, k, v = self._project_qkv(x, ve, kv_source=kv_source)
        if self.variant == "differential_attention":
            y = self._differential_attention(q, k, v, cos_sin, kv_cache)
        else:
            q, k = apply_qk_features(q, k, cos_sin, self.config, self.layer_idx, self.qk_gain)
            if self.variant == "canon_abcd":
                flat = torch.cat(
                    (
                        q.reshape(batch, time, -1),
                        k.reshape(batch, time, -1),
                        v.reshape(batch, time, -1),
                    ),
                    dim=-1,
                )
                q_flat, k_flat, v_flat = self.canon_b(flat).split(self.n_embd, dim=-1)
                q = q_flat.view(batch, time, self.n_head, self.head_dim)
                k = k_flat.view(batch, time, self.n_kv_head, self.head_dim)
                v = v_flat.view(batch, time, self.n_kv_head, self.head_dim)
            y = self._standard_attention(q, k, v, window_size, kv_cache)
            if self.variant == "gated_attention":
                gate = torch.sigmoid(self.output_gate(x)).unsqueeze(-1)
                y = y * gate
            elif self.variant == "exclusive_attention":
                value_direction = F.normalize(v, dim=-1)
                y = y - (y * value_direction).sum(dim=-1, keepdim=True) * value_direction
        return self.c_proj(y.contiguous().view(batch, time, -1))


class SotaMLP(MLP):
    def __init__(self, config: SotaPoolConfig):
        super().__init__(config)
        self.variant = config.sota_variant
        if self.variant == "xielu":
            self.activation = XIELU()
        if self.variant == "canon_abcd":
            self.canon_d = CausalDepthwiseConv(
                4 * config.n_embd, kernel_size=config.canon_kernel_size
            )

    def reset_sota_parameters(self) -> None:
        if hasattr(self, "activation"):
            self.activation.reset_parameters()
        if hasattr(self, "canon_d"):
            self.canon_d.reset_parameters()

    def forward(self, x):
        x = self.c_fc(x)
        if self.variant == "canon_abcd":
            x = self.canon_d(x)
        x = self.activation(x) if self.variant == "xielu" else F.relu(x).square()
        return self.c_proj(x)


class SotaBlock(Block):
    def __init__(self, config: SotaPoolConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.variant = config.sota_variant
        self.attn = SotaAttention(config, layer_idx)
        self.mlp = SotaMLP(config)
        if self.variant == "dynamic_tanh":
            self.attn_dyt = DynamicTanh(config.n_embd)
            self.mlp_dyt = DynamicTanh(config.n_embd)
        if self.variant == "canon_abcd":
            self.canon_a = CausalDepthwiseConv(config.n_embd, config.canon_kernel_size)
            self.canon_c = CausalDepthwiseConv(config.n_embd, config.canon_kernel_size)

    def reset_sota_parameters(self) -> None:
        self.attn.reset_sota_parameters()
        self.mlp.reset_sota_parameters()
        for name in ("attn_dyt", "mlp_dyt", "canon_a", "canon_c"):
            if hasattr(self, name):
                getattr(self, name).reset_parameters()

    def forward(
        self,
        x,
        ve,
        cos_sin,
        window_size,
        kv_cache,
        attention_input=None,
        kv_input=None,
    ):
        if self.variant == "dynamic_tanh":
            attention_input = self.attn_dyt(x) if attention_input is None else attention_input
            x = x + self.attn(
                attention_input, ve, cos_sin, window_size, kv_cache, kv_source=kv_input
            )
            return x + self.mlp(self.mlp_dyt(x))
        attn_input = norm(x) if attention_input is None else attention_input
        if self.variant == "canon_abcd":
            attn_input = self.canon_a(attn_input)
        attn_output = self.attn(attn_input, ve, cos_sin, window_size, kv_cache, kv_source=kv_input)
        if self.variant == "peri_ln":
            attn_output = norm(attn_output)
        x = x + attn_output
        mlp_input = norm(x)
        if self.variant == "canon_abcd":
            mlp_input = self.canon_c(mlp_input)
        mlp_output = self.mlp(mlp_input)
        if self.variant == "peri_ln":
            mlp_output = norm(mlp_output)
        return x + mlp_output


class SotaPoolGPT(GPT):
    def __init__(self, config: SotaPoolConfig, pad_vocab_size_to: int = 64):
        super().__init__(config, pad_vocab_size_to=pad_vocab_size_to)
        self.transformer.h = nn.ModuleList(
            [SotaBlock(config, layer_idx) for layer_idx in range(config.n_layer)]
        )
        # BoV needs the baseline W_V tensor to exist until init_weights() so its
        # aligned value table can be initialized from norm(wte) @ W_V.  Meta
        # reference models are never initialized, so accounting treats those
        # soon-to-be-removed tensors as dead while materialized models finalize
        # the replacement before training/checkpointing.
        self._bov_finalized = config.sota_variant != "bank_of_values"
        if config.sota_variant == "dynamic_tanh":
            self.input_dyt = DynamicTanh(config.n_embd)
            self.output_dyt = DynamicTanh(config.n_embd)

    @torch.no_grad()
    def init_weights(self):
        # Initializes every shared tensor in the same order and with the same
        # rule as the baseline model.
        GPT.init_weights(self)
        for block in self.transformer.h:
            block.reset_sota_parameters()
        if self.config.sota_variant == "dynamic_tanh":
            self.input_dyt.reset_parameters()
            self.output_dyt.reset_parameters()
        if self.config.sota_variant == "bank_of_values":
            self._initialize_and_finalize_bov()

    @torch.no_grad()
    def _initialize_and_finalize_bov(self) -> None:
        token_embedding = self.transformer.wte.weight
        target_layers = []
        for layer_idx, block in enumerate(self.transformer.h):
            if not block.attn.is_bov_target:
                continue
            target_layers.append(layer_idx)
            table = block.attn.value_table
            compute_dtype = self.config.runtime.compute_dtype
            table.to(dtype=compute_dtype)
            chunk = 2048
            for start in range(0, token_embedding.size(0), chunk):
                end = min(start + chunk, token_embedding.size(0))
                source = norm(token_embedding[start:end].to(compute_dtype))
                table.weight[start:end].copy_(block.attn.c_v(source))
            # BoV replaces W_V.  Initializing and then deleting it preserves the
            # baseline RNG stream for every shared parameter while removing the
            # dead matrix from the parameter/checkpoint contract.
            del block.attn.c_v
            if str(layer_idx) in self.value_embeds:
                del self.value_embeds[str(layer_idx)]
            block.attn.ve_gate = None
        if not target_layers:
            raise RuntimeError("BoV selected no target layers")
        self._bov_finalized = True

    def _residual_norm(self, x: torch.Tensor, *, output: bool = False) -> torch.Tensor:
        if self.config.sota_variant != "dynamic_tanh":
            return norm(x)
        return self.output_dyt(x) if output else self.input_dyt(x)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        _, time = idx.size()
        if time > self.cos.size(1):
            raise ValueError(f"sequence length {time} exceeds rotary cache {self.cos.size(1)}")
        if idx.device != self.cos.device:
            raise ValueError("rotary embeddings and tokens are on different devices")
        compute_dtype = self.config.runtime.compute_dtype
        if self.cos.dtype != compute_dtype:
            raise ValueError(f"rotary embeddings must use {compute_dtype}, got {self.cos.dtype}")
        offset = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, offset : offset + time], self.sin[:, offset : offset + time]

        x = self._residual_norm(self.transformer.wte(idx).to(compute_dtype))
        if kv_cache is None:
            if time <= 1:
                raise ValueError("training forward requires more than one token")
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                self.smear_gate(x[:, 1:, : self.smear_gate_channels])
            )
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            previous = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if time > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                    self.smear_gate(x[:, 1:, : self.smear_gate_channels])
                )
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif previous is not None:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                    self.smear_gate(x[:, :, : self.smear_gate_channels])
                )
                x = x + gate * previous

        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = None
        cached_attention_input = None
        midpoint_kv_input = None
        cache_start = self.config.n_layer - self.config.cached_attention_layers
        for layer_idx, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[layer_idx] * x + self.x0_lambdas[layer_idx] * x0
            if self.config.cached_attention_layers and layer_idx == cache_start:
                cached_attention_input = self._residual_norm(x)
            if block.attn.is_bov_target:
                ve = block.attn.value_table(idx).to(x.dtype)
            else:
                ve = (
                    self.value_embeds[str(layer_idx)](idx).to(x.dtype)
                    if str(layer_idx) in self.value_embeds
                    else None
                )
            attention_input = cached_attention_input if layer_idx >= cache_start else None
            kv_input = midpoint_kv_input if layer_idx > backout_layer else None
            x = block(
                x,
                ve,
                cos_sin,
                self.window_sizes[layer_idx],
                kv_cache,
                attention_input=attention_input,
                kv_input=kv_input,
            )
            if layer_idx == backout_layer:
                x_backout = x
                if self.config.reuse_midpoint_kv:
                    midpoint_kv_input = self._residual_norm(x)
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = self._residual_norm(x, output=True)

        raw_logits = self.lm_head(x)[..., : self.config.vocab_size]
        logits = raw_logits
        if self.config.loss_fp32:
            logits = logits.float()
        if self.config.logit_transform == "symmetric-softcap":
            logits = 15 * torch.tanh(logits / 15)
        elif self.config.logit_transform == "asymmetric":
            logits = 23 * torch.sigmoid((logits + 5) / 7.5)
        else:
            raise ValueError(f"unknown logit_transform={self.config.logit_transform!r}")
        if targets is None:
            return logits
        return language_model_loss(
            logits,
            raw_logits,
            targets,
            reduction=loss_reduction,
            z_loss_weight=self.config.z_loss_weight,
            training=self.training,
        )

    def _parameter_groups_for_accounting(self):
        value_table_ids = {
            id(module.weight) for module in self.modules() if isinstance(module, nn.Embedding)
        }
        dead_bov_ids = set()
        if not self._bov_finalized:
            for block in self.transformer.h:
                if not block.attn.is_bov_target:
                    continue
                dead_bov_ids.add(id(block.attn.c_v.weight))
                if block.attn.ve_gate is not None:
                    dead_bov_ids.add(id(block.attn.ve_gate.weight))
        dense_matrices, architecture = [], []
        for parameter in self.transformer.h.parameters():
            if id(parameter) in value_table_ids or id(parameter) in dead_bov_ids:
                continue
            if parameter.ndim == 2:
                dense_matrices.append(parameter)
            else:
                architecture.append(parameter)
        if self.config.sota_variant == "dynamic_tanh":
            architecture.extend(self.input_dyt.parameters())
            architecture.extend(self.output_dyt.parameters())
        return dense_matrices, architecture

    def num_scaling_params(self):
        wte = self.transformer.wte.weight.numel()
        if self._bov_finalized:
            value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        else:
            start_layer = _bov_start_layer(self.config)
            value_embeds = sum(
                module.weight.numel()
                for key, module in self.value_embeds.items()
                if int(key) < start_layer
            )
        bov_tables = sum(
            block.attn.value_table.weight.numel()
            for block in self.transformer.h
            if block.attn.is_bov_target
        )
        lm_head = self.lm_head.weight.numel()
        matrices, architecture_params = self._parameter_groups_for_accounting()
        transformer_matrices = sum(p.numel() for p in matrices)
        architecture = sum(p.numel() for p in architecture_params)
        scalars = (
            self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.smear_gate.weight.numel()
            + self.smear_lambda.numel()
            + self.backout_lambda.numel()
        )
        # smear_gate is a dense matrix outside the trunk but is intentionally
        # retained in nanochat's scalar/control accounting.
        total = (
            wte
            + value_embeds
            + bov_tables
            + lm_head
            + transformer_matrices
            + architecture
            + scalars
        )
        if self._bov_finalized:
            assert total == sum(p.numel() for p in self.parameters()), "parameter count mismatch"
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "bov_tables": bov_tables,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "architecture": architecture,
            "scalars": scalars,
            "total": total,
        }

    def estimate_flops(self):
        counts = self.num_scaling_params()
        dense = counts["transformer_matrices"] + counts["lm_head"]
        seq_len = self.config.sequence_len
        attention = 0
        for window_size in self.window_sizes:
            effective = min(window_size[0], seq_len) if window_size[0] >= 0 else seq_len
            attention += (
                12 * self.config.n_head * (self.config.n_embd // self.config.n_head) * effective
            )
        if self.config.sota_variant == "differential_attention":
            # QK work is unchanged (two half-width maps); AV is doubled.
            attention = int(attention * 1.5)
        architecture = 0
        if self.config.sota_variant == "canon_abcd":
            architecture = 6 * counts["architecture"]
        elif self.config.sota_variant == "exclusive_attention":
            architecture = 12 * self.config.n_layer * self.config.n_embd
        return 6 * dense + attention + architecture

    def estimate_executed_flops(self):
        return self.estimate_flops()

    def get_architecture_state(self) -> dict:
        state = {
            "family": "sota_pool",
            "variant": self.config.sota_variant,
            "extra_adamw_lr": self.config.sota_extra_lr,
            "shared_backbone": "nanochat_d14",
            "sources_pinned_in_campaign_manifest": True,
        }
        if self.config.sota_variant == "gated_attention":
            state.update(
                gate="query_dependent_headwise_sigmoid_after_sdpa", gate_layers=self.config.n_layer
            )
        elif self.config.sota_variant == "exclusive_attention":
            state.update(
                projection="attention_output_orthogonal_to_self_value", target_layers="all"
            )
        elif self.config.sota_variant == "differential_attention":
            state.update(
                attention_maps=2,
                qk_head_dim=self.config.n_embd // self.config.n_head // 2,
                target_layers="all",
            )
        elif self.config.sota_variant == "xielu":
            state.update(replaces="relu_squared", alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5)
        elif self.config.sota_variant == "dynamic_tanh":
            state.update(replaces="residual_stream_rmsnorms", alpha_init=1.0, qk_norm="retained")
        elif self.config.sota_variant == "peri_ln":
            state.update(post_sublayer_rmsnorm=True, pre_sublayer_rmsnorm="retained")
        elif self.config.sota_variant == "canon_abcd":
            state.update(
                canon_set="ABCD",
                kernel=self.config.canon_kernel_size,
                residual=True,
                activation=False,
            )
        elif self.config.sota_variant == "bank_of_values":
            state.update(
                target_layers=[
                    i for i, block in enumerate(self.transformer.h) if block.attn.is_bov_target
                ],
                standard_v="replaced",
                additive_value_embedding_in_target_layers="disabled",
                gamma_init=1.0,
                table_lr="embedding_lr*0.5",
            )
        return state
