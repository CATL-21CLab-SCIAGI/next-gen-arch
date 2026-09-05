"""Full Qwen3.8-Flash-Next text mechanisms used by the Megatron adapter.

This is a from-scratch training variant, not a loader for Qwen's released
weights.  Only the mechanisms absent from Megatron Core live here: gated
delta attention, four-stream gated residuals, and owner-sharded PLE.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

SOURCE_MODEL = "Qwen/Qwen3.8-Flash-Next"
SOURCE_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"
SOURCE_CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
FULL_ARCH_FAMILY = "qwen38_flash_next_dense_ple"
QUARTER_DEPTH48_NO_MTP_ARCH_FAMILY = "qwen38_flash_next_dense_ple_quarter_depth48_no_mtp"
BILLION_DEPTH48_NO_MTP_ARCH_FAMILY = "qwen38_flash_next_dense_ple_1b_depth48_no_mtp"

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10_007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_hash_multipliers(
    unigram_vocab_size: int,
    ngram_size: int,
    *,
    ple_layer_index: int = 0,
    seed: int = 1_234,
) -> torch.Tensor:
    """Return the pinned Qwen SplitMix64 odd n-gram multipliers."""
    maximum = ((1 << 63) - 1) // max(unigram_vocab_size, 1)
    half_bound = max(1, maximum // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    values = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        values.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(values, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


@dataclass(frozen=True)
class Qwen38FlashNextFullConfig:
    vocab_size: int = 248_320
    sequence_len: int = 2_048
    max_position_embeddings: int = 262_144
    num_hidden_layers: int = 48
    hidden_size: int = 2_560
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25

    full_attention_interval: int = 4
    attention_heads: int = 24
    attention_kv_heads: int = 2
    attention_head_dim: int = 256

    linear_qk_heads: int = 16
    linear_v_heads: int = 48
    linear_key_dim: int = 128
    linear_value_dim: int = 128
    linear_conv_kernel: int = 4

    num_experts: int = 512
    num_experts_per_token: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    router_aux_loss_coefficient: float = 0.001
    router_z_loss_coefficient: float = 0.0

    residual_streams: int = 4
    residual_low_rank: int = 320

    ngram_size: int = 3
    ngram_heads_per_order: int = 8
    ngram_vocab_size_base: int = 20_000_000
    ngram_embedding_dim: int = 2_560
    ngram_partitions: int = 128
    ngram_layer: int = 1
    ngram_conv_kernel: int = 4
    ngram_hash_seed: int = 1_234
    eos_token_id: int = 248_044

    mtp_num_layers: int = 3
    mtp_use_repeated_layer: bool = True
    mtp_loss_scaling_factor: float = 0.1
    pipeline_layers: tuple[int, ...] = (12, 13, 13, 10)
    arch_family: str = FULL_ARCH_FAMILY

    def __post_init__(self) -> None:
        positive = (
            self.vocab_size,
            self.sequence_len,
            self.num_hidden_layers,
            self.hidden_size,
            self.full_attention_interval,
            self.attention_heads,
            self.attention_kv_heads,
            self.attention_head_dim,
            self.linear_qk_heads,
            self.linear_v_heads,
            self.linear_key_dim,
            self.linear_value_dim,
            self.num_experts,
            self.num_experts_per_token,
            self.moe_intermediate_size,
            self.shared_expert_intermediate_size,
            self.residual_streams,
            self.residual_low_rank,
            self.ngram_size,
            self.ngram_heads_per_order,
            self.ngram_vocab_size_base,
            self.ngram_embedding_dim,
            self.ngram_partitions,
        )
        if min(positive) < 1:
            raise ValueError("Qwen3.8-Flash-Next dimensions must be positive")
        if self.attention_heads % self.attention_kv_heads:
            raise ValueError("attention KV heads must divide query heads")
        if self.linear_v_heads % self.linear_qk_heads:
            raise ValueError("GDN QK heads must divide value heads")
        if self.mtp_num_layers < 0:
            raise ValueError("MTP layer count must be non-negative")
        if not 0 < self.num_experts_per_token <= self.num_experts:
            raise ValueError("invalid routed expert count")
        if self.ngram_embedding_dim % self.ngram_heads:
            raise ValueError("PLE embedding width must divide its hash heads")
        if sum(self.pipeline_layers) != self.num_hidden_layers:
            raise ValueError("pipeline layer layout must contain every backbone layer")
        if self.arch_family == FULL_ARCH_FAMILY:
            if self.residual_streams != 4:
                raise ValueError("the full-model contract requires four residual streams")
            if self.pipeline_layers != (12, 13, 13, 10):
                raise ValueError("the full-model PP4 layout is fixed at 12/13/13/10")
            if not self.mtp_use_repeated_layer or self.mtp_num_layers != 3:
                raise ValueError("the full-model variant repeats one MTP layer at three depths")
            if self.router_z_loss_coefficient != 0:
                raise ValueError("the full-model variant has no router z-loss")
        elif self.arch_family in (
            QUARTER_DEPTH48_NO_MTP_ARCH_FAMILY,
            BILLION_DEPTH48_NO_MTP_ARCH_FAMILY,
        ):
            expected = {
                "num_hidden_layers": 48,
                "hidden_size": 640,
                "attention_heads": 6,
                "attention_kv_heads": 1,
                "attention_head_dim": 64,
                "linear_qk_heads": 4,
                "linear_v_heads": 12,
                "linear_key_dim": 32,
                "linear_value_dim": 32,
                "num_experts": 128,
                "num_experts_per_token": 3,
                "moe_intermediate_size": 160,
                "shared_expert_intermediate_size": 160,
                "router_aux_loss_coefficient": 0.01,
                "router_z_loss_coefficient": 0.001,
                "residual_streams": 1,
                "residual_low_rank": 80,
                "ngram_heads_per_order": 2,
                "ngram_vocab_size_base": 5_000_000,
                "ngram_embedding_dim": 640,
                "ngram_partitions": 32,
                "mtp_num_layers": 0,
            }
            expected_pipeline = (12, 12, 12, 12)
            if self.arch_family == BILLION_DEPTH48_NO_MTP_ARCH_FAMILY:
                expected.update(
                    hidden_size=384,
                    num_experts=64,
                    moe_intermediate_size=112,
                    shared_expert_intermediate_size=112,
                    residual_low_rank=48,
                    ngram_vocab_size_base=1_000_000,
                    ngram_embedding_dim=384,
                )
                expected_pipeline = (48,)
            drift = {
                field: getattr(self, field)
                for field, value in expected.items()
                if getattr(self, field) != value
            }
            if drift:
                raise ValueError(f"{self.arch_family} contract drift: {drift}")
            if self.pipeline_layers != expected_pipeline:
                raise ValueError(f"{self.arch_family} pipeline layout must be {expected_pipeline}")
            if self.mtp_use_repeated_layer or self.mtp_loss_scaling_factor != 0:
                raise ValueError(f"{self.arch_family} must not construct or weight MTP")
        else:
            raise ValueError(f"unsupported Flash-Next architecture family: {self.arch_family}")

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.ngram_heads_per_order

    @property
    def ngram_branch_dim(self) -> int:
        return self.ngram_embedding_dim // self.ngram_heads

    @property
    def ngram_head_vocab_sizes(self) -> tuple[int, ...]:
        return tuple(
            find_nth_prime_after(self.ngram_vocab_size_base - 1, index + 1)
            for index in range(self.ngram_heads)
        )

    @property
    def ngram_total_rows(self) -> int:
        return sum(self.ngram_head_vocab_sizes)

    @property
    def ngram_padded_rows(self) -> int:
        return math.ceil(self.ngram_total_rows / self.ngram_partitions) * self.ngram_partitions

    @property
    def ngram_rows_per_partition(self) -> int:
        return self.ngram_padded_rows // self.ngram_partitions

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pipeline_layers"] = list(self.pipeline_layers)
        payload["ngram_head_vocab_sizes"] = list(self.ngram_head_vocab_sizes)
        payload["ngram_total_rows"] = self.ngram_total_rows
        payload["ngram_padded_rows"] = self.ngram_padded_rows
        return payload

    @classmethod
    def tiny(cls, **overrides: Any) -> Qwen38FlashNextFullConfig:
        config = cls(
            vocab_size=64,
            sequence_len=8,
            max_position_embeddings=32,
            num_hidden_layers=48,
            hidden_size=32,
            attention_heads=4,
            attention_kv_heads=1,
            attention_head_dim=8,
            linear_qk_heads=2,
            linear_v_heads=4,
            linear_key_dim=4,
            linear_value_dim=4,
            num_experts=8,
            num_experts_per_token=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            residual_low_rank=8,
            ngram_heads_per_order=2,
            ngram_vocab_size_base=31,
            ngram_embedding_dim=32,
            ngram_partitions=8,
            eos_token_id=63,
        )
        return replace(config, **overrides)

    @classmethod
    def billion_depth48_no_mtp(cls) -> Qwen38FlashNextFullConfig:
        """Approximately 1B total weights, with all 48 layers and node-local EP8."""
        return replace(
            cls.quarter_depth48_no_mtp(),
            hidden_size=384,
            num_experts=64,
            moe_intermediate_size=112,
            shared_expert_intermediate_size=112,
            residual_low_rank=48,
            ngram_vocab_size_base=1_000_000,
            ngram_embedding_dim=384,
            pipeline_layers=(48,),
            arch_family=BILLION_DEPTH48_NO_MTP_ARCH_FAMILY,
        )

    @classmethod
    def quarter_depth48_no_mtp(cls) -> Qwen38FlashNextFullConfig:
        """Quarter divisible shapes while retaining all 48 backbone layers."""
        return cls(
            num_hidden_layers=48,
            hidden_size=640,
            attention_heads=6,
            attention_kv_heads=1,
            attention_head_dim=64,
            linear_qk_heads=4,
            linear_v_heads=12,
            linear_key_dim=32,
            linear_value_dim=32,
            num_experts=128,
            num_experts_per_token=3,
            moe_intermediate_size=160,
            shared_expert_intermediate_size=160,
            router_aux_loss_coefficient=0.01,
            router_z_loss_coefficient=0.001,
            residual_streams=1,
            residual_low_rank=80,
            ngram_heads_per_order=2,
            ngram_vocab_size_base=5_000_000,
            ngram_embedding_dim=640,
            ngram_partitions=32,
            mtp_num_layers=0,
            mtp_use_repeated_layer=False,
            mtp_loss_scaling_factor=0.0,
            pipeline_layers=(12, 12, 12, 12),
            arch_family=QUARTER_DEPTH48_NO_MTP_ARCH_FAMILY,
        )


class GroupRMSNorm(nn.Module):
    """RMS-normalize each packed residual stream independently."""

    def __init__(self, hidden_size: int, streams: int, eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.streams = streams
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(streams * hidden_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.size(-1) != self.streams * self.hidden_size:
            raise ValueError("invalid packed residual width")
        grouped = inputs.unflatten(-1, (self.streams, self.hidden_size))
        weight = self.weight.unflatten(0, (self.streams, self.hidden_size)).to(inputs.dtype)
        variance = grouped.float().square().mean(dim=-1, keepdim=True)
        normalized = grouped * torch.rsqrt(variance + self.eps).to(grouped.dtype)
        return (normalized * weight).flatten(-2)


class FourStreamGatedResidual(nn.Module):
    """Official gated residual with the configured packed stream count."""

    def __init__(self, config: Qwen38FlashNextFullConfig, *, combine: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.streams = config.residual_streams
        packed_size = self.hidden_size * self.streams
        self.norm = GroupRMSNorm(self.hidden_size, self.streams, config.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(packed_size, config.residual_low_rank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.residual_low_rank, packed_size, bias=False)
        self.block_inject_weight = (
            nn.Linear(packed_size, self.streams, bias=False) if combine else None
        )
        for parameter in self.parameters():
            parameter.is_embedding_or_output_parameter = True
            parameter.archlab_optimizer = "adamw"

    def forward(
        self, packed: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.norm(packed)
        latent = F.silu(self.input_mix_weight_down(normalized) / self.streams)
        weights = torch.sigmoid(self.input_mix_weight_up(latent))
        mixed = (
            weights.unflatten(-1, (self.streams, self.hidden_size))
            * normalized.unflatten(-1, (self.streams, self.hidden_size))
        ).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed
        injection = 2 * torch.sigmoid(self.block_inject_weight(normalized) / self.streams)
        return mixed, packed, injection

    @staticmethod
    def inject(packed: torch.Tensor, branch: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return packed + (branch.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)


class CausalDepthwiseConv1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        zero_init: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # GDN uses the validated FLA causal convolution on CUDA. PLE's dilated
        # convolution is not exposed by FLA and stays in cuDNN/PyTorch.
        if inputs.device.type == "cuda" and self.dilation == 1:
            try:
                from fla.modules.convolution import causal_conv1d
            except ImportError as error:  # pragma: no cover - container contract
                raise RuntimeError("Qwen3.8 GDN requires the container FLA kernel") from error
            output, _state = causal_conv1d(
                x=inputs,
                weight=self.weight.to(inputs.dtype),
                activation="silu",
                backend="triton",
            )
            return output
        padding = (self.kernel_size - 1) * self.dilation
        channels_first = F.pad(inputs.transpose(1, 2), (padding, 0))
        output = F.conv1d(
            channels_first,
            self.weight.to(inputs.dtype).unsqueeze(1),
            groups=inputs.size(-1),
            dilation=self.dilation,
        )
        return F.silu(output.transpose(1, 2))


def gated_delta_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """FP32 recurrent oracle for tensors in [batch, sequence, heads, width]."""
    query = F.normalize(query.float(), dim=-1) / math.sqrt(query.size(-1))
    key = F.normalize(key.float(), dim=-1)
    value, decay, beta = value.float(), decay.float(), beta.float()
    state = query.new_zeros(query.size(0), query.size(2), query.size(3), value.size(3))
    outputs = []
    for token in range(query.size(1)):
        state = state * decay[:, token].exp().unsqueeze(-1).unsqueeze(-1)
        prediction = torch.einsum("bhk,bhkv->bhv", key[:, token], state)
        update = beta[:, token].unsqueeze(-1) * (value[:, token] - prediction)
        state = state + key[:, token].unsqueeze(-1) * update.unsqueeze(-2)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, query[:, token]))
    return torch.stack(outputs, dim=1).to(value.dtype)


class GatedDeltaNet(nn.Module):
    """Full-shape Qwen GDN with physically separate beta and decay gates."""

    def __init__(self, config: Qwen38FlashNextFullConfig):
        super().__init__()
        self.q_heads = config.linear_qk_heads
        self.v_heads = config.linear_v_heads
        self.key_dim = config.linear_key_dim
        self.value_dim = config.linear_value_dim
        q_width = self.q_heads * self.key_dim
        v_width = self.v_heads * self.value_dim
        self.in_proj_qkv = nn.Linear(config.hidden_size, 2 * q_width + v_width, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, v_width, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.v_heads, bias=False)
        self.conv1d = CausalDepthwiseConv1d(2 * q_width + v_width, config.linear_conv_kernel)
        # The pinned source initializes A log-uniformly over [0.01, 16] and
        # uses a unit time-step bias.
        self.A_log = nn.Parameter(
            torch.empty(self.v_heads, dtype=torch.float32).uniform_(0.01, 16).log_()
        )
        self.dt_bias = nn.Parameter(torch.ones(self.v_heads, dtype=torch.float32))
        self.norm = nn.RMSNorm(self.value_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(v_width, config.hidden_size, bias=False)
        for module in (self.in_proj_z, self.in_proj_b, self.in_proj_a):
            module.weight.is_embedding_or_output_parameter = True
            module.weight.archlab_optimizer = "adamw"
        for parameter in (self.conv1d.weight, self.A_log, self.dt_bias, self.norm.weight):
            parameter.is_embedding_or_output_parameter = True
            parameter.archlab_optimizer = "adamw"

    def _kernel(self, query, key, value, decay, beta):
        if query.device.type == "cuda":
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            except ImportError as error:  # pragma: no cover - container contract
                raise RuntimeError("Qwen3.8 GDN requires the container FLA kernel") from error
            output, _state = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=decay,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
            )
            return output
        return gated_delta_reference(query, key, value, decay, beta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Megatron tensors are [sequence, batch, hidden]; convolution kernels
        # use [batch, sequence, channels].
        inputs = hidden_states.transpose(0, 1)
        batch, sequence, _ = inputs.shape
        q_width = self.q_heads * self.key_dim
        v_width = self.v_heads * self.value_dim
        mixed = self.conv1d(self.in_proj_qkv(inputs))
        query, key, value = torch.split(mixed, (q_width, q_width, v_width), dim=-1)
        query = query.view(batch, sequence, self.q_heads, self.key_dim)
        key = key.view(batch, sequence, self.q_heads, self.key_dim)
        value = value.view(batch, sequence, self.v_heads, self.value_dim)
        repeats = self.v_heads // self.q_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
        beta = self.in_proj_b(inputs).sigmoid()
        decay = -self.A_log.float().exp().view(1, 1, -1) * F.softplus(
            self.in_proj_a(inputs).float() + self.dt_bias.view(1, 1, -1)
        )
        output = self._kernel(query, key, value, decay, beta)
        output = self.norm(output)
        gate = torch.sigmoid(
            self.in_proj_z(inputs).view(batch, sequence, self.v_heads, self.value_dim).float()
        ).to(output.dtype)
        return self.out_proj((output * gate).flatten(-2)).transpose(0, 1)


class PLEHash(nn.Module):
    """Exact Layer-2 PLE n-gram hashing without the embedding storage."""

    def __init__(self, config: Qwen38FlashNextFullConfig):
        super().__init__()
        self.ngram_size = config.ngram_size
        self.heads_per_order = config.ngram_heads_per_order
        self.eos_token_id = config.eos_token_id
        sizes = torch.tensor(config.ngram_head_vocab_sizes, dtype=torch.long)
        offsets = torch.tensor([0, *torch.cumsum(sizes, dim=0)[:-1].tolist()], dtype=torch.long)
        self.register_buffer("head_vocab_sizes", sizes, persistent=True)
        self.register_buffer("head_offsets", offsets, persistent=True)
        self.register_buffer(
            "multipliers",
            build_hash_multipliers(
                config.vocab_size,
                config.ngram_size,
                ple_layer_index=0,
                seed=config.ngram_hash_seed,
            ),
            persistent=True,
        )

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch, sequence = token_ids.shape
        positions = torch.arange(sequence, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_inclusive = torch.cummax(eos_positions, dim=1).values
        previous = torch.cat(
            (eos_positions.new_full((batch, 1), -1), previous_inclusive[:, :-1]), 1
        )
        position_in_segment = positions.unsqueeze(0) - (previous + 1)
        source = positions - shift
        shifted = token_ids.gather(1, source.clamp_min(0).expand(batch, -1))
        valid = (position_in_segment >= shift) & (source.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.long()
        shifted = [
            self._shift_right_ignore_eos(token_ids, shift) for shift in range(self.ngram_size)
        ]
        blocks = []
        for order in range(2, self.ngram_size + 1):
            start = (order - 2) * self.heads_per_order
            stop = start + self.heads_per_order
            mixed = shifted[0] * self.multipliers[0]
            for position in range(1, order):
                mixed = torch.bitwise_xor(mixed, shifted[position] * self.multipliers[position])
            sizes = self.head_vocab_sizes[start:stop]
            offsets = self.head_offsets[start:stop]
            blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes) + offsets)
        return torch.cat(blocks, dim=-1)


def ple_partition_ownership(partitions: int, owners: int) -> tuple[tuple[int, ...], ...]:
    if partitions < 1 or owners < 1 or partitions % owners:
        raise ValueError("PLE partitions must divide evenly across owners")
    return tuple(tuple(range(owner, partitions, owners)) for owner in range(owners))


class OwnerShardedPLEEmbedding(nn.Module):
    """Contiguous PLE partitions assigned round-robin to EP ranks."""

    def __init__(
        self,
        config: Qwen38FlashNextFullConfig,
        *,
        owner_rank: int,
        owner_world_size: int,
        process_group: dist.ProcessGroup | None = None,
        replica_rank: int = 0,
    ):
        super().__init__()
        if config.ngram_partitions % owner_world_size:
            raise ValueError("PLE partition count must be divisible by the EP world size")
        if not 0 <= owner_rank < owner_world_size:
            raise ValueError("invalid PLE owner rank")
        self.partitions = config.ngram_partitions
        self.rows_per_partition = config.ngram_rows_per_partition
        self.embedding_dim = config.ngram_branch_dim
        self.owner_rank = owner_rank
        self.owner_world_size = owner_world_size
        self.process_group = process_group
        if replica_rank < 0:
            raise ValueError("PLE replica rank must be non-negative")
        self.replica_rank = replica_rank
        self.global_partitions = ple_partition_ownership(self.partitions, owner_world_size)[
            owner_rank
        ]
        self.tables = nn.ParameterList()
        for _partition in self.global_partitions:
            # A flattened parameter deliberately selects native Adam and zero
            # weight decay in Megatron's standard optimizer grouping.
            parameter = nn.Parameter(torch.empty(self.rows_per_partition * self.embedding_dim))
            parameter.allreduce = False
            parameter.expert_parallel = True
            parameter.is_embedding_or_output_parameter = True
            parameter.archlab_optimizer = "adam"
            parameter.archlab_no_weight_decay = True
            self.tables.append(parameter)

    def reset_parameters(self, std: float = 0.02) -> None:
        for table in self.tables:
            nn.init.normal_(table, mean=0.0, std=std)

    def _local_lookup(self, encoded_ids: torch.Tensor) -> torch.Tensor:
        output = self.tables[0].new_empty((encoded_ids.numel(), self.embedding_dim))
        local_partition = torch.div(encoded_ids, self.rows_per_partition, rounding_mode="floor")
        local_row = torch.remainder(encoded_ids, self.rows_per_partition)
        for slot, table in enumerate(self.tables):
            selected = torch.nonzero(local_partition == slot, as_tuple=False).flatten()
            if selected.numel():
                values = F.embedding(
                    local_row.index_select(0, selected),
                    table.view(self.rows_per_partition, self.embedding_dim),
                )
                output.index_copy_(0, selected, values)
        return output

    def forward(self, global_ids: torch.Tensor) -> torch.Tensor:
        shape = global_ids.shape
        flat = global_ids.reshape(-1).long()
        partition = torch.div(flat, self.rows_per_partition, rounding_mode="floor")
        owner = torch.remainder(partition, self.owner_world_size)
        local_partition = torch.div(partition, self.owner_world_size, rounding_mode="floor")
        encoded = local_partition * self.rows_per_partition + torch.remainder(
            flat, self.rows_per_partition
        )
        if self.owner_world_size == 1:
            return self._local_lookup(encoded).view(*shape, self.embedding_dim)
        if not dist.is_initialized():
            raise RuntimeError("distributed PLE lookup requires an initialized process group")

        send_order = torch.argsort(owner, stable=True)
        send_ids = encoded.index_select(0, send_order).contiguous()
        send_counts = torch.bincount(owner, minlength=self.owner_world_size).to(
            device=flat.device, dtype=torch.int64
        )
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.process_group)
        recv_ids = torch.empty(int(recv_counts.sum().item()), dtype=torch.long, device=flat.device)
        dist.all_to_all_single(
            recv_ids,
            send_ids,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
            group=self.process_group,
        )
        recv_values = self._local_lookup(recv_ids)
        returned = recv_values.new_empty((send_ids.numel(), self.embedding_dim))
        from torch.distributed.nn.functional import all_to_all_single

        returned = all_to_all_single(
            returned,
            recv_values,
            output_split_sizes=send_counts.tolist(),
            input_split_sizes=recv_counts.tolist(),
            group=self.process_group,
        )
        output = returned.new_empty(returned.shape)
        output.index_copy_(0, send_order, returned)
        return output.view(*shape, self.embedding_dim)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Expose owner partitions once, with explicit expert-DP replica identity."""
        del metadata
        from megatron.core.dist_checkpointing.mapping import ShardedTensor

        state: dict[str, Any] = {}
        global_key = f"{prefix}tables.weight"
        for slot, (partition, parameter) in enumerate(
            zip(self.global_partitions, self.tables, strict=True)
        ):
            local_key = f"{prefix}tables.{slot}"
            state[local_key] = ShardedTensor.from_rank_offsets(
                global_key,
                parameter,
                *sharded_offsets,
                (len(sharded_offsets), partition, self.partitions),
                prepend_axis_num=len(sharded_offsets),
                replica_id=(0, 0, self.replica_rank),
            )
        return state


class DistributedPLE(nn.Module):
    """Full Layer-2 PLE projection and injection over packed GR streams."""

    def __init__(
        self,
        config: Qwen38FlashNextFullConfig,
        *,
        owner_rank: int,
        owner_world_size: int,
        process_group: dist.ProcessGroup | None = None,
        replica_rank: int = 0,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.streams = config.residual_streams
        packed_size = self.hidden_size * self.streams
        self.hash = PLEHash(config)
        self.embedding = OwnerShardedPLEEmbedding(
            config,
            owner_rank=owner_rank,
            owner_world_size=owner_world_size,
            process_group=process_group,
            replica_rank=replica_rank,
        )
        self.key_proj = nn.Linear(config.ngram_embedding_dim, packed_size, bias=False)
        self.value_proj = nn.Linear(config.ngram_embedding_dim, self.hidden_size, bias=False)
        self.norm_key = GroupRMSNorm(self.hidden_size, self.streams, config.rms_norm_eps)
        self.norm_query = GroupRMSNorm(self.hidden_size, self.streams, config.rms_norm_eps)
        self.norm_conv = GroupRMSNorm(self.hidden_size, self.streams, config.rms_norm_eps)
        self.conv = CausalDepthwiseConv1d(
            packed_size,
            config.ngram_conv_kernel,
            dilation=config.ngram_size,
            zero_init=True,
        )
        for parameter in (
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.conv.weight,
        ):
            parameter.is_embedding_or_output_parameter = True
            parameter.archlab_optimizer = "adamw"

    def forward(self, token_ids: torch.Tensor, packed_states: torch.Tensor) -> torch.Tensor:
        # token IDs arrive [batch, sequence], packed states [sequence, batch, 4H].
        embeddings = self.embedding(self.hash(token_ids)).flatten(-2).to(packed_states.dtype)
        packed_batch = packed_states.transpose(0, 1)
        key = self.norm_key(self.key_proj(embeddings)).unflatten(
            -1, (self.streams, self.hidden_size)
        )
        query = self.norm_query(packed_batch).unflatten(-1, (self.streams, self.hidden_size))
        value = self.value_proj(embeddings)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (gate.sigmoid() * value.unsqueeze(-2)).flatten(-2)
        output = gated + self.conv(self.norm_conv(gated))
        return output.transpose(0, 1)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Recurse explicitly so the owner-sharded table contract is retained."""
        from megatron.core.transformer.utils import sharded_state_dict_default

        state: dict[str, Any] = {}
        for name, module in self.named_children():
            state.update(
                sharded_state_dict_default(
                    module,
                    f"{prefix}{name}.",
                    sharded_offsets,
                    metadata,
                )
            )
        return state


def parameter_count_contract(
    config: Qwen38FlashNextFullConfig | None = None,
) -> dict[str, int]:
    """Closed-form physical parameter count for the supported native-MCore model."""
    config = config or Qwen38FlashNextFullConfig()
    h, streams, rank = config.hidden_size, config.residual_streams, config.residual_low_rank
    routed = config.num_experts * (
        h * 2 * config.moe_intermediate_size + config.moe_intermediate_size * h
    )
    shared = (
        h * 2 * config.shared_expert_intermediate_size
        + config.shared_expert_intermediate_size * h
        + h
    )
    moe = h * config.num_experts + routed + shared
    residual = streams * h + streams * h * rank + rank * streams * h + streams * h * streams
    final_mixer = streams * h + streams * h * rank + rank * streams * h
    attention = (
        h
        * (
            config.attention_heads * config.attention_head_dim
            + 2 * config.attention_kv_heads * config.attention_head_dim
        )
        + config.attention_heads * config.attention_head_dim * h
    )
    q_width = config.linear_qk_heads * config.linear_key_dim
    v_width = config.linear_v_heads * config.linear_value_dim
    gdn = (
        h * (2 * q_width + v_width)
        + h * v_width
        + 2 * h * config.linear_v_heads
        + (2 * q_width + v_width) * config.linear_conv_kernel
        + 2 * config.linear_v_heads
        + config.linear_value_dim
        + v_width * h
    )
    ple_tables = config.ngram_padded_rows * config.ngram_branch_dim
    packed = streams * h
    ple_projection = (
        config.ngram_embedding_dim * packed
        + config.ngram_embedding_dim * h
        + 3 * packed
        + packed * config.ngram_conv_kernel
    )
    dense_layers = config.num_hidden_layers // config.full_attention_interval
    gdn_layers = config.num_hidden_layers - dense_layers
    backbone = (
        dense_layers * attention
        + gdn_layers * gdn
        + config.num_hidden_layers * (moe + 2 * residual)
        + final_mixer
    )
    mtp_inner = attention + moe + 2 * residual + final_mixer
    # Native repeated-MTP owns two input norms, a 2H->H projection, and one final norm.
    mtp_wrapper = 2 * h + 2 * h * h + h
    embeddings_and_head = 2 * config.vocab_size * h
    if not config.mtp_num_layers:
        mtp_inner = 0
        mtp_wrapper = 0
    total = ple_tables + ple_projection + embeddings_and_head + backbone + mtp_inner + mtp_wrapper
    return {
        "embeddings_and_head": embeddings_and_head,
        "ple_tables": ple_tables,
        "ple_projection": ple_projection,
        "backbone": backbone,
        "shared_mtp_inner": mtp_inner,
        "native_mtp_wrapper": mtp_wrapper,
        "total": total,
    }
