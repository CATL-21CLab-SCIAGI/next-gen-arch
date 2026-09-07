"""Additive simplicial residual branch; never replaces pretrained attention.

This leaf accepts the four-stream state AFTER an existing attention residual
update and returns it with a learned addition, BEFORE the existing MoE read.
It does not load a checkpoint or claim support for any execution backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from archlab.architectures.qwen38_flash_next_full import (
    FourStreamGatedResidual,
    GroupRMSNorm,
    Qwen38FlashNextFullConfig,
)
from archlab.architectures.simplicial_attention import (
    reference_simplicial,
    simplicial_attention,
)


@dataclass(frozen=True)
class SimplicialAdapterConfig:
    hidden_size: int = 2560
    query_heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    residual_streams: int = 4
    residual_low_rank: int = 320
    short_window: int = 16
    long_window: int = 128
    rotary_fraction: float = 0.25
    rope_theta: float = 10_000_000.0
    rms_norm_eps: float = 1e-6
    initializer_std: float = 0.02
    output_initialization: Literal["zeros", "normal"] = "zeros"

    def __post_init__(self):
        if self.output_initialization not in ("zeros", "normal"):
            raise ValueError("output_initialization must be zeros or normal")
        sizes = (self.hidden_size, self.query_heads, self.kv_heads, self.head_dim,
                 self.residual_streams, self.residual_low_rank,
                 self.short_window, self.long_window)
        if any(not isinstance(value, int) or value < 1 for value in sizes):
            raise ValueError("adapter dimensions must be positive integers")
        if self.query_heads % self.kv_heads:
            raise ValueError("query heads must divide into KV groups")
        if self.query_heads // self.kv_heads > 128:
            raise ValueError("the kernel supports at most 128 query heads per KV group")
        if self.head_dim not in (16, 32, 64, 128, 256):
            raise ValueError("unsupported simplicial head dimension")
        if self.short_window > self.long_window:
            raise ValueError("short window must not exceed long window")
        rotary_dim = self.head_dim * self.rotary_fraction
        if not 0 < self.rotary_fraction <= 1 or int(rotary_dim) != rotary_dim or int(rotary_dim) % 2:
            raise ValueError("rotary fraction must select a positive even head subdimension")
        if any(not math.isfinite(value) or value <= 0
               for value in (self.rope_theta, self.rms_norm_eps, self.initializer_std)):
            raise ValueError("RoPE, normalization and initialization scales must be finite and positive")

    def parameter_count(self) -> int:
        width, packed = self.hidden_size, self.hidden_size * self.residual_streams
        q, kv = self.query_heads * self.head_dim, self.kv_heads * self.head_dim
        projections = width * (3 * q + 4 * kv)  # Q, output gate, O; K1/K2/V1/V2
        norms = 3 * self.head_dim
        residual = packed * (2 * self.residual_low_rank + self.residual_streams + 1)
        return projections + norms + residual


def partial_rope(values: torch.Tensor, positions: torch.Tensor, config: SimplicialAdapterConfig):
    """Ordinary partial RoPE, explicitly not relative-invariant trilinear RoPE."""
    dim = int(config.head_dim * config.rotary_fraction)
    inv_freq = config.rope_theta ** (-torch.arange(0, dim, 2, device=values.device).float() / dim)
    phase = positions.float().unsqueeze(-1) * inv_freq
    phase = torch.cat((phase, phase), dim=-1).unsqueeze(-2)
    rotating = values[..., :dim].float()
    first, second = rotating.chunk(2, dim=-1)
    rotated = rotating * phase.cos() + torch.cat((-second, first), dim=-1) * phase.sin()
    return torch.cat((rotated.to(values.dtype), values[..., dim:]), dim=-1)


class SimplicialResidualAdapter(nn.Module):
    """Independent branch with explicit zero or nonzero output initialization.

    Construction is CPU/meta-only; execution adapters may move the module to
    CUDA afterward. This keeps initialization from touching CUDA RNG state or
    allocating a GPU unexpectedly. The reference backend is a small-test oracle,
    not a fallback for production training. There is no FFN in the added branch.
    """

    def __init__(self, config: SimplicialAdapterConfig, *, seed: int = 42, backend: str = "triton"):
        super().__init__()
        if backend not in ("triton", "reference"):
            raise ValueError("choose the validated Triton path or the explicit test oracle")
        if torch.get_default_device().type not in ("cpu", "meta"):
            raise ValueError("construct adapters on CPU/meta, then move to the execution device")
        self.config = config
        self.backend = backend
        # No global seed change and no modification of existing model objects.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            residual_config = Qwen38FlashNextFullConfig(
                hidden_size=config.hidden_size, residual_streams=config.residual_streams,
                residual_low_rank=config.residual_low_rank, rms_norm_eps=config.rms_norm_eps,
                zero_centered_gamma=True,
            )
            self.residual = FourStreamGatedResidual(residual_config)
            q, kv = config.query_heads * config.head_dim, config.kv_heads * config.head_dim
            self.q = nn.Linear(config.hidden_size, q, bias=False)
            self.output_gate = nn.Linear(config.hidden_size, q, bias=False)
            self.k1 = nn.Linear(config.hidden_size, kv, bias=False)
            self.k2 = nn.Linear(config.hidden_size, kv, bias=False)
            self.v1 = nn.Linear(config.hidden_size, kv, bias=False)
            self.v2 = nn.Linear(config.hidden_size, kv, bias=False)
            self.output = nn.Linear(q, config.hidden_size, bias=False)
            self.q_norm = GroupRMSNorm(config.head_dim, 1, config.rms_norm_eps, zero_centered=True)
            self.k1_norm = GroupRMSNorm(config.head_dim, 1, config.rms_norm_eps, zero_centered=True)
            self.k2_norm = GroupRMSNorm(config.head_dim, 1, config.rms_norm_eps, zero_centered=True)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=config.initializer_std)
            # Both modes consume the same RNG draws and differ only in O.
            # Zero-centered RMSNorm offsets remain zero: their effective scale
            # is one, not a zero gate on the branch.
            if config.output_initialization == "zeros":
                nn.init.zeros_(self.output.weight)

    def forward(self, packed: torch.Tensor, *, position_ids: torch.Tensor | None = None):
        config = self.config
        if packed.ndim != 3 or packed.shape[-1] != config.hidden_size * config.residual_streams:
            raise ValueError("expected [batch, sequence, packed residual-stream width]")
        batch, length = packed.shape[:2]
        if batch < 1 or length < 1:
            raise ValueError("empty adapter input")
        if position_ids is None:
            position_ids = torch.arange(length, device=packed.device).expand(batch, -1)
        if position_ids.shape != (batch, length) or position_ids.device != packed.device:
            raise ValueError("positions must match the batch, sequence and execution device")
        mixed, residual, injection = self.residual(packed)
        qshape = (batch, length, config.query_heads, config.head_dim)
        kvshape = (batch, length, config.kv_heads, config.head_dim)
        q = partial_rope(self.q_norm(self.q(mixed).reshape(qshape)), position_ids, config)
        k1 = partial_rope(self.k1_norm(self.k1(mixed).reshape(kvshape)), position_ids, config)
        k2 = partial_rope(self.k2_norm(self.k2(mixed).reshape(kvshape)), position_ids, config)
        v1, v2 = self.v1(mixed).reshape(kvshape), self.v2(mixed).reshape(kvshape)
        core = simplicial_attention if self.backend == "triton" else reference_simplicial
        attended = core(q, k1, k2, v1, v2, config.short_window, config.long_window).flatten(-2)
        attended = (attended.float() * torch.sigmoid(self.output_gate(mixed).float())).to(attended.dtype)
        branch = self.output(attended)
        return FourStreamGatedResidual.inject(residual, branch, injection)
