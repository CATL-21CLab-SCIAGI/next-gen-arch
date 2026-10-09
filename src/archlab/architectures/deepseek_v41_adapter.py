"""The V4.1 proposal's additive branch, independent of the base-model backend.

This does not implement or replace V4.1's attention, mHC or Engram. The caller
must insert it after attention hc_post, retaining the original single-pass
coefficient flow. Each batch row is one contiguous document segment; packing
across documents is deliberately unsupported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from archlab.architectures.simplicial_attention import (
    reference_simplicial,
    simplicial_attention,
)


@dataclass(frozen=True)
class V41AdapterConfig:
    width: int = 5120
    streams: int = 4
    query_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 128
    short_window: int = 32
    long_window: int = 512
    norm_eps: float = 1e-6
    initializer_std: float = 0.02

    def __post_init__(self):
        dimensions = (
            self.width,
            self.streams,
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.short_window,
            self.long_window,
        )
        if any(type(x) is not int or x < 1 for x in dimensions):
            raise ValueError("adapter dimensions must be positive integers")
        if self.query_heads % self.kv_heads or self.query_heads // self.kv_heads > 128:
            raise ValueError("unsupported grouped-query geometry")
        if self.head_dim not in (16, 32, 64, 128, 256):
            raise ValueError("unsupported simplicial head dimension")
        if self.short_window > self.long_window:
            raise ValueError("short window must not exceed long window")
        if any(not math.isfinite(x) or x <= 0 for x in (self.norm_eps, self.initializer_std)):
            raise ValueError("normalization and initialization scales must be finite and positive")

    def parameter_count(self):
        q, kv = self.query_heads * self.head_dim, self.kv_heads * self.head_dim
        return self.width * (3 * q + 4 * kv) + self.width + 3 * self.head_dim + 2 * self.streams


class RMSNorm(nn.Module):
    """Ordinary unit-initialized RMSNorm, with FP32 normalization arithmetic."""

    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class V41SimplicialAdapter(nn.Module):
    def __init__(self, config: V41AdapterConfig, *, seed=42, backend="triton"):
        super().__init__()
        if backend not in ("triton", "reference", "deterministic"):
            raise ValueError(
                "choose Triton, deterministic Triton, or the explicit small-test reference"
            )
        if torch.get_default_device().type not in ("cpu", "meta"):
            raise ValueError("construct on CPU/meta, then move to the execution device")
        self.config, self.backend = config, backend
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            q, kv = config.query_heads * config.head_dim, config.kv_heads * config.head_dim
            self.read_logits = nn.Parameter(torch.zeros(config.streams))
            self.write_logits = nn.Parameter(torch.zeros(config.streams))
            self.input_norm = RMSNorm(config.width, config.norm_eps)
            self.q = nn.Linear(config.width, q, bias=False)
            self.k1 = nn.Linear(config.width, kv, bias=False)
            self.k2 = nn.Linear(config.width, kv, bias=False)
            self.v1 = nn.Linear(config.width, kv, bias=False)
            self.v2 = nn.Linear(config.width, kv, bias=False)
            self.output_gate = nn.Linear(config.width, q, bias=False)
            self.output = nn.Linear(q, config.width, bias=False)
            self.q_norm = RMSNorm(config.head_dim, config.norm_eps)
            self.k1_norm = RMSNorm(config.head_dim, config.norm_eps)
            self.k2_norm = RMSNorm(config.head_dim, config.norm_eps)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=config.initializer_std)
            nn.init.zeros_(self.output.weight)

    def forward(self, streams):
        # Keep FP32 optimizer/master parameters without autocasting the frozen
        # backbone's FP32 mHC/router arithmetic. Only this branch is autocast.
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=streams.is_cuda and streams.dtype == torch.bfloat16,
        ):
            return self._forward(streams)

    def _forward(self, streams):
        c = self.config
        if streams.ndim != 4 or streams.shape[-2:] != (c.streams, c.width):
            raise ValueError("expected [batch, sequence, residual_streams, width]")
        batch, length = streams.shape[:2]
        if batch < 1 or length < 1:
            raise ValueError("empty adapter input")
        read = self.read_logits.float().softmax(-1)
        x = self.input_norm((streams.float() * read[None, None, :, None]).sum(-2).to(streams.dtype))
        q_shape = (batch, length, c.query_heads, c.head_dim)
        kv_shape = (batch, length, c.kv_heads, c.head_dim)
        q = self.q_norm(self.q(x).reshape(q_shape))
        k1 = self.k1_norm(self.k1(x).reshape(kv_shape))
        k2 = self.k2_norm(self.k2(x).reshape(kv_shape))
        v1, v2 = self.v1(x).reshape(kv_shape), self.v2(x).reshape(kv_shape)
        if self.backend == "deterministic":
            from archlab.architectures.simplicial_deterministic import (
                deterministic_simplicial_attention,
            )

            core = deterministic_simplicial_attention
        else:
            core = simplicial_attention if self.backend == "triton" else reference_simplicial
        # Keep triple products and value products in FP32. BF16-intermediate
        # core arithmetic failed the production-window gradient oracle near
        # cancellation; projections and residual outputs remain BF16.
        with torch.autocast(streams.device.type, enabled=False):
            attended = (
                core(*(t.float() for t in (q, k1, k2, v1, v2)), c.short_window, c.long_window)
                .to(q.dtype)
                .flatten(-2)
            )
        attended = (attended.float() * self.output_gate(x).float().sigmoid()).to(attended.dtype)
        branch = self.output(attended)
        write = 2 * self.write_logits.float().sigmoid()
        return (streams.float() + branch.float().unsqueeze(-2) * write[None, None, :, None]).to(
            streams.dtype
        )
