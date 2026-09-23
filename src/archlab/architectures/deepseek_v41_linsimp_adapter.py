"""Geometry-matched linearized 2-simplicial and linear-attention adapters.

Both arms reuse the simplicial initialization stream and residual read/write
maps. The controlled factor is the mixer: one-mode random-feature linear
attention versus LinSimp (global RF state plus a short explicit anchor window).
Full-softmax local attention is intentionally not the baseline.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter
from archlab.architectures.linsimp_attention import (
    linear_rf_attention,
    linsimp_attention,
    orthogonal_feature_bank,
)


def linear_adapter_parameter_count(config: V41AdapterConfig) -> int:
    # Same tensors as the normal control, plus a learned temperature scalar.
    q, kv = config.query_heads * config.head_dim, config.kv_heads * config.head_dim
    return config.width * (3 * q + 2 * kv) + config.width + 2 * config.head_dim + 2 * config.streams + 1


def linsimp_adapter_parameter_count(config: V41AdapterConfig) -> int:
    return config.parameter_count() + 1


class _LinearFamilyAdapter(nn.Module):
    feature_rank_multiple = 4

    def _feature_rank(self):
        return self.config.head_dim * self.feature_rank_multiple

    def _install_feature_bank(self, seed):
        bank = orthogonal_feature_bank(
            self.config.query_heads,
            self.config.head_dim,
            self._feature_rank(),
            device="cpu",
            dtype=torch.float32,
            seed=seed + 7919,
        )
        self.register_buffer("feature_bank", bank, persistent=True)
        self.temperature = nn.Parameter(torch.tensor([math.sqrt(float(self.config.head_dim))]))


class V41LinearAttentionAdapter(_LinearFamilyAdapter):
    """One-mode positive-random-feature linear attention control."""

    def __init__(self, config: V41AdapterConfig, *, seed=42, backend="reference"):
        super().__init__()
        if backend not in ("reference", "deterministic"):
            raise ValueError("linear adapter uses the reference RF kernel")
        self.config, self.backend = config, backend
        source = V41SimplicialAdapter(config, seed=seed, backend="reference")
        self.read_logits, self.write_logits = source.read_logits, source.write_logits
        for name in ("input_norm", "q", "q_norm", "output_gate", "output"):
            setattr(self, name, getattr(source, name))
        self.k, self.v, self.k_norm = source.k2, source.v2, source.k2_norm
        self._install_feature_bank(seed)

    def forward(self, streams):
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
        q = self.q_norm(self.q(x).reshape(batch, length, c.query_heads, c.head_dim))
        k = self.k_norm(self.k(x).reshape(batch, length, c.kv_heads, c.head_dim))
        v = self.v(x).reshape(batch, length, c.kv_heads, c.head_dim)
        with torch.autocast(streams.device.type, enabled=False):
            attended = linear_rf_attention(
                q.float(),
                k.float(),
                v.float(),
                self.feature_bank,
                temperature=self.temperature.float().reshape(()),
            ).to(q.dtype)
        attended = attended.flatten(-2)
        attended = (attended.float() * self.output_gate(x).float().sigmoid()).to(attended.dtype)
        branch = self.output(attended)
        write = 2 * self.write_logits.float().sigmoid()
        return (streams.float() + branch.float().unsqueeze(-2) * write[None, None, :, None]).to(
            streams.dtype
        )


class V41LinSimpAdapter(_LinearFamilyAdapter):
    """LinSimp treatment: global RF key mode plus short explicit anchor mode."""

    def __init__(self, config: V41AdapterConfig, *, seed=42, backend="reference"):
        super().__init__()
        if backend not in ("reference", "deterministic"):
            raise ValueError("LinSimp adapter uses the reference RF kernel")
        self.config, self.backend = config, backend
        source = V41SimplicialAdapter(config, seed=seed, backend="reference")
        self.read_logits, self.write_logits = source.read_logits, source.write_logits
        for name in (
            "input_norm",
            "q",
            "k1",
            "k2",
            "v1",
            "v2",
            "output_gate",
            "output",
            "q_norm",
            "k1_norm",
            "k2_norm",
        ):
            setattr(self, name, getattr(source, name))
        self._install_feature_bank(seed)

    def forward(self, streams):
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
        # Map existing short/long axes onto LinSimp (k,r) / (v,u).
        k = self.k2_norm(self.k2(x).reshape(kv_shape))
        r = self.k1_norm(self.k1(x).reshape(kv_shape))
        v = self.v2(x).reshape(kv_shape)
        u = self.v1(x).reshape(kv_shape)
        with torch.autocast(streams.device.type, enabled=False):
            attended = linsimp_attention(
                q.float(),
                k.float(),
                r.float(),
                v.float(),
                u.float(),
                self.feature_bank,
                window=c.short_window,
                temperature=self.temperature.float().reshape(()),
            ).to(q.dtype)
        attended = attended.flatten(-2)
        attended = (attended.float() * self.output_gate(x).float().sigmoid()).to(attended.dtype)
        branch = self.output(attended)
        write = 2 * self.write_logits.float().sigmoid()
        return (streams.float() + branch.float().unsqueeze(-2) * write[None, None, :, None]).to(
            streams.dtype
        )
