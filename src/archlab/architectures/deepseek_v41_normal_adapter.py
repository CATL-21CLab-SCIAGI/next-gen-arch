"""Geometry-matched 1-simplicial control for the V4.1 additive adapter.

The normal branch retains the baseline's long-axis K2/V2 initialization,
query/output/gate weights, RMS norms, and residual-stream read/write maps.
The short K1/V1 axis is absent. This is not a parameter-matched experiment.
"""

from __future__ import annotations

import torch
from torch import nn

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter
from archlab.architectures.local_attention import (
    deterministic_local_attention,
    reference_local_attention,
)


def normal_adapter_parameter_count(config: V41AdapterConfig) -> int:
    q, kv = config.query_heads * config.head_dim, config.kv_heads * config.head_dim
    return config.width * (3 * q + 2 * kv) + config.width + 2 * config.head_dim + 2 * config.streams


class V41NormalAttentionAdapter(nn.Module):
    def __init__(self, config: V41AdapterConfig, *, seed=42, backend="flash-attn-deterministic"):
        super().__init__()
        if backend not in ("flash-attn-deterministic", "reference"):
            raise ValueError("choose deterministic FlashAttention or the small-test reference")
        self.config, self.backend = config, backend
        # Reuse the exact baseline initialization stream, including the draws
        # for omitted projections. Shared tensors are therefore byte-identical.
        source = V41SimplicialAdapter(config, seed=seed, backend="reference")
        self.read_logits, self.write_logits = source.read_logits, source.write_logits
        for name in ("input_norm", "q", "q_norm", "output_gate", "output"):
            setattr(self, name, getattr(source, name))
        self.k, self.v, self.k_norm = source.k2, source.v2, source.k2_norm

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
            if self.backend == "reference":
                attended = reference_local_attention(
                    q.float(), k.float(), v.float(), c.long_window
                ).to(q.dtype)
            else:
                attended = deterministic_local_attention(q, k, v, c.long_window)
        attended = attended.flatten(-2)
        attended = (attended.float() * self.output_gate(x).float().sigmoid()).to(attended.dtype)
        branch = self.output(attended)
        write = 2 * self.write_logits.float().sigmoid()
        return (streams.float() + branch.float().unsqueeze(-2) * write[None, None, :, None]).to(
            streams.dtype
        )
