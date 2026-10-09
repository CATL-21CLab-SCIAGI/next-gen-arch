"""Native TE projections packed for MCore gated attention, with separate Muon matrices.

The frozen MCore Muon splitter only supports ungated QKV. Keeping Q, gate, K,
and V as distinct parameters avoids that splitter without altering the runtime.
This adapter intentionally supports TP1 only. Attention and normalization remain
native SelfAttention/TENorm, including their backward and checkpoint machinery.
"""

from __future__ import annotations

import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.transformer.module import MegatronModule


class SplitGatedQKV(MegatronModule):
    def __init__(self, input_size, output_size, *, config, **kwargs):
        super().__init__(config=config)
        if config.tensor_model_parallel_size != 1 or not config.attention_output_gate:
            raise ValueError("split gated QKV requires gated attention and TP1")
        self.groups = config.num_query_groups
        q_width = config.num_attention_heads * config.kv_channels
        kv_width = self.groups * config.kv_channels
        if output_size != 2 * (q_width + kv_width):
            raise ValueError("native gated QKV output shape drift")
        if kwargs.get("bias") or kwargs.get("gather_output"):
            raise ValueError("split gated QKV requires bias-free local projections")
        self.q = TEColumnParallelLinear(input_size, q_width, config=config, **kwargs)
        self.gate = TEColumnParallelLinear(input_size, q_width, config=config, **kwargs)
        self.k = TEColumnParallelLinear(input_size, kv_width, config=config, **kwargs)
        self.v = TEColumnParallelLinear(input_size, kv_width, config=config, **kwargs)

    def forward(self, hidden_states):
        # MCore packs each KV group as [all Q heads, all gate heads, K, V].
        parts = [
            projection(hidden_states)[0].unflatten(-1, (self.groups, -1))
            for projection in (self.q, self.gate, self.k, self.v)
        ]
        return torch.cat(parts, dim=-1).flatten(-2), None
