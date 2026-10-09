"""DP-only native projection/norm/gate adapter for the controlled DSW pilots.

The complete baseline is constructed before swapping any modules. Existing
parameter objects and names are retained; extra K/V initialization is isolated
from the baseline RNG. No container runtime classes or functions are patched.
"""

from __future__ import annotations

import hashlib
from copy import copy, deepcopy

import torch
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule

from archlab.architectures.simplicial_attention import simplicial_attention

PILOT_LAYERS = (8, 16, 24, 32, 40, 48)
EXTRA_MARKERS = (".attention.k2.", ".attention.v2.", ".attention.k2_layernorm.")


def parameter_hashes(model, *, common_only=False):
    """Hash actual BF16/FP32 parameter bytes, not seeds or configuration labels."""
    result = {}
    for name, parameter in model.named_parameters():
        if common_only and any(marker in name for marker in EXTRA_MARKERS):
            continue
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
        result[name] = hashlib.sha256(raw.tobytes()).hexdigest()
    return result


class PilotAttention(MegatronModule):
    def __init__(self, source, arm, *, short_window=16, long_window=128, seed=42):
        super().__init__(config=source.config)
        if arm not in ("B", "C") or source.config.tensor_model_parallel_size != 1:
            raise ValueError("pilot attention supports only B/C with TP1")
        if not source.config.attention_output_gate or not source.config.qk_layernorm:
            raise ValueError("pilot must retain native output gate and Q/K normalization")
        self.arm = arm
        self.short_window, self.long_window = short_window, long_window
        self.layer_number = source.layer_number
        self.pg_collection = source.pg_collection
        self.linear_qkv = source.linear_qkv
        self.linear_proj = source.linear_proj
        self.q_layernorm = source.q_layernorm
        self.k_layernorm = source.k_layernorm
        if arm == "B":
            local_config = copy(source.config)
            local_config.window_size = (long_window - 1, 0)
            local_config.window_attn_skip_freq = None
            self.core_attention = TEDotProductAttention(
                config=local_config, layer_number=self.layer_number,
                attn_mask_type=AttnMaskType.causal, attention_type="self",
                pg_collection=self.pg_collection,
            )
        else:
            weight = source.linear_qkv.k.weight
            device = weight.device
            devices = [device.index] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed + 100_000 + self.layer_number)
                self.k2 = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                                          device=device, dtype=weight.dtype)
                self.v2 = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                                          device=device, dtype=weight.dtype)
                self.config.init_method(self.k2.weight)
                self.config.init_method(self.v2.weight)
                self.k2_layernorm = deepcopy(source.k_layernorm)

    def qkv_gate(self, hidden_states):
        mixed, _ = self.linear_qkv(hidden_states)
        groups = self.config.num_query_groups
        heads = self.config.num_attention_heads
        dim = self.config.kv_channels
        mixed = mixed.reshape(*mixed.shape[:2], groups, -1)
        width = heads // groups * dim
        q, gate, k, v = torch.split(mixed, [width, width, dim, dim], dim=-1)
        q = self.q_layernorm(q.reshape(*q.shape[:2], heads, dim))
        k = self.k_layernorm(k)
        return q, k, v, gate.reshape(*gate.shape[:2], heads, dim)

    def forward(self, hidden_states, attention_mask=None, rotary_pos_emb=None,
                rotary_pos_cos=None, rotary_pos_sin=None, inference_context=None,
                packed_seq_params=None, sequence_len_offset=None, **kwargs):
        if any(x is not None for x in (attention_mask, rotary_pos_cos, rotary_pos_sin,
                                       inference_context, packed_seq_params, sequence_len_offset)):
            raise ValueError("pilot supports unpadded causal full-prefix training/evaluation only")
        if rotary_pos_emb is None:
            raise ValueError("the approved RoPE pilot must not omit positional embeddings")
        q, k1, v1, gate = self.qkv_gate(hidden_states)
        qpos, kpos = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb,) * 2
        q = apply_rotary_pos_emb(q, qpos, config=self.config, cp_group=self.pg_collection.cp)
        k1 = apply_rotary_pos_emb(k1, kpos, config=self.config, cp_group=self.pg_collection.cp)
        if self.arm == "B":
            output = self.core_attention(q, k1, v1, None, attn_mask_type=AttnMaskType.causal)
        else:
            shape = (*hidden_states.shape[:2], self.config.num_query_groups, self.config.kv_channels)
            k2 = self.k2_layernorm(self.k2(hidden_states).reshape(shape))
            k2 = apply_rotary_pos_emb(k2, kpos, config=self.config, cp_group=self.pg_collection.cp)
            v2 = self.v2(hidden_states).reshape(shape)
            output = simplicial_attention(
                *(x.transpose(0, 1) for x in (q, k1, k2, v1, v2)),
                self.short_window, self.long_window,
            ).transpose(0, 1).contiguous().flatten(-2)
        # Native attention gates per channel BEFORE its output projection.
        output = (output * torch.sigmoid(gate.flatten(-2).float())).to(output.dtype)
        return self.linear_proj(output)


def install_pilot_attention(model, arm, *, seed=42, short_window=16, long_window=128):
    if arm not in ("A", "B", "C"):
        raise ValueError("unknown pilot arm")
    if arm == "A":
        return
    for layer in model.decoder.layers:
        if layer.layer_number in PILOT_LAYERS:
            layer.attention = PilotAttention(layer.attention, arm, seed=seed,
                                             short_window=short_window, long_window=long_window)
