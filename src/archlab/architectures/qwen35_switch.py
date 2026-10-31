"""Switch Qwen3.5 token mixers to higher-order mechanisms, inheriting the original layer's weights.

Two variants (user design, 2026-10-29):

- ``SimplicialSwitch`` replaces full attention with full causal determinant 2-simplicial
  attention (O(n^3)). It inherits gated Q, K, V, q/k norms and output projections. A second
  key/value stream is new. Scores sum signed determinants of 3-coordinate chunks, scaled by
  the teacher's head_dim**-0.5. All three score streams use the same SO(3) chunk rotations;
  the first two coordinates rotate and the third stays fixed. The teacher's unique partial
  RoPE frequencies are assigned to the leading chunks; remaining chunks are unrotated.
  Trailing head_dim % 3 score coordinates are excluded, while values retain every coordinate.
  This is not the old elementwise-trilinear form and has no softmax identity initialization.
- ``TriadicSwitch`` replaces a Gated DeltaNet layer with triadic GDN (E slices with a joint delta
  residual). It reuses the layer's projections, causal convolution, decay and write gates, gated
  norm and ``out_proj``. New: ``k2 = L2(softplus(W x))`` and ``q2 = L2(softplus(V x))`` over E
  slices, plus a per-slice decay offset. With W = V = 0 and zero offsets the slices are uniform and
  share one decay, which is exactly GDN.

Stage 1 trains the entire replacement mixer, including independent copies of inherited
parameters, on frozen teacher inputs and targets. The original teacher is never registered
inside a switch or modified. New branches start from N(0, std). Only unpadded, uncached full
sequences are supported; these modules do not implement serving or incremental decoding.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from archlab.architectures.simplicial_attention import (
    oriented_determinant_simplicial_attention,
)
from archlab.architectures.triadic_attention import triadic_gdn_attention


@dataclass(frozen=True)
class SwitchConfig:
    init_std: float = 0.02  # new projection weights; the bias of the second stream starts at 1
    feature_dim: int = 4  # triadic slices E (kernel supports 1, 2, 4, 8, 12, 16)
    backend: str = "kernel"  # "kernel" (GPU) or "reference" (CPU oracles)
    window_1: int | None = None  # causal window for inherited (k1, v1) stream; None = full sequence
    window_2: int | None = None  # causal window for auxiliary (k2, v2) stream; None = full sequence

    def __post_init__(self):
        if not math.isfinite(self.init_std) or self.init_std < 0:
            raise ValueError("init_std must be finite and nonnegative")
        if self.backend not in {"kernel", "reference"}:
            raise ValueError("backend must be kernel or reference")
        if self.feature_dim not in (1, 2, 4, 8, 12, 16):
            raise ValueError("unsupported triadic feature_dim")
        for name in ("window_1", "window_2"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be None or a positive integer")


def _chunk_rope(x, cos, sin):
    """SO(3) z-axis rotations using Qwen's unique (first-half) rotary frequencies.

    x is [B,L,H,D]; cos/sin are [B,L,1,R]. Applying the same position offset
    to all three score streams preserves every chunk determinant. This changes
    the coordinate pairing from the teacher's 2D RoPE and is part of the switch.
    """
    chunks = x.shape[-1] // 3
    if cos.shape != sin.shape or cos.shape[-1] % 2:
        raise ValueError("expected paired Qwen rotary frequencies")
    count = min(chunks, cos.shape[-1] // 2)
    leading = x[..., :3 * count].unflatten(-1, (count, 3))
    a, b, c = leading.unbind(-1)
    co, si = cos[..., :count].to(x.dtype), sin[..., :count].to(x.dtype)
    rotated = torch.stack((a * co - b * si, a * si + b * co, c), -1).flatten(-2)
    return torch.cat((rotated, x[..., 3 * count:]), dim=-1)


def _check_sequence(hidden_states, kwargs):
    if hidden_states.ndim != 3 or min(hidden_states.shape) < 1:
        raise ValueError("expected nonempty [batch, sequence, hidden] states")
    if any(kwargs.get(name) is not None for name in ("attention_mask", "past_key_values", "cache_params")):
        raise ValueError("switches require unpadded full sequences without a cache")


def _autocast(hidden_states):
    dev_type = hidden_states.device.type
    if torch.is_autocast_enabled(dev_type):
        return torch.autocast(dev_type, dtype=torch.get_autocast_dtype(dev_type), enabled=True)
    return torch.autocast(
        dev_type,
        dtype=hidden_states.dtype,
        enabled=hidden_states.dtype in (torch.bfloat16, torch.float16),
    )


class SimplicialSwitch(nn.Module):
    """Full causal 2-simplicial attention in place of a Qwen3.5 full-attention layer."""

    def __init__(self, teacher: nn.Module, config: SwitchConfig, *, seed: int):
        super().__init__()
        self.inherited = copy.deepcopy(teacher).requires_grad_(True)
        self.config = config
        heads_kv, dim = teacher.k_proj.out_features // teacher.head_dim, teacher.head_dim
        width = teacher.k_proj.in_features
        generator = torch.Generator().manual_seed(seed)
        self.k2 = nn.Linear(width, heads_kv * dim, bias=True)
        self.v2 = nn.Linear(width, heads_kv * dim, bias=True)
        for layer in (self.k2, self.v2):
            with torch.no_grad():
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * config.init_std)
                layer.bias.fill_(1.0)

    def forward(self, hidden_states, position_embeddings, **kwargs):
        _check_sequence(hidden_states, kwargs)
        with _autocast(hidden_states):
            return self._forward(hidden_states, position_embeddings)

    def _forward(self, hidden_states, position_embeddings):
        t = self.inherited
        batch, length = hidden_states.shape[:2]
        shape = (batch, length, -1, t.head_dim)
        query, gate = torch.chunk(t.q_proj(hidden_states).view(batch, length, -1, t.head_dim * 2), 2, dim=-1)
        query = t.q_norm(query)
        key = t.k_norm(t.k_proj(hidden_states).view(shape))
        value = t.v_proj(hidden_states).view(shape)
        cos, sin = (x.unsqueeze(2) for x in position_embeddings)  # [B, L, 1, rotary]
        x = hidden_states.to(self.k2.weight.dtype)
        key2 = self.k2(x).view(shape).to(key.dtype)
        value2 = self.v2(x).view(shape).to(value.dtype)
        query, key, key2 = (_chunk_rope(tensor, cos, sin) for tensor in (query, key, key2))
        operands = (query, key, key2, value, value2)
        if self.config.backend == "reference":
            operands = tuple(tensor.float() for tensor in operands)
        elif torch.is_autocast_enabled(hidden_states.device.type):
            amp_dtype = torch.get_autocast_dtype(hidden_states.device.type)
            operands = tuple(tensor.to(amp_dtype) for tensor in operands)
        w1 = length if self.config.window_1 is None else min(length, self.config.window_1)
        w2 = length if self.config.window_2 is None else min(length, self.config.window_2)
        attended = oriented_determinant_simplicial_attention(
            *operands, w1, w2, scale=t.head_dim**-0.5, backend=self.config.backend
        )
        attended = attended.reshape(batch, length, -1) * torch.sigmoid(gate.reshape(batch, length, -1))
        return t.o_proj(attended.to(hidden_states.dtype))


class TriadicSwitch(nn.Module):
    """Triadic GDN in place of a Qwen3.5 Gated DeltaNet layer."""

    def __init__(self, teacher: nn.Module, config: SwitchConfig, *, seed: int):
        super().__init__()
        self.inherited = copy.deepcopy(teacher).requires_grad_(True)
        self.config = config
        heads, width, features = teacher.num_v_heads, teacher.hidden_size, config.feature_dim
        generator = torch.Generator().manual_seed(seed)
        self.k2 = nn.Linear(width, heads * features, bias=False)
        self.q2 = nn.Linear(width, heads * features, bias=False)
        for layer in (self.k2, self.q2):
            with torch.no_grad():
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * config.init_std)
        self.decay_offset = nn.Parameter(torch.zeros(heads, features))  # per-slice log-scale on the shared decay

    def forward(self, hidden_states, **kwargs):
        _check_sequence(hidden_states, kwargs)
        with _autocast(hidden_states):
            return self._forward(hidden_states)

    def _forward(self, hidden_states):
        t = self.inherited
        batch, length = hidden_states.shape[:2]
        mixed = t.in_proj_qkv(hidden_states).transpose(1, 2)
        weight = t.conv1d.weight.squeeze(1)
        mixed = F.silu(F.conv1d(F.pad(mixed, (t.conv_kernel_size - 1, 0)), weight.unsqueeze(1),
                               bias=t.conv1d.bias, groups=mixed.shape[1]))
        query, key, value = torch.split(mixed.transpose(1, 2), [t.key_dim, t.key_dim, t.value_dim], dim=-1)
        query = query.reshape(batch, length, -1, t.head_k_dim)
        key = key.reshape(batch, length, -1, t.head_k_dim)
        value = value.reshape(batch, length, -1, t.head_v_dim)
        repeat = t.num_v_heads // t.num_k_heads
        if repeat > 1:
            query, key = query.repeat_interleave(repeat, dim=2), key.repeat_interleave(repeat, dim=2)
        # Match the teacher's q/k L2 norm, including its epsilon for near-zero vectors.
        query, key = query.float(), key.float()
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        beta = t.in_proj_b(hidden_states).float().sigmoid()
        decay = -t.A_log.float().exp() * F.softplus(t.in_proj_a(hidden_states).float() + t.dt_bias)  # [B, L, H]
        decay = decay.unsqueeze(-1) * self.decay_offset.exp()  # [B, L, H, E]
        slices = (batch, length, t.num_v_heads, self.config.feature_dim)
        x = hidden_states.to(self.k2.weight.dtype)
        key2 = F.normalize(F.softplus(self.k2(x).float()).view(slices), dim=-1)
        query2 = F.normalize(F.softplus(self.q2(x).float()).view(slices), dim=-1)
        backend = "official" if self.config.backend == "kernel" else "reference"
        dtype = torch.bfloat16 if backend == "official" else torch.float32
        core = triadic_gdn_attention(query.to(dtype), key.to(dtype), value.to(dtype), key2, query2, decay, beta,
                                     backend=backend)
        core = core[0] if isinstance(core, tuple) else core
        gate = t.in_proj_z(hidden_states).reshape(batch, length, -1, t.head_v_dim)
        normed = t.norm(core.reshape(-1, t.head_v_dim).to(hidden_states.dtype), gate.reshape(-1, t.head_v_dim))
        return t.out_proj(normed.reshape(batch, length, -1))


def mixer(layer: nn.Module) -> tuple[str, nn.Module]:
    """The token mixer of a Qwen3.5 decoder layer and its attribute name."""
    for name in ("self_attn", "linear_attn"):
        if hasattr(layer, name):
            return name, getattr(layer, name)
    raise TypeError("not a Qwen3.5 decoder layer")


def build_switches(layers, layer_types, variant: str, config: SwitchConfig, *, seed: int) -> dict[int, nn.Module]:
    """Switch modules keyed by layer index: full attention for "simplicial", GDN for "triadic"."""
    target = {"simplicial": ("full_attention", SimplicialSwitch), "triadic": ("linear_attention", TriadicSwitch)}
    if variant not in target or len(layers) != len(layer_types):
        raise ValueError("variant or decoder layer types do not match the model")
    kind, cls = target[variant]
    return {i: cls(mixer(layers[i])[1], config, seed=seed + i) for i, t in enumerate(layer_types) if t == kind}
