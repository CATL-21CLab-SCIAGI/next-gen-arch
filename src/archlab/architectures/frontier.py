"""Controlled components from public frontier-model technical reports.

This module contains the mechanisms that can be isolated on the fixed nanochat
d14 dense backbone.  Report-scale MoE, multimodal, data, post-training, and
serving components remain in ``frontier_report_campaign/component_registry.json``
instead of being silently approximated here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from archlab.architectures.base import (
    GPT,
    GPTConfig,
    HeadSplitLinear,
    Linear,
    apply_rotary_emb,
    has_ve,
    init_projection_uniform_,
    language_model_loss,
    norm,
)

FRONTIER_VARIANTS = frozenset(
    {
        "inkling_relative_attention",
        "inkling_sconv_kv",
        "inkling_sconv_residual",
        "hybrid_swa_5_1_w512",
        "inkling_lr2_weight_decay",
        "partial_rope_25",
        "zero_centered_rmsnorm",
        "kimi_situ_glu",
        "shared_mtp3",
        "attention_sink",
        "per_head_muon",
        "qwen_gdn",
        "glm_simple_gdn",
        "deepseek_csa",
        "deepseek_hca",
        "glm_mla_muon_split",
        "motif_gdla",
        "motif_mhc_anneal",
    }
)


@dataclass
class FrontierPoolConfig(GPTConfig):
    arch_family: str = "frontier_pool"
    frontier_variant: str = "inkling_relative_attention"
    frontier_extra_lr: float = 0.005
    relative_dim: int = 16
    relative_extent: int = 1024
    sconv_kernel_size: int = 4
    mtp_depth: int = 3
    mtp_loss_weight: float = 0.1
    mhc_num_streams: int = 4
    mhc_sinkhorn_iterations: int = 20
    mhc_anneal_steps: int = 1907

    def __post_init__(self) -> None:
        if self.frontier_variant not in FRONTIER_VARIANTS:
            raise ValueError(f"unknown frontier_variant={self.frontier_variant!r}")
        if self.relative_dim <= 0 or self.relative_extent <= 0:
            raise ValueError("relative_dim and relative_extent must be positive")
        if self.sconv_kernel_size <= 0:
            raise ValueError("sconv_kernel_size must be positive")
        if self.mtp_depth <= 0 or self.mtp_loss_weight < 0:
            raise ValueError("invalid MTP configuration")
        if (
            self.mhc_num_streams <= 1
            or self.mhc_sinkhorn_iterations <= 0
            or self.mhc_anneal_steps <= 0
        ):
            raise ValueError("invalid modified mHC configuration")
        if self.frontier_variant == "partial_rope_25":
            rotary_dim = (self.n_embd // self.n_head) // 4
            if rotary_dim % 2:
                raise ValueError("25% partial RoPE requires an even rotary dimension")


class ZeroCenteredRMSNorm(nn.Module):
    """Qwen3-Next form: RMSNorm(x) * (1 + weight), with weight initialized at zero."""

    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return norm(x) * (1.0 + self.weight.to(x.dtype))


class MotifMHCConnection(nn.Module):
    """Four-stream mHC with Motif 3's post-map scale annealed 2 -> 1."""

    def __init__(self, config: FrontierPoolConfig):
        super().__init__()
        self.num_streams = config.mhc_num_streams
        self.hidden_size = config.n_embd
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations
        size = self.num_streams * (self.num_streams + 2)
        self.mapping_proj = nn.Linear(self.num_streams * self.hidden_size, size, bias=False)
        self.alpha = nn.Parameter(torch.empty(3))
        self.bias = nn.Parameter(torch.empty(size))
        # Keep the annealed scale as a tensor buffer.  A Python float attribute
        # becomes a Dynamo guard, so changing it once per optimizer step would
        # force a full graph recompile at every step.
        self.register_buffer("post_scale", torch.tensor(2.0, dtype=torch.float32))

    def set_post_scale(self, scale: float) -> None:
        self.post_scale.fill_(float(scale))

    def _sinkhorn(self, logits):
        matrix = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
        for _ in range(self.sinkhorn_iterations):
            matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(1e-6)
        return matrix

    def mappings(self, streams):
        n = self.num_streams
        flat = streams.flatten(-2).float()
        inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + 1e-6)
        raw = F.linear(flat, self.mapping_proj.weight.to(dtype=flat.dtype))
        scales = torch.cat(
            (
                self.alpha[0].expand(n),
                self.alpha[1].expand(n),
                self.alpha[2].expand(n * n),
            )
        )
        raw = raw * inv_rms * scales + self.bias
        pre = raw[..., :n].sigmoid()
        post = self.post_scale.to(raw.dtype) * raw[..., n : 2 * n].sigmoid()
        residual = self._sinkhorn(raw[..., 2 * n :].view(*streams.shape[:-2], n, n))
        return pre, post, residual

    def prepare(self, streams):
        pre, post, residual = self.mappings(streams)
        branch_input = (streams.float() * pre.unsqueeze(-1)).sum(dim=-2)
        return branch_input.to(streams.dtype), post, residual

    def combine(self, streams, branch_output, post, residual):
        mixed = torch.matmul(residual, streams.float())
        written = post.unsqueeze(-1) * branch_output.float().unsqueeze(-2)
        return (mixed + written).to(streams.dtype)


class InklingShortConvolution(nn.Module):
    """Inkling's FP32 residual causal depthwise convolution."""

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        # Store as a matrix so the existing Muon path treats it as a matrix.
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.float()
        y = F.conv1d(
            residual.transpose(1, 2),
            self.weight.float().unsqueeze(1),
            padding=self.kernel_size - 1,
            groups=self.channels,
        )[..., : x.size(1)]
        return (residual + y.transpose(1, 2)).to(x.dtype)


class CausalDepthwiseConv1d(nn.Module):
    """Training-only causal convolution matching Qwen3-Next's GDN front end."""

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cuda":
            try:
                from fla.modules.convolution import causal_conv1d
            except ImportError as exc:  # pragma: no cover - remote preflight
                raise RuntimeError("Qwen GDN requires the pinned fla-core CUDA kernel") from exc
            y, _ = causal_conv1d(
                x=x, weight=self.weight.to(x.dtype), activation="silu", backend="triton"
            )
            return y
        y = F.conv1d(
            x.transpose(1, 2),
            self.weight.to(x.dtype).unsqueeze(1),
            padding=self.kernel_size - 1,
            groups=x.size(-1),
        )[..., : x.size(1)]
        return F.silu(y.transpose(1, 2))


def gated_delta_reference(q, k, v, g, beta):
    """Small FP32 oracle for GDN/SimpleGDN unit tests."""
    q = F.normalize(q.float(), dim=-1)
    k = F.normalize(k.float(), dim=-1)
    v, g, beta = v.float(), g.float(), beta.float()
    state = q.new_zeros(q.size(0), q.size(2), q.size(3), v.size(3))
    outputs = []
    for token in range(q.size(1)):
        state = state * g[:, token].exp().unsqueeze(-1).unsqueeze(-1)
        prediction = torch.einsum("bhk,bhkv->bhv", k[:, token], state)
        update = beta[:, token].unsqueeze(-1) * (v[:, token] - prediction)
        state = state + k[:, token].unsqueeze(-1) * update.unsqueeze(-2)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, token]) * q.size(-1) ** -0.5)
    return torch.stack(outputs, dim=1).to(q.dtype)


class QwenGatedDeltaAttention(nn.Module):
    """Qwen3-Next GDN scaled to d14 while retaining 128-d key/value heads."""

    def __init__(self, config: FrontierPoolConfig, layer_idx: int, *, simple: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.simple = simple
        self.hidden_size = config.n_embd
        self.num_k_heads = config.n_head
        self.num_v_heads = config.n_head if simple else 2 * config.n_head
        # The controlled d14 contract yields the report's 128-d heads; tiny
        # tests scale this dimension with the backbone.
        self.head_k_dim = config.n_embd // config.n_head
        self.head_v_dim = self.head_k_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.ve_gate_channels = min(12, self.hidden_size)
        self.per_head_muon = config.per_head_muon
        if simple:
            # GLM-5 SimpleGDN directly reuses baseline-shaped Q/K/V projections.
            if self.per_head_muon:
                self.c_q = HeadSplitLinear(config.n_embd, self.num_k_heads, self.head_k_dim)
                self.c_k = HeadSplitLinear(config.n_embd, self.num_k_heads, self.head_k_dim)
                self.c_v = HeadSplitLinear(config.n_embd, self.num_v_heads, self.head_v_dim)
            else:
                self.c_q = Linear(config.n_embd, self.key_dim, bias=False)
                self.c_k = Linear(config.n_embd, self.key_dim, bias=False)
                self.c_v = Linear(config.n_embd, self.value_dim, bias=False)
        else:
            if self.per_head_muon:
                self.q_proj = HeadSplitLinear(config.n_embd, self.num_k_heads, self.head_k_dim)
                self.k_proj = HeadSplitLinear(config.n_embd, self.num_k_heads, self.head_k_dim)
                self.v_proj = HeadSplitLinear(config.n_embd, self.num_v_heads, self.head_v_dim)
                self.z_proj = HeadSplitLinear(config.n_embd, self.num_v_heads, self.head_v_dim)
            else:
                self.in_proj_qkvz = Linear(
                    config.n_embd, 2 * self.key_dim + 2 * self.value_dim, bias=False
                )
            self.in_proj_ba = Linear(config.n_embd, 2 * self.num_v_heads, bias=False)
            self.conv = CausalDepthwiseConv1d(
                2 * self.key_dim + self.value_dim, config.sconv_kernel_size
            )
            self.a_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
            self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.output_norm_weight = nn.Parameter(torch.empty(self.head_v_dim, dtype=torch.float32))
        self.out_proj = Linear(self.value_dim, config.n_embd, bias=False)
        self.ve_gate = (
            Linear(self.ve_gate_channels, self.num_v_heads, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def _kernel(self, q, k, v, g, beta):
        if q.device.type == "cuda":
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule, wy_fast
            except ImportError as exc:  # pragma: no cover - remote preflight
                raise RuntimeError("Qwen GDN requires fla-core==0.4.0") from exc
            # Triton's sm_103 pipeline rejects one of FLA's speculative
            # four-warp autotune candidates before it can benchmark the valid
            # candidate. Restrict only the failing WY-backward autotuner to its
            # deterministic two-warp/two-stage candidate. This leaves FLA's
            # operator and gradients intact and avoids modifying the pinned
            # wheel in-place.
            tuner = wy_fast.prepare_wy_repr_bwd_kernel.fn
            tuner.configs = [
                candidate
                for candidate in tuner.configs
                if candidate.num_warps == 2 and candidate.num_stages == 2
            ]
            output, _ = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
            )
            return output
        return gated_delta_reference(q, k, v, g, beta)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError("frontier GDN cache inference is outside this campaign")
        batch, time, _ = x.shape
        if self.simple:
            q, k, v = self.c_q(x), self.c_k(x), self.c_v(x)
            z = torch.zeros_like(v)
            beta = x.new_ones(batch, time, self.num_v_heads)
            g = x.new_zeros(batch, time, self.num_v_heads)
        else:
            if self.per_head_muon:
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                z = self.z_proj(x)
            else:
                mixed = self.in_proj_qkvz(x)
                q, k, v, z = torch.split(
                    mixed, [self.key_dim, self.key_dim, self.value_dim, self.value_dim], dim=-1
                )
            qkv = self.conv(torch.cat([q, k, v], dim=-1))
            q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            b, a = self.in_proj_ba(x).chunk(2, dim=-1)
            beta = torch.sigmoid(b)
            g = -self.a_log.float().exp().view(1, 1, -1) * F.softplus(
                a.float() + self.dt_bias.view(1, 1, -1)
            )
        q = q.view(batch, time, self.num_k_heads, self.head_k_dim)
        k = k.view(batch, time, self.num_k_heads, self.head_k_dim)
        v = v.view(batch, time, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads != self.num_k_heads:
            repeat = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)
        if ve is not None:
            ve = ve.view(batch, time, -1, self.head_v_dim)
            if ve.size(2) != self.num_v_heads:
                ve = ve.repeat_interleave(self.num_v_heads // ve.size(2), dim=2)
            ve_gate = 3 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + ve_gate.unsqueeze(-1) * ve
        output = self._kernel(q, k, v, g, beta)
        output = F.rms_norm(
            output,
            (self.head_v_dim,),
            self.output_norm_weight.to(output.dtype),
            eps=1e-6,
        )
        if not self.simple:
            output = output * F.silu(z.view(batch, time, self.num_v_heads, self.head_v_dim))
        return self.out_proj(output.reshape(batch, time, self.value_dim))


class DeepSeekCompressedAttention(nn.Module):
    """DeepSeek-V4 CSA/HCA compressor bundle adapted to the fixed d14 heads.

    At the campaign's 2K context, CSA top-k=512 retains every causally eligible
    compressed block. The separately reported DSA arm therefore owns the
    selector ablation, while this class isolates the learned compressor, shared
    K=V MQA, partial/inverse RoPE, and local-window branch.
    """

    def __init__(self, config: FrontierPoolConfig, layer_idx: int, *, heavy: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self.heavy = heavy
        self.compression = 128 if heavy else 4
        self.local_window = 128
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        if self.head_dim != 128:
            # Tiny tests scale the paper's fixed 64-d partial RoPE proportionally.
            self.rotary_dim = max(2, self.head_dim // 2)
            self.rotary_dim -= self.rotary_dim % 2
        else:
            self.rotary_dim = 64
        query_latent = config.n_embd // 2
        self.q_down = Linear(config.n_embd, query_latent, bias=False)
        self.q_up = Linear(query_latent, config.n_embd, bias=False)
        self.c_a = Linear(config.n_embd, self.head_dim, bias=False)
        self.z_a = Linear(config.n_embd, self.head_dim, bias=False)
        if not heavy:
            self.c_b = Linear(config.n_embd, self.head_dim, bias=False)
            self.z_b = Linear(config.n_embd, self.head_dim, bias=False)
        self.position_bias_a = nn.Parameter(torch.empty(self.compression, self.head_dim))
        if not heavy:
            self.position_bias_b = nn.Parameter(torch.empty(self.compression, self.head_dim))
        self.group_out = nn.ModuleList(
            [Linear(self.head_dim, self.head_dim, bias=False) for _ in range(self.n_head)]
        )
        self.out_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate = None

    def _pad_groups(self, tensor: torch.Tensor, fill: float = 0.0):
        pad = (-tensor.size(1)) % self.compression
        if pad:
            tensor = F.pad(tensor, (0, 0, 0, pad), value=fill)
        return tensor.view(tensor.size(0), -1, self.compression, tensor.size(-1))

    def _compress(self, x: torch.Tensor) -> torch.Tensor:
        c_a = self._pad_groups(self.c_a(x))
        z_a = self._pad_groups(self.z_a(x), fill=-torch.inf)
        logits_a = z_a.float() + self.position_bias_a.float().view(1, 1, self.compression, -1)
        if self.heavy:
            weights = F.softmax(logits_a, dim=2).to(c_a.dtype)
            return (weights * c_a).sum(dim=2)
        c_b = self._pad_groups(self.c_b(x))
        z_b = self._pad_groups(self.z_b(x), fill=-torch.inf)
        previous_c = torch.cat([torch.zeros_like(c_b[:, :1]), c_b[:, :-1]], dim=1)
        previous_z = torch.cat([torch.full_like(z_b[:, :1], -torch.inf), z_b[:, :-1]], dim=1)
        logits_b = previous_z.float() + self.position_bias_b.float().view(
            1, 1, self.compression, -1
        )
        logits = torch.cat([logits_a, logits_b], dim=2)
        weights = F.softmax(logits, dim=2).to(c_a.dtype)
        weight_a, weight_b = weights.split(self.compression, dim=2)
        return (weight_a * c_a).sum(dim=2) + (weight_b * previous_c).sum(dim=2)

    def _partial_rope(self, x, cos, sin, *, inverse: bool = False):
        rotary = self.rotary_dim
        rotated = apply_rotary_emb(
            x[..., -rotary:],
            cos[..., : rotary // 2],
            -sin[..., : rotary // 2] if inverse else sin[..., : rotary // 2],
        )
        return torch.cat([x[..., :-rotary], rotated], dim=-1)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError(
                "compressed-attention cache inference is outside this campaign"
            )
        batch, time, _ = x.shape
        cos, sin = cos_sin
        q = self.q_up(self.q_down(x)).view(batch, time, self.n_head, self.head_dim)
        q = norm(q)
        q = self._partial_rope(q, cos, sin)

        compressed = norm(self._compress(x))
        groups = compressed.size(1)
        block_positions = torch.arange(groups, device=x.device) * self.compression
        block_positions = (block_positions + self.compression - 1).clamp_max(time - 1)
        comp_cos, comp_sin = cos[:, block_positions], sin[:, block_positions]
        compressed = self._partial_rope(compressed.unsqueeze(2), comp_cos, comp_sin).squeeze(2)

        local = norm(self.c_a(x))
        if ve is not None:
            # Preserve nanochat's value-embedding path in the controlled
            # backbone. CSA/HCA share one K=V stream, so reduce the seven
            # baseline value heads into that shared local value stream.
            shared_ve = ve.view(batch, time, -1, self.head_dim).mean(dim=2)
            local = local + shared_ve
        local = self._partial_rope(local.unsqueeze(2), cos, sin).squeeze(2)
        key_value = torch.cat([compressed, local], dim=1)

        token_pos = torch.arange(time, device=x.device)
        group_pos = torch.arange(groups, device=x.device)
        compressed_valid = group_pos.unsqueeze(0) < (token_pos // self.compression).unsqueeze(1)
        local_distance = token_pos.unsqueeze(1) - token_pos.unsqueeze(0)
        local_valid = (local_distance >= 0) & (local_distance < self.local_window)
        allowed = torch.cat([compressed_valid, local_valid], dim=1)

        q_t = q.transpose(1, 2)
        kv_t = key_value.unsqueeze(1).expand(-1, self.n_head, -1, -1)
        output = F.scaled_dot_product_attention(
            q_t,
            kv_t,
            kv_t,
            attn_mask=allowed.view(1, 1, time, groups + time),
            dropout_p=0.0,
        ).transpose(1, 2)
        output = self._partial_rope(output, cos, sin, inverse=True)
        output = torch.cat(
            [projection(output[:, :, head]) for head, projection in enumerate(self.group_out)],
            dim=-1,
        )
        return self.out_proj(output)


class GLMMultiLatentAttention(nn.Module):
    """GLM-5 MLA control with width-capped latent and per-head Muon matrices."""

    def __init__(self, config: FrontierPoolConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.runtime = config.runtime
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.latent_dim = min(256, config.n_embd)
        self.q_down = Linear(config.n_embd, self.latent_dim, bias=False)
        self.kv_down = Linear(config.n_embd, self.latent_dim, bias=False)
        self.q_up = HeadSplitLinear(self.latent_dim, self.n_head, self.head_dim)
        self.k_up = HeadSplitLinear(self.latent_dim, self.n_head, self.head_dim)
        self.v_up = HeadSplitLinear(self.latent_dim, self.n_head, self.head_dim)
        self.out_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate_channels = min(12, config.n_embd)
        self.ve_gate = (
            Linear(self.ve_gate_channels, self.n_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError("MLA cache inference is outside this pretraining control")
        batch, time, _ = x.shape
        q_latent = self.q_down(x)
        kv_latent = self.kv_down(x)
        q = self.q_up(q_latent).view(batch, time, self.n_head, self.head_dim)
        k = self.k_up(kv_latent).view(batch, time, self.n_head, self.head_dim)
        v = self.v_up(kv_latent).view(batch, time, self.n_head, self.head_dim)
        if ve is not None:
            ve = ve.view_as(v)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        output = self.runtime.attention.flash_attn_func(
            q, k, v, causal=True, window_size=window_size
        )
        return self.out_proj(output.reshape(batch, time, -1))


class MotifGroupedDifferentialLatentAttention(nn.Module):
    """Motif 3 GDLA adapted from 64:16 to six signal and one noise head."""

    def __init__(self, config: FrontierPoolConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.runtime = config.runtime
        self.signal_heads = config.n_head - 1
        self.noise_heads = 1
        self.group_ratio = self.signal_heads
        self.head_dim = config.n_embd // config.n_head
        self.rotary_dim = self.head_dim // 2
        self.content_dim = self.head_dim - self.rotary_dim
        self.latent_dim = 256
        self.q_down = Linear(config.n_embd, self.latent_dim, bias=False)
        self.q_up = HeadSplitLinear(self.latent_dim, config.n_head, self.head_dim)
        self.kv_down = Linear(config.n_embd, self.latent_dim + self.rotary_dim, bias=False)
        self.kv_up = Linear(self.latent_dim, self.content_dim + self.head_dim, bias=False)
        self.lambda_proj = Linear(config.n_embd, self.signal_heads, bias=False)
        self.output_gate = Linear(self.latent_dim, self.signal_heads * self.head_dim, bias=False)
        self.out_proj = Linear(self.signal_heads * self.head_dim, config.n_embd, bias=False)
        self.ve_gate = None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError("GDLA cache inference is outside this campaign")
        batch, time, _ = x.shape
        query_latent = norm(self.q_down(x))
        q = self.q_up(query_latent).view(
            batch, time, self.signal_heads + self.noise_heads, self.head_dim
        )
        kv_and_rope = self.kv_down(x)
        kv_latent, rotary_key = torch.split(kv_and_rope, [self.latent_dim, self.rotary_dim], dim=-1)
        content_key, value = torch.split(
            self.kv_up(norm(kv_latent)), [self.content_dim, self.head_dim], dim=-1
        )
        cos, sin = cos_sin
        q_content, q_rope = q[..., : self.content_dim], q[..., self.content_dim :]
        q_rope = apply_rotary_emb(
            q_rope, cos[..., : self.rotary_dim // 2], sin[..., : self.rotary_dim // 2]
        )
        rotary_key = apply_rotary_emb(
            rotary_key.view(batch, time, 1, self.rotary_dim),
            cos[..., : self.rotary_dim // 2],
            sin[..., : self.rotary_dim // 2],
        )
        key = torch.cat([content_key.view(batch, time, 1, self.content_dim), rotary_key], dim=-1)
        value = value.view(batch, time, 1, self.head_dim)
        if ve is not None:
            value = value + ve.view(batch, time, -1, self.head_dim).mean(dim=2, keepdim=True)
        q = norm(torch.cat([q_content, q_rope], dim=-1)) * 1.2
        key = norm(key) * 1.2
        signal_q, noise_q = q[:, :, : self.signal_heads], q[:, :, self.signal_heads :]
        signal = self.runtime.attention.flash_attn_func(
            signal_q,
            key.expand(-1, -1, self.signal_heads, -1),
            value.expand(-1, -1, self.signal_heads, -1),
            causal=True,
            window_size=window_size,
        )
        noise = self.runtime.attention.flash_attn_func(
            noise_q, key, value, causal=True, window_size=window_size
        ).expand(-1, -1, self.signal_heads, -1)
        coefficient = torch.sigmoid(self.lambda_proj(x)).unsqueeze(-1)
        differential = signal - coefficient * noise
        gate = torch.sigmoid(self.output_gate(query_latent)).view_as(differential)
        return self.out_proj((gate * differential).reshape(batch, time, -1))


class SiTUGLU(nn.Module):
    """Kimi K3 Sigmoid Tanh Unit GLU with beta1=4 and beta2=25."""

    def __init__(self, config: FrontierPoolConfig):
        super().__init__()
        # Three GLU matrices approximately match the baseline's two 4d matrices.
        self.intermediate_size = round(8 * config.n_embd / 3)
        self.gate_proj = Linear(config.n_embd, self.intermediate_size, bias=False)
        self.up_proj = Linear(config.n_embd, self.intermediate_size, bias=False)
        self.down_proj = Linear(self.intermediate_size, config.n_embd, bias=False)
        self.beta1 = 4.0
        self.beta2 = 25.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = self.beta1 * torch.tanh(gate / self.beta1) * torch.sigmoid(gate)
        up = self.beta2 * torch.tanh(up / self.beta2)
        return self.down_proj(gate * up)


class FrontierMLP(nn.Module):
    def __init__(self, config: FrontierPoolConfig):
        super().__init__()
        self.variant = config.frontier_variant
        if self.variant == "kimi_situ_glu":
            self.situ = SiTUGLU(config)
        else:
            self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
            self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "kimi_situ_glu":
            return self.situ(x)
        return self.c_proj(F.relu(self.c_fc(x)).square())


class FrontierAttention(nn.Module):
    def __init__(self, config: FrontierPoolConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.runtime = config.runtime
        self.variant = config.frontier_variant
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        if self.n_kv_head != self.n_head:
            raise ValueError("frontier pool controls require the baseline MHA head layout")
        projection_cls = (
            HeadSplitLinear if config.per_head_muon or self.variant == "per_head_muon" else Linear
        )
        if projection_cls is HeadSplitLinear:
            self.c_q = projection_cls(self.n_embd, self.n_head, self.head_dim)
            self.c_k = projection_cls(self.n_embd, self.n_head, self.head_dim)
            self.c_v = projection_cls(self.n_embd, self.n_head, self.head_dim)
        else:
            self.c_q = projection_cls(self.n_embd, self.n_embd, bias=False)
            self.c_k = projection_cls(self.n_embd, self.n_embd, bias=False)
            self.c_v = projection_cls(self.n_embd, self.n_embd, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = min(12, self.n_embd)
        self.ve_gate = (
            Linear(self.ve_gate_channels, self.n_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        if self.variant == "inkling_sconv_kv":
            self.k_sconv = InklingShortConvolution(self.n_embd, config.sconv_kernel_size)
            self.v_sconv = InklingShortConvolution(self.n_embd, config.sconv_kernel_size)
        if self.variant == "inkling_relative_attention":
            self.r_proj = Linear(self.n_embd, self.n_head * config.relative_dim, bias=False)
            self.relative_bank = nn.Parameter(
                torch.empty(config.relative_dim, config.relative_extent)
            )
        if self.variant == "attention_sink":
            self.sink_logit = nn.Parameter(torch.empty(self.n_head))

    def _project(self, x: torch.Tensor, ve: torch.Tensor | None):
        batch, time, _ = x.shape
        q = self.c_q(x).view(batch, time, self.n_head, self.head_dim)
        k_flat = self.c_k(x)
        v_flat = self.c_v(x)
        if self.variant == "inkling_sconv_kv":
            k_flat = self.k_sconv(k_flat)
            v_flat = self.v_sconv(v_flat)
        k = k_flat.view(batch, time, self.n_head, self.head_dim)
        v = v_flat.view(batch, time, self.n_head, self.head_dim)
        if ve is not None:
            ve = ve.view(batch, time, self.n_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        return q, k, v

    def _relative_attention(self, x, q, k, v):
        batch, time = x.shape[:2]
        q, k = norm(q) * 1.2, norm(k) * 1.2
        scores = torch.einsum("bthd,bshd->bhts", q, k) * self.head_dim**-0.5
        rel = self.r_proj(x).view(batch, time, self.n_head, self.config.relative_dim)
        rel_logits = torch.einsum("bthd,dr->bhtr", rel, self.relative_bank.to(rel.dtype))
        positions = torch.arange(time, device=x.device)
        distance = positions[:, None] - positions[None, :]
        gather = distance.clamp(0, self.config.relative_extent - 1)
        gather = gather.view(1, 1, time, time).expand(batch, self.n_head, -1, -1)
        bias = rel_logits.gather(-1, gather)
        valid = (distance >= 0) & (distance < self.config.relative_extent)
        bias = bias.masked_fill(~valid.view(1, 1, time, time), 0.0)
        causal = distance >= 0
        scores = (scores + bias).masked_fill(~causal.view(1, 1, time, time), -torch.inf)
        probs = F.softmax(scores.float(), dim=-1).to(v.dtype)
        return torch.einsum("bhts,bshd->bthd", probs, v)

    def _sink_attention(self, q, k, v, window_size):
        batch, time = q.shape[:2]
        q, k = norm(q) * 1.2, norm(k) * 1.2
        positions = torch.arange(time, device=q.device)
        distance = positions[:, None] - positions[None, :]
        valid = distance >= 0
        if window_size[0] >= 0:
            valid = valid & (distance <= window_size[0])
        # A zero-valued extra KV entry plus a per-head additive score realizes
        # the learned softmax-denominator sink without materializing attention
        # probabilities in Python.
        zero_kv = k.new_zeros(batch, 1, self.n_head, self.head_dim)
        k_plus = torch.cat([k, zero_kv], dim=1).transpose(1, 2)
        v_plus = torch.cat([v, zero_kv], dim=1).transpose(1, 2)
        bias = q.new_full((1, self.n_head, time, time + 1), -torch.inf)
        bias[..., :time] = torch.where(
            valid.view(1, 1, time, time),
            q.new_zeros(()),
            q.new_full((), -torch.inf),
        )
        bias[..., -1] = self.sink_logit.to(q.dtype).view(1, self.n_head, 1)
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2), k_plus, v_plus, attn_mask=bias, dropout_p=0.0
        )
        return output.transpose(1, 2)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None and self.variant in {
            "inkling_relative_attention",
            "attention_sink",
        }:
            raise NotImplementedError(
                f"{self.variant} cache inference is outside this training campaign"
            )
        batch, time, _ = x.shape
        q, k, v = self._project(x, ve)
        if self.variant == "inkling_relative_attention":
            y = self._relative_attention(x, q, k, v)
        else:
            cos, sin = cos_sin
            if self.variant == "partial_rope_25":
                rotary_dim = self.head_dim // 4
                q = torch.cat(
                    [
                        apply_rotary_emb(
                            q[..., :rotary_dim],
                            cos[..., : rotary_dim // 2],
                            sin[..., : rotary_dim // 2],
                        ),
                        q[..., rotary_dim:],
                    ],
                    dim=-1,
                )
                k = torch.cat(
                    [
                        apply_rotary_emb(
                            k[..., :rotary_dim],
                            cos[..., : rotary_dim // 2],
                            sin[..., : rotary_dim // 2],
                        ),
                        k[..., rotary_dim:],
                    ],
                    dim=-1,
                )
            else:
                q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            if self.variant == "attention_sink":
                y = self._sink_attention(q, k, v, window_size)
            else:
                q, k = norm(q) * 1.2, norm(k) * 1.2
                if kv_cache is None:
                    y = self.runtime.attention.flash_attn_func(
                        q, k, v, causal=True, window_size=window_size
                    )
                else:
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
                        kv_cache.advance(time)
        return self.c_proj(y.contiguous().view(batch, time, self.n_embd))


class FrontierBlock(nn.Module):
    def __init__(self, config: FrontierPoolConfig, layer_idx: int):
        super().__init__()
        self.variant = config.frontier_variant
        use_qwen_gdn = self.variant == "qwen_gdn" and layer_idx % 4 != 3
        use_simple_gdn = self.variant == "glm_simple_gdn" and layer_idx % 2 == 0
        use_csa = self.variant == "deepseek_csa"
        use_hca = self.variant == "deepseek_hca"
        use_mla = self.variant == "glm_mla_muon_split"
        use_gdla = self.variant == "motif_gdla"
        if use_gdla:
            self.attn = MotifGroupedDifferentialLatentAttention(config, layer_idx)
        elif use_mla:
            self.attn = GLMMultiLatentAttention(config, layer_idx)
        elif use_csa or use_hca:
            self.attn = DeepSeekCompressedAttention(config, layer_idx, heavy=use_hca)
        elif use_qwen_gdn or use_simple_gdn:
            self.attn = QwenGatedDeltaAttention(config, layer_idx, simple=use_simple_gdn)
        else:
            self.attn = FrontierAttention(config, layer_idx)
        self.mlp = FrontierMLP(config)
        if self.variant == "zero_centered_rmsnorm":
            self.attn_norm = ZeroCenteredRMSNorm(config.n_embd)
            self.mlp_norm = ZeroCenteredRMSNorm(config.n_embd)
        if self.variant == "inkling_sconv_residual":
            self.attn_sconv = InklingShortConvolution(config.n_embd, config.sconv_kernel_size)
            self.mlp_sconv = InklingShortConvolution(config.n_embd, config.sconv_kernel_size)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        attn_input = self.attn_norm(x) if self.variant == "zero_centered_rmsnorm" else norm(x)
        attn_output = self.attn(attn_input, ve, cos_sin, window_size, kv_cache)
        if self.variant == "inkling_sconv_residual":
            attn_output = self.attn_sconv(attn_output)
        x = x + attn_output
        mlp_input = self.mlp_norm(x) if self.variant == "zero_centered_rmsnorm" else norm(x)
        mlp_output = self.mlp(mlp_input)
        if self.variant == "inkling_sconv_residual":
            mlp_output = self.mlp_sconv(mlp_output)
        return x + mlp_output


class SharedMTP(nn.Module):
    """Three teacher-forced future-token heads sharing one transformer block."""

    def __init__(self, config: FrontierPoolConfig):
        super().__init__()
        self.depth = config.mtp_depth
        self.mix = Linear(2 * config.n_embd, config.n_embd, bias=False)
        # This shared block is intentionally baseline-shaped.
        control = FrontierPoolConfig(
            **{**config.__dict__, "frontier_variant": "inkling_lr2_weight_decay"}
        )
        self.block = FrontierBlock(control, config.n_layer)

    def forward(self, hidden, idx, embed, cos_sin, lm_head, vocab_size):
        losses = []
        state = hidden
        for offset in range(1, self.depth + 1):
            valid = idx.size(1) - offset
            if valid <= 0:
                break
            teacher = norm(embed(idx[:, offset:]).to(state.dtype))
            state = self.mix(torch.cat([state[:, :valid], teacher], dim=-1))
            state = self.block(
                state, None, (cos_sin[0][:, :valid], cos_sin[1][:, :valid]), (valid, 0), None
            )
            logits = lm_head(norm(state))[..., :vocab_size].float()
            logits = 15 * torch.tanh(logits / 15)
            target = idx[:, offset + 1 :] if offset + 1 < idx.size(1) else idx[:, :0]
            if target.numel() == 0:
                break
            losses.append(
                F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), target.reshape(-1))
            )
        return torch.stack(losses).mean() if losses else hidden.new_zeros((), dtype=torch.float32)


class FrontierPoolGPT(GPT):
    def __init__(self, config: FrontierPoolConfig, pad_vocab_size_to: int = 64):
        super().__init__(config, pad_vocab_size_to=pad_vocab_size_to)
        self.transformer.h = nn.ModuleList(
            [FrontierBlock(config, i) for i in range(config.n_layer)]
        )
        if config.frontier_variant == "hybrid_swa_5_1_w512":
            self.window_sizes = [
                ((512, 0) if i % 6 < 5 else (config.sequence_len, 0)) for i in range(config.n_layer)
            ]
            self.window_sizes[-1] = (config.sequence_len, 0)
        if config.frontier_variant == "motif_gdla":
            self.window_sizes = [
                ((config.sequence_len, 0) if i % 4 == 0 else (128, 0))
                for i in range(config.n_layer)
            ]
        if config.frontier_variant == "zero_centered_rmsnorm":
            self.embed_norm = ZeroCenteredRMSNorm(config.n_embd)
            self.final_norm = ZeroCenteredRMSNorm(config.n_embd)
        if config.frontier_variant == "shared_mtp3":
            self.mtp = SharedMTP(config)
        if config.frontier_variant == "motif_mhc_anneal":
            self.mhc_connections = nn.ModuleList(
                [MotifMHCConnection(config) for _ in range(2 * config.n_layer)]
            )
            self._training_step = 0
        self._training_metrics = {}

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = math.sqrt(3.0) * self.config.n_embd**-0.5
        for block in self.transformer.h:
            if isinstance(block.attn, MotifGroupedDifferentialLatentAttention):
                attn = block.attn
                for module in attn.modules():
                    if isinstance(module, Linear):
                        torch.nn.init.uniform_(module.weight, -s, s)
                    elif isinstance(module, HeadSplitLinear):
                        for weight in module.weights:
                            torch.nn.init.uniform_(weight, -s, s)
                torch.nn.init.zeros_(attn.out_proj.weight)
                torch.nn.init.zeros_(attn.lambda_proj.weight)
            elif isinstance(block.attn, GLMMultiLatentAttention):
                attn = block.attn
                torch.nn.init.uniform_(attn.q_down.weight, -s, s)
                torch.nn.init.uniform_(attn.kv_down.weight, -s, s)
                for projection in (attn.q_up, attn.k_up, attn.v_up):
                    for weight in projection.weights:
                        torch.nn.init.uniform_(weight, -s, s)
                torch.nn.init.zeros_(attn.out_proj.weight)
            elif isinstance(block.attn, DeepSeekCompressedAttention):
                attn = block.attn
                for module in attn.modules():
                    if isinstance(module, Linear):
                        torch.nn.init.uniform_(module.weight, -s, s)
                torch.nn.init.zeros_(attn.out_proj.weight)
                torch.nn.init.zeros_(attn.position_bias_a)
                if hasattr(attn, "position_bias_b"):
                    torch.nn.init.zeros_(attn.position_bias_b)
            elif isinstance(block.attn, QwenGatedDeltaAttention):
                attn = block.attn
                if attn.simple:
                    for projection in (attn.c_q, attn.c_k, attn.c_v):
                        init_projection_uniform_(projection, -s, s)
                else:
                    if attn.per_head_muon:
                        for projection in (attn.q_proj, attn.k_proj, attn.v_proj, attn.z_proj):
                            init_projection_uniform_(projection, -s, s)
                    else:
                        torch.nn.init.uniform_(attn.in_proj_qkvz.weight, -s, s)
                    torch.nn.init.uniform_(attn.in_proj_ba.weight, -s, s)
                    torch.nn.init.kaiming_uniform_(attn.conv.weight, a=math.sqrt(5))
                    attn.a_log.uniform_(0.01, 16.0).log_()
                    attn.dt_bias.fill_(1.0)
                attn.output_norm_weight.fill_(1.0)
                torch.nn.init.zeros_(attn.out_proj.weight)
            else:
                for projection in (block.attn.c_q, block.attn.c_k, block.attn.c_v):
                    init_projection_uniform_(projection, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            if block.mlp.variant == "kimi_situ_glu":
                torch.nn.init.uniform_(block.mlp.situ.gate_proj.weight, -s * 0.4, s * 0.4)
                torch.nn.init.uniform_(block.mlp.situ.up_proj.weight, -s * 0.4, s * 0.4)
                torch.nn.init.zeros_(block.mlp.situ.down_proj.weight)
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
        for i in range(self.config.n_layer):
            self.resid_lambdas.data[i] = 1.15 - 0.10 * i / max(self.config.n_layer - 1, 1)
            self.x0_lambdas.data[i] = 0.20 - 0.15 * i / max(self.config.n_layer - 1, 1)
        for value_embed in self.value_embeds.values():
            torch.nn.init.uniform_(value_embed.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        self.smear_lambda.zero_()
        self.backout_lambda.fill_(0.2)

        # Variant-exclusive parameters are initialized only after every shared
        # tensor, preserving paired baseline initialization exactly.
        for block in self.transformer.h:
            if hasattr(block.attn, "relative_bank"):
                torch.nn.init.normal_(block.attn.r_proj.weight, std=0.02)
                torch.nn.init.normal_(block.attn.relative_bank, std=0.02)
            if hasattr(block.attn, "sink_logit"):
                block.attn.sink_logit.zero_()
            for name in ("k_sconv", "v_sconv"):
                if hasattr(block.attn, name):
                    getattr(block.attn, name).reset_parameters()
            for name in ("attn_sconv", "mlp_sconv"):
                if hasattr(block, name):
                    getattr(block, name).reset_parameters()
            for name in ("attn_norm", "mlp_norm"):
                if hasattr(block, name):
                    getattr(block, name).reset_parameters()
        if hasattr(self, "embed_norm"):
            self.embed_norm.reset_parameters()
            self.final_norm.reset_parameters()
        if hasattr(self, "mtp"):
            torch.nn.init.uniform_(self.mtp.mix.weight, -s, s)
            block = self.mtp.block
            for projection in (block.attn.c_q, block.attn.c_k, block.attn.c_v):
                torch.nn.init.uniform_(projection.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        if hasattr(self, "mhc_connections"):
            for connection in self.mhc_connections:
                torch.nn.init.xavier_uniform_(connection.mapping_proj.weight)
                connection.alpha.fill_(0.01)
                connection.bias.zero_()
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        compute_dtype = self.config.runtime.compute_dtype
        if compute_dtype != torch.float16:
            self.transformer.wte.to(dtype=compute_dtype)
            for value_embed in self.value_embeds.values():
                value_embed.to(dtype=compute_dtype)

    def _residual_norm(self, x: torch.Tensor, *, final: bool = False) -> torch.Tensor:
        if self.config.frontier_variant != "zero_centered_rmsnorm":
            return norm(x)
        return self.final_norm(x) if final else self.embed_norm(x)

    def _trunk(self, idx, kv_cache=None):
        _, time = idx.shape
        if time > self.cos.size(1):
            raise ValueError("sequence exceeds rotary cache")
        offset = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, offset : offset + time], self.sin[:, offset : offset + time]
        x = self._residual_norm(self.transformer.wte(idx).to(self.config.runtime.compute_dtype))
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
        use_mhc = self.config.frontier_variant == "motif_mhc_anneal"
        if use_mhc:
            x = x.unsqueeze(-2).expand(-1, -1, self.config.mhc_num_streams, -1).contiguous()
        x0 = x
        backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            if use_mhc:
                attention_connection = self.mhc_connections[2 * i]
                branch_input, post, residual = attention_connection.prepare(x)
                branch_output = block.attn(
                    norm(branch_input), ve, cos_sin, self.window_sizes[i], kv_cache
                )
                x = attention_connection.combine(x, branch_output, post, residual)
                mlp_connection = self.mhc_connections[2 * i + 1]
                branch_input, post, residual = mlp_connection.prepare(x)
                branch_output = block.mlp(norm(branch_input))
                x = mlp_connection.combine(x, branch_output, post, residual)
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == self.config.n_layer // 2:
                backout = x
        if backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * backout
        if use_mhc:
            x = x.mean(dim=-2)
        return self._residual_norm(x, final=True), cos_sin

    def set_training_step(self, step: int) -> None:
        if self.config.frontier_variant != "motif_mhc_anneal":
            return
        self._training_step = int(step)
        progress = min(max(self._training_step / self.config.mhc_anneal_steps, 0.0), 1.0)
        scale = 2.0 - progress
        for connection in self.mhc_connections:
            connection.set_post_scale(scale)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        hidden, cos_sin = self._trunk(idx, kv_cache)
        raw_logits = self.lm_head(hidden)[..., : self.config.vocab_size]
        logits = raw_logits.float()
        logits = 15 * torch.tanh(logits / 15)
        if targets is None:
            return logits
        lm_loss = language_model_loss(
            logits,
            raw_logits,
            targets,
            reduction=loss_reduction,
            z_loss_weight=self.config.z_loss_weight,
            training=self.training,
        )
        if self.config.frontier_variant != "shared_mtp3" or not self.training:
            return lm_loss
        mtp_loss = self.mtp(
            hidden,
            idx,
            self.transformer.wte,
            cos_sin,
            self.lm_head,
            self.config.vocab_size,
        )
        self._training_metrics = {
            "train/lm_loss": lm_loss.detach(),
            "mtp/loss": mtp_loss.detach(),
        }
        return lm_loss + self.config.mtp_loss_weight * mtp_loss

    def consume_training_metrics(self) -> dict:
        metrics, self._training_metrics = self._training_metrics, {}
        return metrics

    def num_scaling_params(self):
        wte = self.transformer.wte.weight.numel()
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = self.lm_head.weight.numel()
        controls = (
            self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.smear_gate.weight.numel()
            + self.smear_lambda.numel()
            + self.backout_lambda.numel()
        )
        trunk = sum(p.numel() for p in self.transformer.h.parameters())
        known = wte + value_embeds + lm_head + controls + trunk
        architecture = sum(p.numel() for p in self.parameters()) - known
        total = known + architecture
        assert total == sum(p.numel() for p in self.parameters())
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": trunk,
            "architecture": architecture,
            "scalars": controls,
            "total": total,
        }

    def estimate_flops(self):
        base = super().estimate_flops()
        head_dim = self.config.n_embd // self.config.n_head
        seq_len = self.config.sequence_len
        dense_layer = 12 * self.config.n_head * head_dim * seq_len

        if self.config.frontier_variant in {"qwen_gdn", "glm_simple_gdn"}:
            recurrent = [
                block.attn
                for block in self.transformer.h
                if isinstance(block.attn, QwenGatedDeltaAttention)
            ]
            # Linear recurrent state update/read FLOPs. Projection and causal
            # convolution FLOPs are already represented by matrix parameters.
            recurrent_internal = sum(
                6 * attn.num_v_heads * attn.head_k_dim * attn.head_v_dim for attn in recurrent
            )
            return base - len(recurrent) * dense_layer + recurrent_internal

        if self.config.frontier_variant in {"deepseek_csa", "deepseek_hca"}:
            heavy = self.config.frontier_variant == "deepseek_hca"
            compression = 128 if heavy else 4
            groups = math.ceil(seq_len / compression)
            selected_groups = groups if heavy else min(groups, 512)
            algorithmic_attention = (
                12
                * self.config.n_head
                * head_dim
                * (selected_groups + min(seq_len, 128))
                * self.config.n_layer
            )
            return base - self.config.n_layer * dense_layer + algorithmic_attention

        if self.config.frontier_variant == "shared_mtp3":
            depth = self.config.mtp_depth
            shared_matrices = (
                sum(p.numel() for p in self.mtp.block.parameters() if p.ndim >= 2)
                + self.mtp.mix.weight.numel()
            )
            repeated_weight_flops = 6 * (
                (depth - 1) * shared_matrices + depth * self.lm_head.weight.numel()
            )
            repeated_attention = depth * dense_layer
            return base + repeated_weight_flops + repeated_attention

        return base

    def estimate_executed_flops(self):
        algorithmic = self.estimate_flops()
        if self.config.frontier_variant not in {"deepseek_csa", "deepseek_hca"}:
            return algorithmic
        head_dim = self.config.n_embd // self.config.n_head
        seq_len = self.config.sequence_len
        compression = 128 if self.config.frontier_variant == "deepseek_hca" else 4
        groups = math.ceil(seq_len / compression)
        algorithmic_attention = (
            12
            * self.config.n_head
            * head_dim
            * ((groups if compression == 128 else min(groups, 512)) + min(seq_len, 128))
            * self.config.n_layer
        )
        # The controlled PyTorch SDPA fallback materializes the masked local
        # token axis, so label this separately from the report's sparse cost.
        fallback_attention = (
            12 * self.config.n_head * head_dim * (seq_len + groups) * self.config.n_layer
        )
        return algorithmic - algorithmic_attention + fallback_attention

    def get_architecture_state(self) -> dict:
        state = {
            "family": "frontier_pool",
            "variant": self.config.frontier_variant,
            "shared_backbone": "nanochat_d14",
            "component_registry": "frontier_report_campaign/component_registry.json",
        }
        if self.config.frontier_variant == "inkling_relative_attention":
            state.update(
                position="content_dependent_relative_no_rope",
                d_rel=self.config.relative_dim,
                extent=self.config.relative_extent,
            )
        elif self.config.frontier_variant in {"inkling_sconv_kv", "inkling_sconv_residual"}:
            state.update(kernel=self.config.sconv_kernel_size, compute="fp32", residual=True)
        elif self.config.frontier_variant == "hybrid_swa_5_1_w512":
            state.update(
                local_global_ratio="5:1", sliding_window=512, layer_windows=self.window_sizes
            )
        elif self.config.frontier_variant == "partial_rope_25":
            state.update(rotary_fraction=0.25)
        elif self.config.frontier_variant == "zero_centered_rmsnorm":
            state.update(norm="rms_norm_times_one_plus_zero_initialized_weight")
        elif self.config.frontier_variant == "kimi_situ_glu":
            state.update(
                beta1=4.0,
                beta2=25.0,
                intermediate_size=self.transformer.h[0].mlp.situ.intermediate_size,
            )
        elif self.config.frontier_variant == "shared_mtp3":
            state.update(
                shared_mtp_depth=self.config.mtp_depth, loss_weight=self.config.mtp_loss_weight
            )
        elif self.config.frontier_variant == "attention_sink":
            state.update(sink="learned_per_head_softmax_denominator_with_zero_value")
        elif self.config.frontier_variant == "inkling_lr2_weight_decay":
            state.update(weight_decay_schedule="coupled_to_lr_squared")
        elif self.config.frontier_variant == "per_head_muon":
            state.update(muon_logical_matrix="one_qkv_projection_per_attention_head")
        elif self.config.frontier_variant == "qwen_gdn":
            state.update(
                token_mixer="Qwen3-Next GatedDeltaNet",
                execution_backend="fla_chunk_sm103_2w2s",
                layer_pattern="GGGF",
                key_heads=self.config.n_head,
                value_heads=2 * self.config.n_head,
                key_head_dim=128,
                value_head_dim=128,
                conv_kernel=4,
            )
        elif self.config.frontier_variant == "glm_simple_gdn":
            state.update(
                token_mixer="GLM-5 SimpleGDN",
                execution_backend="fla_chunk_sm103_2w2s",
                layer_pattern="GF",
                conv=False,
                explicit_gates=False,
                qkv_reuse=True,
            )
        elif self.config.frontier_variant in {"deepseek_csa", "deepseek_hca"}:
            heavy = self.config.frontier_variant == "deepseek_hca"
            state.update(
                token_mixer="DeepSeek-V4 HCA" if heavy else "DeepSeek-V4 CSA",
                compression=128 if heavy else 4,
                overlap=not heavy,
                shared_key_value=True,
                query_latent=self.config.n_embd // 2,
                partial_rope_dimensions=64,
                inverse_rope_output=True,
                local_window=128,
                csa_topk=512 if not heavy else None,
                selector_identity_at_sequence_2048=not heavy,
                dsa_selector_artifact="outputs/dsa-ablation-results.json" if not heavy else None,
            )
        elif self.config.frontier_variant == "glm_mla_muon_split":
            state.update(
                token_mixer="GLM-5 Multi-Latent Attention",
                latent_dimensions=256,
                muon_split="independent_qkv_up_projection_matrix_per_head",
                attention_heads=self.config.n_head,
                fixed_backbone_adaptation="retains d14's seven 128-d heads",
            )
        elif self.config.frontier_variant == "motif_gdla":
            state.update(
                token_mixer="Motif 3 Grouped Differential Latent Attention",
                signal_heads=self.config.n_head - 1,
                noise_heads=1,
                group_ratio=self.config.n_head - 1,
                report_ratio="64 signal : 16 noise (g=4)",
                fixed_backbone_adaptation="6 signal : 1 noise (g=6)",
                latent_dimensions=256,
                query_dependent_lambda=True,
                elementwise_output_gate=True,
                layer_pattern="FSSS",
                sliding_window=128,
            )
        elif self.config.frontier_variant == "motif_mhc_anneal":
            state.update(
                residual="Motif 3 modified manifold-constrained hyper-connections",
                streams=self.config.mhc_num_streams,
                sinkhorn_iterations=self.config.mhc_sinkhorn_iterations,
                post_scale_start=2.0,
                post_scale_end=1.0,
                anneal_steps=self.config.mhc_anneal_steps,
                current_step=self._training_step,
            )
        return state
