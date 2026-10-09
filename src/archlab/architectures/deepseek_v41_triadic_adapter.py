"""Triadic/GDN branches using the existing DeepSeek residual adapter shell.

The official dependency owns GPU delta-rule and causal-convolution kernels.
Both E=1 GDN and E>1 Triadic use the same causal q/k/v convolution, normalized
keys/queries, sigmoid write strength and per-slice log decay. Unlike RF LinSimp,
the second axis is a learned feature vector of the same token, not a second
token window. This is an explicit mechanism comparison, not a softmax drop-in.
"""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter
from archlab.architectures.triadic_attention import (
    SUPPORTED_FEATURE_DIMS,
    causal_conv_attention,
    triadic_gdn_attention,
)


def _geometry(config, feature_dim):
    if config.query_heads != config.kv_heads or config.head_dim != 128:
        raise ValueError("matched Triadic adapters require equal Q/KV heads and head_dim 128")
    if feature_dim not in SUPPORTED_FEATURE_DIMS:
        raise ValueError("unsupported Triadic feature dimension")


def triadic_adapter_parameter_count(config: V41AdapterConfig, *, feature_dim=4):
    _geometry(config, feature_dim)
    heads, width, dim = config.query_heads, config.width, config.head_dim
    features = heads * feature_dim
    qkv_channels = 3 * heads * dim
    extra_channels = 2 * features if feature_dim > 1 else 0
    # Common read/write, input norm, q/k norms, q/k/v, output gate and output.
    common = width * (5 * heads * dim) + width + 2 * dim + 2 * config.streams
    return (
        common
        + width * heads  # beta
        + width * features  # log-decay projection
        + 2 * features  # A_log and dt_bias
        + width * extra_channels  # learned positive feature projections
        + 4 * (qkv_channels + extra_channels)  # causal depthwise convolution
    )


def _normalize(x, epsilon):
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + epsilon)


def _initialization_generator(seed, name):
    """A CPU stream whose draws do not depend on other projection shapes."""
    identity = hashlib.sha256(f"triadic-gdn:{seed}:{name}".encode()).digest()
    return torch.Generator(device="cpu").manual_seed(int.from_bytes(identity[:8], "little"))


class V41TriadicAttentionAdapter(nn.Module):
    """E=1 is GDN; E>1 is the joint-key Triadic gated delta rule."""

    def __init__(self, config: V41AdapterConfig, *, feature_dim=4, seed=42, backend="official"):
        super().__init__()
        _geometry(config, feature_dim)
        if backend not in ("official", "reference"):
            raise ValueError("Triadic adapter backend must be official or reference")
        self.config, self.feature_dim, self.backend = config, feature_dim, backend
        source = V41SimplicialAdapter(config, seed=seed, backend="reference")
        self.read_logits, self.write_logits = source.read_logits, source.write_logits
        for name in ("input_norm", "q", "q_norm", "output_gate", "output"):
            setattr(self, name, getattr(source, name))
        self.k, self.v, self.k_norm = source.k2, source.v2, source.k2_norm
        heads, width, dim = config.query_heads, config.width, config.head_dim
        self.qkv_channels = 3 * heads * dim
        features = heads * feature_dim
        extra_channels = 2 * features if feature_dim > 1 else 0
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 15485863)
            self.beta_projection = nn.Linear(width, heads, bias=False)
            self.decay_projection = nn.Linear(width, features, bias=False)
            if feature_dim > 1:
                self.k2 = nn.Linear(width, features, bias=False)
                self.q2 = nn.Linear(width, features, bias=False)
            else:
                self.k2 = self.q2 = None
            self.conv_weight = nn.Parameter(torch.empty(self.qkv_channels + extra_channels, 4))
            nn.init.uniform_(self.conv_weight, -0.5, 0.5)
            for projection in (self.beta_projection, self.decay_projection, self.k2, self.q2):
                if projection is not None:
                    nn.init.normal_(projection.weight, std=config.initializer_std)
            self.A_log = nn.Parameter(torch.rand(heads, feature_dim).mul(15).add(1).log())
            dt = torch.rand(heads, feature_dim).mul(math.log(100)).add(math.log(0.001)).exp()
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
            # E-dependent constructors consume different amounts of random
            # state. Match the shared delta write and Q/K/V convolution with
            # independent named streams, while retaining the common source
            # Q/K/V values and the other projection initialization paths.
            nn.init.normal_(self.beta_projection.weight, std=config.initializer_std,
                            generator=_initialization_generator(seed, "beta"))
            nn.init.uniform_(self.conv_weight[:self.qkv_channels], -0.5, 0.5,
                             generator=_initialization_generator(seed, "qkv-conv"))
        expected = triadic_adapter_parameter_count(config, feature_dim=feature_dim)
        if sum(parameter.numel() for parameter in self.parameters()) != expected:
            raise ValueError("Triadic adapter parameter count differs from its contract")

    def mechanism_contract(self):
        return {
            "kind": "gdn" if self.feature_dim == 1 else "triadic-gdn",
            "feature_dim": self.feature_dim,
            "causal_conv_width": 4,
            "conv_activation": "SiLU(q,k,v); identity(k2,q2) before positive feature map",
            "second_feature_map": "L2(softplus)",
            "first_feature_map": "L2",
            "normalization_epsilon": self.config.norm_eps,
            "read_scale": self.config.head_dim**-0.5,
            "decay": "-exp(A_log)*softplus(decay_projection(x)+dt_bias)",
            "write_strength": "sigmoid(beta_projection(x))",
            "state_update": "slice decay then one joint delta residual",
            "state_elements_per_head": self.feature_dim * self.config.head_dim**2,
            "backend": self.backend,
        }

    def forward(self, streams, *, cu_seqlens=None):
        with torch.autocast(
            streams.device.type,
            dtype=torch.bfloat16,
            enabled=streams.is_cuda and streams.dtype == torch.bfloat16,
        ):
            return self._forward(streams, cu_seqlens=cu_seqlens)

    def _forward(self, streams, *, cu_seqlens):
        c = self.config
        if streams.ndim != 4 or streams.shape[-2:] != (c.streams, c.width):
            raise ValueError("expected [batch, sequence, residual_streams, width]")
        batch, length = streams.shape[:2]
        if batch < 1 or length < 1:
            raise ValueError("empty adapter input")
        read = self.read_logits.float().softmax(-1)
        x = self.input_norm((streams.float() * read[None, None, :, None]).sum(-2).to(streams.dtype))
        projections = (self.q(x), self.k(x), self.v(x))
        with torch.autocast(streams.device.type, enabled=False):
            x32 = x.float()
            parts = [projection.float() for projection in projections]
            # Q/K/V projections retain BF16 matrix arithmetic on the GPU; gates
            # and the small feature axis stay FP32, including convolution.
            if self.feature_dim > 1:
                parts += [
                    F.linear(x32, self.k2.weight.float()),
                    F.linear(x32, self.q2.weight.float()),
                ]
            joined = torch.cat(parts, -1)
            convolved = causal_conv_attention(
                joined,
                self.conv_weight.float(),
                self.qkv_channels,
                backend=self.backend,
                cu_seqlens=cu_seqlens,
            )
            channels = c.query_heads * c.head_dim
            q, k, v = convolved[..., : self.qkv_channels].split(channels, -1)
            shape = (batch, length, c.query_heads, c.head_dim)
            q = _normalize(self.q_norm(q.reshape(shape)).float(), c.norm_eps)
            k = _normalize(self.k_norm(k.reshape(shape)).float(), c.norm_eps)
            v = v.reshape(shape)
            feature_shape = (batch, length, c.query_heads, self.feature_dim)
            if self.feature_dim > 1:
                k2, q2 = convolved[..., self.qkv_channels :].chunk(2, -1)
                k2 = _normalize(F.softplus(k2.reshape(feature_shape)), c.norm_eps)
                q2 = _normalize(F.softplus(q2.reshape(feature_shape)), c.norm_eps)
            else:
                # Positive scalar L2 normalization is constant: do not add
                # unidentifiable learned projections or convolution channels.
                k2 = q2 = x32.new_ones(feature_shape)
            g = -self.A_log.float().exp() * F.softplus(
                F.linear(x32, self.decay_projection.weight.float()).reshape(feature_shape)
                + self.dt_bias.float()
            )
            beta = F.linear(x32, self.beta_projection.weight.float()).sigmoid()
            core_dtype = torch.bfloat16 if self.backend == "official" else x.dtype
            attended = (
                triadic_gdn_attention(
                    q.to(core_dtype),
                    k.to(core_dtype),
                    v.to(core_dtype),
                    k2,
                    q2,
                    g,
                    beta,
                    backend=self.backend,
                    scale=c.head_dim**-0.5,
                    cu_seqlens=cu_seqlens,
                )
                .to(x.dtype)
                .flatten(-2)
            )
        attended = (attended.float() * self.output_gate(x).float().sigmoid()).to(attended.dtype)
        branch = self.output(attended)
        write = 2 * self.write_logits.float().sigmoid()
        return (streams.float() + branch.float().unsqueeze(-2) * write[None, None, :, None]).to(
            streams.dtype
        )
