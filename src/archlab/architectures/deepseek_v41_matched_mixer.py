"""Parameter-matched RF, LinSimp, GDN and Triadic additive residual branches.

The backbone and residual read/write maps are shared. Each arm contains a
useful parallel SwiGLU, compensating for core parameter differences without
dummy weights. The target is the LinSimp core plus an intermediate-1024 MLP.
Aligned intermediate widths leave a small, explicitly reported residual.

The RF arms do not receive GDN's causal convolution, delta rule or decay gates.
GDN E=1 is therefore the mechanism control for Triadic E>1; RF comparisons
compare entire mixer designs, not just an extra key feature axis.
"""

from __future__ import annotations

import copy
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
from archlab.architectures.deepseek_v41_linsimp_adapter import (
    V41LinearAttentionAdapter,
    V41LinSimpAdapter,
    linear_adapter_parameter_count,
    linsimp_adapter_parameter_count,
)
from archlab.architectures.deepseek_v41_triadic_adapter import (
    V41TriadicAttentionAdapter,
    triadic_adapter_parameter_count,
)


def matched_mixer_parameter_contract(
    config: V41AdapterConfig,
    variant,
    *,
    feature_dim=4,
    compensation_anchor=1024,
    compensation_alignment=16,
):
    if type(compensation_anchor) is not int or compensation_anchor < 1:
        raise ValueError("compensation anchor must be a positive integer")
    if type(compensation_alignment) is not int or compensation_alignment < 1:
        raise ValueError("compensation alignment must be a positive integer")
    if variant == "linear":
        core = linear_adapter_parameter_count(config)
    elif variant == "linsimp":
        core = linsimp_adapter_parameter_count(config)
    elif variant in ("gdn", "triadic"):
        features = 1 if variant == "gdn" else feature_dim
        if variant == "triadic" and features <= 1:
            raise ValueError("Triadic requires more than one feature; use gdn for E=1")
        core = triadic_adapter_parameter_count(config, feature_dim=features)
    else:
        raise ValueError("matched mixer variant must be linear, linsimp, gdn or triadic")
    target = linsimp_adapter_parameter_count(config) + 3 * config.width * compensation_anchor
    unit = 3 * config.width * compensation_alignment
    intermediate = max(1, (target - core + unit // 2) // unit) * compensation_alignment
    mlp = 3 * config.width * intermediate
    total = core + mlp
    return {
        "variant": variant,
        "core_parameters": core,
        "compensation_parameters": mlp,
        "compensation_intermediate": intermediate,
        "compensation_anchor": compensation_anchor,
        "compensation_alignment": compensation_alignment,
        "target_parameters": target,
        "total_parameters": total,
        "parameter_difference": total - target,
        "absolute_parameter_error": abs(total - target),
        "relative_parameter_error": abs(total - target) / target,
        "budget_excludes": ["backbone", "token embeddings", "Engram"],
        "compensation": "parallel residual SwiGLU; zero-initialized down projection",
    }


def matched_mixer_parameter_count(config, variant, **kwargs):
    return matched_mixer_parameter_contract(config, variant, **kwargs)["total_parameters"]


def _initial_tensor_sha256(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{tuple(tensor.shape)}:{tensor.dtype}".encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class MatchedMixerAdapter(nn.Module):
    """Drop-in [B,T,residual_streams,width] branch for the shared installer."""

    def __init__(
        self,
        config: V41AdapterConfig,
        *,
        variant,
        feature_dim=4,
        seed=42,
        backend=None,
        compensation_anchor=1024,
        compensation_alignment=16,
    ):
        super().__init__()
        self.config, self.variant = config, variant
        self._parameter_contract = matched_mixer_parameter_contract(
            config,
            variant,
            feature_dim=feature_dim,
            compensation_anchor=compensation_anchor,
            compensation_alignment=compensation_alignment,
        )
        if variant in ("linear", "linsimp"):
            self.backend = "reference" if backend is None else backend
            cls = V41LinearAttentionAdapter if variant == "linear" else V41LinSimpAdapter
            self.core = cls(config, seed=seed, backend=self.backend)
        else:
            self.backend = "official" if backend is None else backend
            self.core = V41TriadicAttentionAdapter(
                config,
                feature_dim=1 if variant == "gdn" else feature_dim,
                seed=seed,
                backend=self.backend,
            )
        intermediate = self._parameter_contract["compensation_intermediate"]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 32452843)
            self.compensation_gate = nn.Linear(config.width, intermediate, bias=False)
            self.compensation_up = nn.Linear(config.width, intermediate, bias=False)
            self.compensation_down = nn.Linear(intermediate, config.width, bias=False)
            for module in (self.compensation_gate, self.compensation_up):
                nn.init.normal_(module.weight, std=config.initializer_std)
            nn.init.zeros_(self.compensation_down.weight)
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != self._parameter_contract["total_parameters"]:
            raise ValueError("matched mixer parameters differ from the declared budget")
        long_key = self.core.k2 if variant == "linsimp" else self.core.k
        long_value = self.core.v2 if variant == "linsimp" else self.core.v
        long_norm = self.core.k2_norm if variant == "linsimp" else self.core.k_norm
        shared = {
            "read_logits": self.core.read_logits, "write_logits": self.core.write_logits,
            "input_norm": self.core.input_norm.weight, "q": self.core.q.weight,
            "q_norm": self.core.q_norm.weight, "k": long_key.weight,
            "k_norm": long_norm.weight, "v": long_value.weight,
            "output_gate": self.core.output_gate.weight, "output": self.core.output.weight,
        }
        delta = {} if variant in ("linear", "linsimp") else {
            "beta_projection": self.core.beta_projection.weight,
            "qkv_causal_conv": self.core.conv_weight[:self.core.qkv_channels],
        }
        # Capture before sharding, dtype conversion, or optimizer updates.
        # These immutable receipts describe initialization, not current state.
        self._initialization_contract = {
            "format": "archlab-matched-mixer-initialization-v2", "seed": seed,
            "common_core_sha256": {name: _initial_tensor_sha256(p) for name, p in shared.items()},
            "delta_shared_sha256": {name: _initial_tensor_sha256(p) for name, p in delta.items()},
            "delta_rng_streams": [] if not delta else ["beta", "qkv-conv"],
        }

    @property
    def read_logits(self):
        return self.core.read_logits

    @property
    def write_logits(self):
        return self.core.write_logits

    @property
    def input_norm(self):
        return self.core.input_norm

    def parameter_contract(self):
        return dict(self._parameter_contract)

    def initialization_contract(self):
        return copy.deepcopy(self._initialization_contract)

    def forward(self, streams):
        with torch.autocast(
            streams.device.type,
            dtype=torch.bfloat16,
            enabled=streams.is_cuda and streams.dtype == torch.bfloat16,
        ):
            mixed = self.core(streams)
            read = self.read_logits.float().softmax(-1)
            x = self.input_norm(
                (streams.float() * read[None, None, :, None]).sum(-2).to(streams.dtype)
            )
            compensation = self.compensation_down(
                F.silu(self.compensation_gate(x)) * self.compensation_up(x)
            )
            write = 2 * self.write_logits.float().sigmoid()
            return (
                mixed.float() + compensation.float().unsqueeze(-2) * write[None, None, :, None]
            ).to(streams.dtype)
