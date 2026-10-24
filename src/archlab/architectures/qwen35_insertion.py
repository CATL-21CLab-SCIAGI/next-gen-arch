"""Additive attention branches inserted after selected Qwen3.5 decoder layers (Hugging Face `qwen3_5`).

Each branch reads the layer's output hidden state, applies a pre-norm attention mechanism and adds a gated,
output-projected result back to the residual stream. With the default zero output projection, the model computes
exactly its parent's function at initialization. Branches are attached with forward hooks, so upstream modules,
parameters and checkpoint keys are unchanged; branch weights live under `model.inserted_branches`.

Mechanisms share placement, width, heads and windows so their comparison isolates the interaction order:
- `simplicial`: local 2-simplicial attention (two key/value streams, short x long window), the repository kernel;
- `local`: ordinary causal sliding-window attention over the long window (one key/value stream), the matched control.
The simplicial branch has two more K/V projections than the local one; report parameters with every comparison.
Decoding with a cache is not supported here: training, distillation and teacher-forced evaluation only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn

from archlab.architectures.local_attention import deterministic_local_attention, reference_local_attention
from archlab.architectures.qwen38_flash_next_full import GroupRMSNorm
from archlab.architectures.simplicial_adapter import SimplicialAdapterConfig, partial_rope
from archlab.architectures.simplicial_attention import reference_simplicial, simplicial_attention


@dataclass(frozen=True)
class InsertionConfig:
    hidden_size: int
    mechanism: Literal["simplicial", "local"] = "simplicial"
    query_heads: int = 16
    kv_heads: int = 4
    head_dim: int = 256
    short_window: int = 16
    long_window: int = 128
    rotary_fraction: float = 0.25
    rope_theta: float = 10_000_000.0
    rms_norm_eps: float = 1e-6
    initializer_std: float = 0.02
    output_initialization: Literal["zeros", "normal"] = "zeros"

    def __post_init__(self):
        if self.mechanism not in ("simplicial", "local"):
            raise ValueError("mechanism must be simplicial or local")
        # The adapter config validates heads, head dimension, windows, RoPE and initialization in one place.
        SimplicialAdapterConfig(hidden_size=self.hidden_size, query_heads=self.query_heads, kv_heads=self.kv_heads,
                                head_dim=self.head_dim, residual_streams=1, residual_low_rank=1,
                                short_window=self.short_window, long_window=self.long_window,
                                rotary_fraction=self.rotary_fraction, rope_theta=self.rope_theta,
                                rms_norm_eps=self.rms_norm_eps, initializer_std=self.initializer_std,
                                output_initialization=self.output_initialization)

    @property
    def rope(self) -> SimplicialAdapterConfig:
        return SimplicialAdapterConfig(head_dim=self.head_dim, rotary_fraction=self.rotary_fraction,
                                       rope_theta=self.rope_theta, query_heads=self.query_heads,
                                       kv_heads=self.kv_heads)

    def branch_parameters(self) -> int:
        q, kv = self.query_heads * self.head_dim, self.kv_heads * self.head_dim
        streams = 2 if self.mechanism == "simplicial" else 1
        return self.hidden_size * (3 * q + 2 * streams * kv) + self.hidden_size + (1 + streams) * self.head_dim


class InsertedBranch(nn.Module):
    """Pre-norm attention branch with a sigmoid output gate; zero output projection by default."""

    def __init__(self, config: InsertionConfig, *, seed: int, backend: Literal["kernel", "reference"] = "kernel"):
        super().__init__()
        if torch.get_default_device().type not in ("cpu", "meta"):
            raise ValueError("construct branches on CPU/meta, then move them to the execution device")
        self.config, self.backend, self.rope_config = config, backend, config.rope
        streams = 2 if config.mechanism == "simplicial" else 1
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            width, q, kv = config.hidden_size, config.query_heads * config.head_dim, config.kv_heads * config.head_dim
            self.norm = GroupRMSNorm(width, 1, config.rms_norm_eps, zero_centered=True)
            self.q = nn.Linear(width, q, bias=False)
            self.output_gate = nn.Linear(width, q, bias=False)
            self.keys = nn.ModuleList(nn.Linear(width, kv, bias=False) for _ in range(streams))
            self.values = nn.ModuleList(nn.Linear(width, kv, bias=False) for _ in range(streams))
            self.output = nn.Linear(q, width, bias=False)
            self.q_norm = GroupRMSNorm(config.head_dim, 1, config.rms_norm_eps, zero_centered=True)
            self.k_norms = nn.ModuleList(GroupRMSNorm(config.head_dim, 1, config.rms_norm_eps, zero_centered=True)
                                         for _ in range(streams))
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=config.initializer_std)
            if config.output_initialization == "zeros":
                nn.init.zeros_(self.output.weight)

    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        config, (batch, length) = self.config, hidden.shape[:2]
        x = self.norm(hidden)
        qshape = (batch, length, config.query_heads, config.head_dim)
        kvshape = (batch, length, config.kv_heads, config.head_dim)
        q = partial_rope(self.q_norm(self.q(x).reshape(qshape)), position_ids, self.rope_config)
        keys = [partial_rope(norm(k(x).reshape(kvshape)), position_ids, self.rope_config)
                for k, norm in zip(self.keys, self.k_norms)]
        values = [v(x).reshape(kvshape) for v in self.values]
        if config.mechanism == "simplicial":
            core = simplicial_attention if self.backend == "kernel" else reference_simplicial
            attended = core(q, keys[0], keys[1], values[0], values[1], config.short_window, config.long_window)
        else:
            core = deterministic_local_attention if self.backend == "kernel" else reference_local_attention
            attended = core(q, keys[0], values[0], config.long_window)
        gated = attended.flatten(-2).float() * torch.sigmoid(self.output_gate(x).float())
        return self.output(gated.to(hidden.dtype))


def stratified_sites(layer_types: list[str], fraction: float = 0.25) -> list[int]:
    """Insertion sites: the same `fraction` of each layer type, evenly spaced within the type (predeclared rule)."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    sites = []
    for kind in sorted(set(layer_types)):
        indices = [i for i, t in enumerate(layer_types) if t == kind]
        count = max(1, round(len(indices) * fraction))
        stride = len(indices) / count
        sites += [indices[int((j + 1) * stride) - 1] for j in range(count)]
    return sorted(sites)


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    inner = getattr(model, "model", model)
    inner = getattr(inner, "language_model", inner)
    return inner.layers


def install(model: nn.Module, config: InsertionConfig, sites: list[int], *, seed: int = 42,
            backend: Literal["kernel", "reference"] = "kernel") -> list:
    """Attach one branch after each site layer; returns hook handles (remove() detaches the branches)."""
    if hasattr(model, "inserted_branches"):
        raise ValueError("branches are already installed")
    layers = decoder_layers(model)
    if not sites or any(not 0 <= s < len(layers) for s in sites) or len(set(sites)) != len(sites):
        raise ValueError("sites must be distinct decoder-layer indices")
    reference = next(model.parameters())
    branches = nn.ModuleDict({str(site): InsertedBranch(config, seed=seed + site, backend=backend) for site in sites})
    model.inserted_branches = branches.to(device=reference.device, dtype=reference.dtype)
    model.inserted_sites = list(sites)

    def hook(module, args, kwargs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        positions = kwargs.get("position_ids")
        if positions is None or positions.ndim == 3:  # multimodal RoPE ids: [3, batch, length]; text rows are equal
            positions = positions[0] if positions is not None else \
                torch.arange(hidden.shape[1], device=hidden.device).expand(hidden.shape[0], -1)
        updated = hidden + model.inserted_branches[str(module.archlab_site)](hidden, positions)
        return (updated, *output[1:]) if isinstance(output, tuple) else updated

    handles = []
    for site in sites:
        layers[site].archlab_site = site
        handles.append(layers[site].register_forward_hook(hook, with_kwargs=True))
    return handles


def freeze_backbone(model: nn.Module) -> dict:
    """Train only inserted branches; returns trainable and frozen parameter counts."""
    trainable = frozen = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("inserted_branches."))
        trainable += parameter.numel() if parameter.requires_grad else 0
        frozen += 0 if parameter.requires_grad else parameter.numel()
    return {"trainable": trainable, "frozen": frozen}


def describe(config: InsertionConfig, sites: list[int], layer_types: list[str]) -> dict:
    return {"config": asdict(config), "sites": sites, "site_types": [layer_types[s] for s in sites],
            "branch_parameters": config.branch_parameters() * len(sites)}
