"""Composable architecture-search and Pareto-combination definitions.

The search path keeps the base attention and residual controls unchanged. The
two alternate FFNs are approximately parameter matched to the
baseline 4x ReLU-squared FFN, while QAT follows torchao's 8da4w numerics:
dynamic asymmetric int8 activations per token and dynamic symmetric int4
weights per output-channel group.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from next_gen_arch.arch.base import EngramMemory, GPT, GPTConfig, Linear, has_ve, norm
from next_gen_arch.arch.frontier import (
    FrontierPoolGPT,
    MotifMHCConnection,
    QwenGatedDeltaAttention,
)
from next_gen_arch.arch.sota import SotaAttention, XIELU
from next_gen_arch.training.optim import DistMuonAdamW, MuonAdamW
from next_gen_arch.training.runtime import COMPUTE_DTYPE, get_dist_info, print0

try:
    import torchao
    from torchao.quantization.qat import IntxFakeQuantizeConfig, IntxFakeQuantizer
except ImportError:  # Baseline and sparse-only runs do not require torchao.
    torchao = None
    IntxFakeQuantizeConfig = None
    IntxFakeQuantizer = None


_MLP_VARIANTS = frozenset({"baseline", "sparser", "colu"})
_QAT_RECIPES = frozenset({"none", "8da4w"})


@dataclass
class ComboSearchConfig(GPTConfig):
    arch_family: str = "combo_search"
    search_mlp: str = "baseline"
    gated_mlp_width: int = -1
    sparser_l1_coeff: float = 0.0
    colu_dim: int = 4
    colu_scaling: str = "soft"
    qat_recipe: str = "none"
    qat_group_size: int = 128
    qat_start_step: int = 0
    qat_min_size: int = 128

    def __post_init__(self) -> None:
        if self.search_mlp not in _MLP_VARIANTS:
            raise ValueError(f"unknown search_mlp={self.search_mlp!r}")
        if self.qat_recipe not in _QAT_RECIPES:
            raise ValueError(f"unknown qat_recipe={self.qat_recipe!r}")
        if self.sparser_l1_coeff < 0:
            raise ValueError("sparser_l1_coeff must be non-negative")
        if self.search_mlp == "baseline" and self.sparser_l1_coeff != 0:
            raise ValueError("L1 activation regularization requires a gated FFN")
        if self.colu_scaling != "soft":
            raise ValueError("the controlled CoLU ablation requires scaling='soft'")
        if self.colu_dim < 2:
            raise ValueError("colu_dim must be at least 2")
        if self.qat_group_size <= 0:
            raise ValueError("qat_group_size must be positive")
        if self.qat_start_step < 0:
            raise ValueError("qat_start_step must be non-negative")
        if self.qat_min_size <= 0:
            raise ValueError("qat_min_size must be positive")


def parameter_matched_gated_width(model_dim: int, *, multiple: int = 1) -> int:
    """Width of a 3-matrix gated FFN matching a 4x 2-matrix FFN.

    Baseline parameters per layer are ``8*d*d``.  A gated FFN uses
    ``3*d*m``, so the exact real-valued match is ``m=8*d/3``.  ``multiple``
    is used by CoLU so the output channels form complete cone groups.
    """
    if model_dim <= 0 or multiple <= 0:
        raise ValueError("model_dim and multiple must be positive")
    target = 8.0 * model_dim / 3.0
    return max(multiple, int(round(target / multiple)) * multiple)


def _ste_replace(original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Forward with ``quantized`` values and identity STE to ``original``."""
    return original + (quantized.to(original.dtype) - original).detach()


def fake_quantize_int8_per_token(x: torch.Tensor) -> torch.Tensor:
    """Dynamic asymmetric int8 fake quantization along the last dimension."""
    if x.device.type == "meta":
        return x
    x32 = x.float()
    xmin = x32.amin(dim=-1, keepdim=True)
    xmax = x32.amax(dim=-1, keepdim=True)
    eps = torch.finfo(torch.float32).eps
    scale = ((xmax - xmin) / 255.0).clamp_min(eps)
    zero = (-128.0 - xmin / scale).round().clamp(-128.0, 127.0)
    q = (x32 / scale).round().add(zero).clamp(-128.0, 127.0)
    dq = (q - zero) * scale
    return _ste_replace(x, dq)


def fake_quantize_int4_per_group(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dynamic symmetric signed-int4 fake quantization with tail padding."""
    if weight.device.type == "meta":
        return weight
    if weight.ndim != 2:
        raise ValueError(f"expected a matrix weight, got shape={tuple(weight.shape)}")
    out_features, in_features = weight.shape
    padded_in = math.ceil(in_features / group_size) * group_size
    w32 = weight.float()
    if padded_in != in_features:
        w32 = F.pad(w32, (0, padded_in - in_features))
    grouped = w32.view(out_features, padded_in // group_size, group_size)
    eps = torch.finfo(torch.float32).eps
    # torchao's symmetric signed-int4 path uses zero point 0.  Keeping the
    # positive endpoint at seven avoids an asymmetric magnitude range.
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(eps)
    q = (grouped / scale).round().clamp(-8.0, 7.0)
    dq = (q * scale).reshape(out_features, padded_in)[:, :in_features]
    return _ste_replace(weight, dq)


class QATLinear(Linear):
    """Linear with torchao-compatible 8da4w fake-quantization semantics."""

    def __init__(self, *args, group_size: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        if torchao is None:
            raise RuntimeError("8da4w QAT requires the pinned torchao==0.15.0 wheel")
        if torchao.__version__ != "0.15.0":
            raise RuntimeError(
                f"8da4w QAT requires torchao==0.15.0, found {torchao.__version__}"
            )
        self.group_size = int(group_size)
        self.fake_quant_enabled = True
        self.activation_fake_quantizer = IntxFakeQuantizer(
            IntxFakeQuantizeConfig(
                torch.int8, "per_token", is_symmetric=False
            )
        )
        self.weight_fake_quantizer = IntxFakeQuantizer(
            IntxFakeQuantizeConfig(
                torch.int4, group_size=self.group_size, is_symmetric=True
            )
        )

    @classmethod
    def from_linear(cls, linear: Linear, *, group_size: int) -> "QATLinear":
        converted = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            group_size=group_size,
        )
        converted.weight = linear.weight
        if linear.bias is not None:
            converted.bias = linear.bias
        return converted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Evaluation always measures the deployable fake-quantized path.  The
        # start-step switch only controls when quantization enters training.
        use_fake_quant = self.fake_quant_enabled or not self.training
        if not use_fake_quant:
            return super().forward(x)
        xq = self.activation_fake_quantizer(x)
        weight = self.weight
        padded_in = math.ceil(weight.size(1) / self.group_size) * self.group_size
        if padded_in != weight.size(1):
            # TorchAO's group primitive requires complete groups. Quantize the
            # real activation dimensions first, then zero-pad both operands so
            # the tail group has the intended zero-filled semantics.
            weight = F.pad(weight, (0, padded_in - weight.size(1)))
            xq = F.pad(xq, (0, padded_in - xq.size(-1)))
        wq = self.weight_fake_quantizer(weight)
        return F.linear(xq, wq.to(dtype=x.dtype), self.bias)


def soft_colu(x: torch.Tensor, *, dim: int, eps: float = 1e-7) -> torch.Tensor:
    """Explicit-axis CoLU with the jax-colu ``scaling='soft'`` equation."""
    if x.size(-1) % dim != 0:
        raise ValueError(f"channels={x.size(-1)} must be divisible by colu dim={dim}")
    groups = x.size(-1) // dim
    t, w = torch.split(x, [groups, groups * (dim - 1)], dim=-1)
    t = t.unflatten(-1, (groups, 1))
    w = w.unflatten(-1, (groups, dim - 1))
    radius = torch.linalg.vector_norm(w.float(), dim=-1, keepdim=True).to(w.dtype)
    t_out = F.silu(t)
    scale = torch.sigmoid(t_out / (radius + eps) - 0.5)
    return torch.cat((t_out.flatten(-2), (scale * w).flatten(-2)), dim=-1)


class GatedSearchMLP(nn.Module):
    """Parameter-matched gated ReLU or gated soft-CoLU feed-forward block."""

    def __init__(self, config: ComboSearchConfig):
        super().__init__()
        multiple = config.colu_dim if config.search_mlp == "colu" else 1
        width = config.gated_mlp_width
        if width <= 0:
            width = parameter_matched_gated_width(config.n_embd, multiple=multiple)
        if config.search_mlp == "colu" and width % config.colu_dim != 0:
            raise ValueError("gated_mlp_width must be divisible by colu_dim")
        self.variant = config.search_mlp
        self.width = width
        self.colu_dim = config.colu_dim
        # Keep c_fc/c_proj names so the baseline initialization and optimizer
        # grouping remain unchanged; c_up is the gated variant's extra matrix.
        self.c_fc = Linear(config.n_embd, width, bias=False)
        self.c_up = Linear(config.n_embd, width, bias=False)
        self.c_proj = Linear(width, config.n_embd, bias=False)
        self.last_l1 = None
        self.last_active_fraction = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_pre = self.c_fc(x)
        if self.variant == "sparser":
            gate = F.relu(gate_pre)
            self.last_active_fraction = (gate_pre > 0).float().mean()
        elif self.variant == "colu":
            gate = soft_colu(gate_pre, dim=self.colu_dim)
            self.last_active_fraction = (gate.abs() > 2.0**-12).float().mean()
        else:  # pragma: no cover - validated in the config
            raise RuntimeError(f"unsupported gated MLP variant {self.variant}")
        hidden = gate * self.c_up(x)
        # Match the released Sakana training code: sum hidden magnitudes per
        # token, then average tokens/layers.  This is intentionally not divided
        # by the hidden width, so published/released coefficients transfer.
        self.last_l1 = hidden.abs().sum(dim=-1).mean()
        return self.c_proj(hidden)


class ComboSearchGPT(GPT):
    """Nanochat GPT with composable FFN activation and fake-QAT controls."""

    def __init__(self, config: ComboSearchConfig, pad_vocab_size_to: int = 64):
        super().__init__(config, pad_vocab_size_to=pad_vocab_size_to)
        self.config = config
        if config.search_mlp != "baseline":
            for block in self.transformer.h:
                block.mlp = GatedSearchMLP(config)
        self._qat_linears: list[QATLinear] = []
        if config.qat_recipe == "8da4w":
            self._replace_qat_linears(self)
        self.training_step = 0
        self._training_metrics: dict[str, torch.Tensor | float] = {}

    def _replace_qat_linears(self, module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, QATLinear):
                continue
            if isinstance(child, Linear):
                if min(child.in_features, child.out_features) < self.config.qat_min_size:
                    continue
                converted = QATLinear.from_linear(
                    child, group_size=self.config.qat_group_size
                )
                setattr(module, name, converted)
                self._qat_linears.append(converted)
            else:
                self._replace_qat_linears(child)

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        if self.config.search_mlp != "baseline":
            n_embd = self.config.n_embd
            s = 3**0.5 * n_embd**-0.5
            for block in self.transformer.h:
                torch.nn.init.uniform_(block.mlp.c_up.weight, -s * 0.4, s * 0.4)

    def set_training_step(self, step: int) -> None:
        self.training_step = int(step)
        enabled = self.training_step >= self.config.qat_start_step
        for module in self._qat_linears:
            module.fake_quant_enabled = enabled

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        lm_output = super().forward(
            idx, targets=targets, kv_cache=kv_cache, loss_reduction=loss_reduction
        )
        if targets is None or loss_reduction != "mean" or not self.training:
            return lm_output

        metrics: dict[str, torch.Tensor | float] = {
            "train/lm_loss": lm_output.detach(),
            "qat/enabled": float(
                self.config.qat_recipe != "none"
                and self.training_step >= self.config.qat_start_step
            ),
        }
        if self.config.search_mlp != "baseline":
            mlps = [block.mlp for block in self.transformer.h]
            l1 = torch.stack([mlp.last_l1 for mlp in mlps]).mean()
            active = torch.stack([mlp.last_active_fraction for mlp in mlps]).mean()
            metrics["sparser/l1_sum_per_token"] = l1.detach()
            metrics["sparser/active_fraction"] = active.detach()
            metrics["sparser/l1_coeff"] = self.config.sparser_l1_coeff
            self._training_metrics = metrics
            return lm_output + self.config.sparser_l1_coeff * l1

        self._training_metrics = metrics
        return lm_output

    def consume_training_metrics(self) -> dict[str, torch.Tensor | float]:
        metrics = self._training_metrics
        self._training_metrics = {}
        return metrics

    def get_architecture_state(self) -> dict:
        width = None
        if self.config.search_mlp != "baseline":
            width = self.transformer.h[0].mlp.width
        return {
            "family": "combo_search",
            "search_mlp": self.config.search_mlp,
            "gated_mlp_width": width,
            "sparser_l1_coeff": self.config.sparser_l1_coeff,
            "colu_dim": self.config.colu_dim if self.config.search_mlp == "colu" else None,
            "colu_scaling": self.config.colu_scaling if self.config.search_mlp == "colu" else None,
            "qat_recipe": self.config.qat_recipe,
            "qat_group_size": self.config.qat_group_size if self.config.qat_recipe != "none" else None,
            "qat_start_step": self.config.qat_start_step if self.config.qat_recipe != "none" else None,
            "qat_min_size": self.config.qat_min_size if self.config.qat_recipe != "none" else None,
            "qat_linear_count": len(self._qat_linears),
            "training_step": self.training_step,
            "sparse_kernel_backend": "dense_reference",
            "qat_backend": "torchao-0.15.0-fake-quant" if self.config.qat_recipe != "none" else None,
        }


# Paired combinations of independently controlled architecture components.
PARETO_COMPONENTS = frozenset(
    {"qwen_gdn", "exclusive_attention", "xielu", "motif_mhc_anneal", "engram"}
)


def canonical_components(value: str) -> tuple[str, ...]:
    components = tuple(sorted(part.strip() for part in value.split(",") if part.strip()))
    if len(components) < 2:
        raise ValueError("pareto_combo requires at least two components")
    if len(components) != len(set(components)):
        raise ValueError("pareto_components contains a duplicate")
    unknown = set(components) - PARETO_COMPONENTS
    if unknown:
        raise ValueError(f"unknown pareto components: {sorted(unknown)}")
    return components


@dataclass
class ParetoComboConfig(GPTConfig):
    arch_family: str = "pareto_combo"
    pareto_components: str = "qwen_gdn,xielu"
    frontier_extra_lr: float = 0.005
    sconv_kernel_size: int = 4
    mhc_num_streams: int = 4
    mhc_sinkhorn_iterations: int = 20
    mhc_anneal_steps: int = 1907
    engram_layers: tuple[int, ...] = (1, 6)
    engram_ngram_orders: tuple[int, ...] = (2, 3)
    engram_num_heads: int = 8
    engram_dim: int = 0
    engram_vocab_multiplier: int = 5
    engram_kernel_size: int = 4
    engram_seed: int = 0

    def __post_init__(self) -> None:
        components = canonical_components(self.pareto_components)
        self.pareto_components = ",".join(components)
        if self.frontier_extra_lr <= 0 or self.sconv_kernel_size <= 0:
            raise ValueError("invalid Pareto-combo optimizer or convolution setting")
        if self.mhc_num_streams <= 1 or self.mhc_sinkhorn_iterations <= 0 or self.mhc_anneal_steps <= 0:
            raise ValueError("invalid Pareto-combo mHC setting")
        if "engram" in components:
            if "motif_mhc_anneal" in components:
                raise ValueError("Engram and mHC stream replication are not a supported controlled pair")
            if not self.engram_layers or any(not 0 <= layer < self.n_layer for layer in self.engram_layers):
                raise ValueError("invalid Engram injection layers")
            if len(set(self.engram_layers)) != len(self.engram_layers):
                raise ValueError("Engram injection layers must be unique")
            if not self.engram_ngram_orders or min(self.engram_ngram_orders) < 1:
                raise ValueError("invalid Engram n-gram orders")

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.pareto_components.split(","))

    @property
    def sota_variant(self) -> str:
        return "exclusive_attention" if "exclusive_attention" in self.components else "baseline"

    @property
    def frontier_variant(self) -> str:
        # FrontierPoolGPT's generic accounting understands Qwen GDN.  The
        # baseline-like sentinel selects its standard dense estimate otherwise.
        return "qwen_gdn" if "qwen_gdn" in self.components else "inkling_lr2_weight_decay"


class ParetoMLP(nn.Module):
    def __init__(self, config: ParetoComboConfig):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.activation = XIELU() if "xielu" in config.components else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.c_fc(x)
        hidden = self.activation(hidden) if self.activation is not None else F.relu(hidden).square()
        return self.c_proj(hidden)


class ParetoBlock(nn.Module):
    def __init__(self, config: ParetoComboConfig, layer_idx: int):
        super().__init__()
        self.is_gdn = "qwen_gdn" in config.components and layer_idx % 4 != 3
        self.attn = (
            QwenGatedDeltaAttention(config, layer_idx)
            if self.is_gdn
            else SotaAttention(config, layer_idx)
        )
        self.mlp = ParetoMLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        return x + self.mlp(norm(x))


class ParetoComboGPT(FrontierPoolGPT):
    def __init__(self, config: ParetoComboConfig, pad_vocab_size_to: int = 64):
        GPT.__init__(self, config, pad_vocab_size_to=pad_vocab_size_to)
        self.config = config
        self.transformer.h = nn.ModuleList(
            [ParetoBlock(config, layer_idx) for layer_idx in range(config.n_layer)]
        )
        if "motif_mhc_anneal" in config.components:
            self.mhc_connections = nn.ModuleList(
                [MotifMHCConnection(config) for _ in range(2 * config.n_layer)]
            )
        self.engrams = nn.ModuleDict()
        if "engram" in config.components:
            memory_dim = config.engram_dim or config.n_embd // 2
            for layer_idx in config.engram_layers:
                self.engrams[str(layer_idx)] = EngramMemory(
                    hidden_size=config.n_embd,
                    memory_dim=memory_dim,
                    vocab_size=config.vocab_size,
                    vocab_multiplier=config.engram_vocab_multiplier,
                    num_heads=config.engram_num_heads,
                    ngram_orders=config.engram_ngram_orders,
                    layer_idx=layer_idx,
                    seed=config.engram_seed,
                    kernel_size=config.engram_kernel_size,
                    optimizer_world_size=8,
                )
            self.register_buffer("engram_token_map", torch.empty(config.vocab_size, dtype=torch.long))
            self.register_buffer("engram_pad_id", torch.empty((), dtype=torch.long))
        self._training_step = 0
        self._training_metrics = {}

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        width = self.config.n_embd
        bound = math.sqrt(3.0) * width**-0.5
        device = self.transformer.wte.weight.device

        # Match GPT.init_weights' layer-by-layer RNG consumption.  Replaced GDN
        # layers consume scratch baseline Q/K/V draws so every surviving shared
        # MLP and all later shared tensors remain bit-identical to the baseline.
        for block in self.transformer.h:
            if block.is_gdn:
                scratch = torch.empty((width, width), device=device)
                for _ in range(3):
                    torch.nn.init.uniform_(scratch, -bound, bound)
            else:
                for projection in (block.attn.c_q, block.attn.c_k, block.attn.c_v):
                    torch.nn.init.uniform_(projection.weight, -bound, bound)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -bound * 0.4, bound * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        for layer_idx in range(self.config.n_layer):
            self.resid_lambdas.data[layer_idx] = 1.15 - 0.10 * layer_idx / max(self.config.n_layer - 1, 1)
            self.x0_lambdas.data[layer_idx] = 0.20 - 0.15 * layer_idx / max(self.config.n_layer - 1, 1)
        for value_embed in self.value_embeds.values():
            torch.nn.init.uniform_(value_embed.weight, -bound, bound)
        for block in self.transformer.h:
            if block.is_gdn:
                if has_ve(block.attn.layer_idx, self.config.n_layer):
                    scratch = torch.empty((self.config.n_head, 12), device=device)
                    torch.nn.init.uniform_(scratch, 0.0, 0.02)
            elif block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        self.smear_lambda.zero_()
        self.backout_lambda.fill_(0.2)

        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        private_seed = torch.initial_seed() ^ 0x50434F4D
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(private_seed)
            for block in self.transformer.h:
                if block.is_gdn:
                    attn = block.attn
                    torch.nn.init.uniform_(attn.in_proj_qkvz.weight, -bound, bound)
                    torch.nn.init.uniform_(attn.in_proj_ba.weight, -bound, bound)
                    torch.nn.init.kaiming_uniform_(attn.conv.weight, a=math.sqrt(5))
                    attn.a_log.uniform_(0.01, 16.0).log_()
                    attn.dt_bias.fill_(1.0)
                    attn.output_norm_weight.fill_(1.0)
                    torch.nn.init.zeros_(attn.out_proj.weight)
                    if attn.ve_gate is not None:
                        torch.nn.init.uniform_(attn.ve_gate.weight, 0.0, 0.02)
                if block.mlp.activation is not None:
                    block.mlp.activation.reset_parameters()
            for connection in getattr(self, "mhc_connections", []):
                torch.nn.init.xavier_uniform_(connection.mapping_proj.weight)
                connection.alpha.fill_(0.01)
                connection.bias.zero_()

        if self.engrams:
            with torch.random.fork_rng(devices=devices):
                # Preserve the isolated Engram arm's component-private stream.
                torch.manual_seed(torch.initial_seed() ^ 0xE6A)
                for engram in self.engrams.values():
                    engram.init_weights()
            self.engram_token_map.copy_(
                torch.arange(self.config.vocab_size, device=self.engram_token_map.device)
            )
            self.engram_pad_id.zero_()

        self.cos, self.sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, self.config.n_embd // self.config.n_head
        )
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for value_embed in self.value_embeds.values():
                value_embed.to(dtype=COMPUTE_DTYPE)
            for engram in self.engrams.values():
                engram.embedding.to(dtype=COMPUTE_DTYPE)

    @torch.no_grad()
    def configure_engram_token_map(self, token_map: torch.Tensor, pad_id: int) -> None:
        if not self.engrams:
            return
        if token_map.shape != self.engram_token_map.shape:
            raise ValueError("compressed Engram token map has the wrong shape")
        self.engram_token_map.copy_(
            token_map.to(device=self.engram_token_map.device, dtype=torch.long)
        )
        self.engram_pad_id.copy_(self.engram_token_map[int(pad_id)])

    def _trunk(self, idx, kv_cache=None):
        _, time = idx.shape
        if time > self.cos.size(1):
            raise ValueError("sequence exceeds rotary cache")
        offset = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, offset : offset + time], self.sin[:, offset : offset + time]
        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        if kv_cache is None:
            if time <= 1:
                raise ValueError("training forward requires more than one token")
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            raise NotImplementedError("Pareto-combination KV-cache inference is outside this campaign")
        use_mhc = "motif_mhc_anneal" in self.config.components
        compressed_idx = self.engram_token_map[idx] if self.engrams else None
        if use_mhc:
            x = x.unsqueeze(-2).expand(-1, -1, self.config.mhc_num_streams, -1).contiguous()
        x0,backout=x,None
        for layer_idx,block in enumerate(self.transformer.h):
            x = self.resid_lambdas[layer_idx] * x + self.x0_lambdas[layer_idx] * x0
            if str(layer_idx) in self.engrams:
                x = x + self.engrams[str(layer_idx)](x, compressed_idx, self.engram_pad_id)
            ve = self.value_embeds[str(layer_idx)](idx).to(x.dtype) if str(layer_idx) in self.value_embeds else None
            if use_mhc:
                connection=self.mhc_connections[2*layer_idx]
                branch,post,residual=connection.prepare(x)
                output=block.attn(norm(branch),ve,cos_sin,self.window_sizes[layer_idx],None)
                x=connection.combine(x,output,post,residual)
                connection=self.mhc_connections[2*layer_idx+1]
                branch,post,residual=connection.prepare(x)
                output=block.mlp(norm(branch))
                x=connection.combine(x,output,post,residual)
            else:
                x=block(x,ve,cos_sin,self.window_sizes[layer_idx],None)
            if layer_idx==self.config.n_layer//2:
                backout=x
        if backout is not None:
            x=x-self.backout_lambda.to(x.dtype)*backout
        if use_mhc:
            x=x.mean(dim=-2)
        return norm(x),cos_sin

    def set_training_step(self, step: int) -> None:
        self._training_step = int(step)
        if "motif_mhc_anneal" not in self.config.components:
            return
        progress=min(max(step/self.config.mhc_anneal_steps,0.0),1.0)
        for connection in self.mhc_connections:
            connection.set_post_scale(2.0-progress)

    def estimate_flops(self):
        flops=FrontierPoolGPT.estimate_flops(self)
        if self.engrams:
            # Hash-table lookups are not dense matrix multiplies; GPT's generic
            # accounting otherwise counts the table rows as trainable matmuls.
            flops -= 6 * sum(engram.embedding.weight.numel() for engram in self.engrams.values())
        if "exclusive_attention" in self.config.components:
            dense_layers=sum(not block.is_gdn for block in self.transformer.h)
            flops += 12 * dense_layers * self.config.n_embd
        return flops

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        if not self.engrams:
            return super().setup_optimizer(
                unembedding_lr=unembedding_lr,
                embedding_lr=embedding_lr,
                matrix_lr=matrix_lr,
                weight_decay=weight_decay,
                scalar_lr=scalar_lr,
            )
        ddp, _, _, _ = get_dist_info()
        scale = (self.config.n_embd / 768) ** -0.5
        print0(f"Scaling the LR for AdamW parameters by {scale:.6f}")
        engram_embeddings = [engram.embedding.weight for engram in self.engrams.values()]
        engram_convs = [engram.short_conv.weight for engram in self.engrams.values()]
        excluded = {
            id(self.transformer.wte.weight), id(self.lm_head.weight), id(self.resid_lambdas),
            id(self.x0_lambdas), id(self.smear_gate.weight), id(self.smear_lambda),
            id(self.backout_lambda),
            *(id(parameter) for parameter in self.value_embeds.parameters()),
            *(id(parameter) for parameter in engram_embeddings),
            *(id(parameter) for parameter in engram_convs),
        }
        matrices, architecture = [], []
        for parameter in self.parameters():
            if id(parameter) in excluded:
                continue
            (matrices if parameter.ndim == 2 else architecture).append(parameter)
        controls = [self.resid_lambdas, self.x0_lambdas, self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        groups = [
            dict(kind="adamw", params=[self.lm_head.weight], lr=unembedding_lr * scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=[self.transformer.wte.weight], lr=embedding_lr * scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=list(self.value_embeds.parameters()), lr=embedding_lr * scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=[self.resid_lambdas], lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind="adamw", params=[self.x0_lambdas], lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=controls[2:], lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=engram_embeddings, lr=matrix_lr * 5.0, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=engram_convs, lr=matrix_lr * 5.0, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if architecture:
            groups.append(dict(kind="adamw", params=architecture, lr=self.config.frontier_extra_lr, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        for shape in sorted({parameter.shape for parameter in matrices}):
            groups.append(dict(kind="muon", params=[parameter for parameter in matrices if parameter.shape == shape], lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay))
        grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
        if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != {id(parameter) for parameter in self.parameters()}:
            raise RuntimeError("Pareto Engram optimizer grouping mismatch")
        optimizer = (DistMuonAdamW if ddp else MuonAdamW)(groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def get_architecture_state(self) -> dict:
        return {
            "family":"pareto_combo",
            "components":list(self.config.components),
            "shared_backbone":"deterministic_nanochat_d14",
            "shared_initialization":"bit_identical_where_shape_compatible",
            "variant_only_rng":"forked_seed_xor_0x50434f4d",
            "gdn_layer_pattern":"GGGF" if "qwen_gdn" in self.config.components else None,
            "exclusive_attention_layers":[i+1 for i,b in enumerate(self.transformer.h) if not b.is_gdn] if "exclusive_attention" in self.config.components else [],
            "xielu_layers":self.config.n_layer if "xielu" in self.config.components else 0,
            "mhc_streams":self.config.mhc_num_streams if "motif_mhc_anneal" in self.config.components else 0,
            "engram_layers":list(self.config.engram_layers) if self.engrams else [],
            "engram_ngram_orders":list(self.config.engram_ngram_orders) if self.engrams else [],
            "engram_num_heads":self.config.engram_num_heads if self.engrams else 0,
            "current_step":self._training_step,
        }
