"""Composable sparse-FFN, soft-CoLU, and QAT variants for the d14 search.

The module deliberately keeps nanochat's attention path and residual controls
unchanged.  The two alternate FFNs are approximately parameter matched to the
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

from nanochat.gpt import GPT, GPTConfig, Linear

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
