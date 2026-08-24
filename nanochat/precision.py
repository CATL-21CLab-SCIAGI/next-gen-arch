"""Precision backend selection for controlled training comparisons."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import importlib
from typing import Any

import torch


def is_blackwell_gpu_name(device_name: str) -> bool:
    name = device_name.lower()
    return any(token in name for token in ("b300", "b200", "b100", "blackwell", "gb300", "gb200"))


def _import_transformer_engine():
    te = importlib.import_module("transformer_engine.pytorch")
    recipe = importlib.import_module("transformer_engine.common.recipe")
    return te, recipe


def _availability_result(result) -> tuple[bool, str]:
    if isinstance(result, tuple):
        ok = bool(result[0])
        reason = str(result[1]) if len(result) > 1 else ""
        return ok, reason
    return bool(result), ""


def _probe_availability(module: Any, fn_name: str) -> tuple[bool, str]:
    fn = getattr(module, fn_name, None)
    if fn is None:
        return True, ""
    for kwargs in ({"return_reason": True}, {}):
        try:
            return _availability_result(fn(**kwargs))
        except TypeError:
            continue
    return _availability_result(fn())


@dataclass
class PrecisionBackend:
    precision_recipe: str
    runtime_backend: str
    reason: str
    te_module: Any | None = None
    te_recipe: Any | None = None
    te_recipe_name: str | None = None
    requires_materialized_construction: bool = False
    stochastic_rounding: str | None = None
    split_accumulator: str | None = None
    requires_full_context_window: bool = False

    @property
    def uses_transformer_engine(self) -> bool:
        return self.te_module is not None

    def describe_controls(self) -> str:
        controls = []
        if self.te_recipe_name is not None:
            controls.append(f"te_recipe={self.te_recipe_name}")
        if self.stochastic_rounding is not None:
            controls.append(f"stochastic_rounding={self.stochastic_rounding}")
        if self.split_accumulator is not None:
            controls.append(f"split_accumulator={self.split_accumulator}")
        return ", ".join(controls) if controls else "default"

    def training_context(self):
        if not self.uses_transformer_engine:
            return nullcontext()
        return self.te_module.fp8_autocast(enabled=True, fp8_recipe=self.te_recipe)

    def eval_context(self):
        # Eval stays in bf16/fp32 even for low-precision training recipes.
        return nullcontext()

    def collect_debug_metrics(self, model) -> dict[str, float]:
        if not self.uses_transformer_engine:
            return {}
        stats: dict[str, list[float]] = {}
        for name, module in model.named_modules():
            extra_state = None
            if hasattr(module, "get_extra_state"):
                try:
                    extra_state = module.get_extra_state()
                except Exception:
                    extra_state = None
            if extra_state is None and hasattr(module, "_extra_state"):
                extra_state = getattr(module, "_extra_state")
            if extra_state is None:
                continue
            _collect_debug_scalars(stats, extra_state, prefix=name)
        out = {}
        for key, values in stats.items():
            if values:
                out[key] = sum(values) / len(values)
        return out


def _collect_debug_scalars(stats: dict[str, list[float]], value, *, prefix: str):
    if isinstance(value, dict):
        for key, inner in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            _collect_debug_scalars(stats, inner, prefix=child_prefix)
        return
    if isinstance(value, (list, tuple)):
        for i, inner in enumerate(value):
            _collect_debug_scalars(stats, inner, prefix=f"{prefix}/{i}")
        return
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return
    lowered = prefix.lower()
    category = None
    if "amax" in lowered:
        category = "quant/backend_amax"
    elif "scale" in lowered:
        category = "quant/backend_scale"
    elif "overflow" in lowered:
        category = "quant/backend_overflow"
    elif "underflow" in lowered:
        category = "quant/backend_underflow"
    if category is None:
        return
    stats.setdefault(category, []).append(float(value.float().mean().item()))


def is_full_context_window_pattern(window_pattern: str) -> bool:
    pattern = window_pattern.strip().upper()
    return bool(pattern) and set(pattern) == {"L"}


def precision_recipe_requires_full_context_window(precision_recipe: str) -> bool:
    return precision_recipe in {"fp8_full", "fp4_blackwell"}


def resolve_precision_backend(
    precision_recipe: str,
    *,
    device_type: str,
    gpu_name: str | None,
    stochastic_rounding: str = "auto",
    split_accumulator: str = "auto",
) -> PrecisionBackend:
    if precision_recipe == "bf16":
        return PrecisionBackend(
            precision_recipe="bf16",
            runtime_backend="native",
            reason="native bf16/bf32 path",
            stochastic_rounding="n/a",
            split_accumulator="n/a",
        )
    if device_type != "cuda":
        raise RuntimeError(f"{precision_recipe} requires CUDA")
    if not gpu_name:
        raise RuntimeError(f"{precision_recipe} requires a CUDA GPU name for capability checks")
    if stochastic_rounding not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported stochastic_rounding mode: {stochastic_rounding}")
    if split_accumulator not in {"auto", "split", "fast"}:
        raise ValueError(f"Unsupported split_accumulator mode: {split_accumulator}")
    if precision_recipe == "fp8_full":
        if not is_blackwell_gpu_name(gpu_name):
            raise RuntimeError(
                f"fp8_full is reserved for Blackwell MXFP8 runs; found '{gpu_name}'. "
                "Use bf16 on non-Blackwell systems instead of the legacy delayed-scaling fallback."
            )
        if stochastic_rounding != "auto":
            raise RuntimeError(
                "fp8_full uses Transformer Engine MXFP8BlockScaling, which does not expose a public "
                "stochastic-rounding control."
            )
        if split_accumulator != "auto":
            raise RuntimeError(
                "fp8_full uses Transformer Engine MXFP8BlockScaling, which does not expose a public "
                "split-accumulator control."
            )
    if precision_recipe == "fp4_blackwell" and not is_blackwell_gpu_name(gpu_name):
        raise RuntimeError(f"fp4_blackwell requires Blackwell hardware, found '{gpu_name}'")

    try:
        te, recipe = _import_transformer_engine()
    except ImportError as exc:
        raise RuntimeError(
            f"{precision_recipe} requires NVIDIA Transformer Engine to be installed"
        ) from exc

    if precision_recipe == "fp8_full":
        if not hasattr(recipe, "MXFP8BlockScaling"):
            raise RuntimeError("Installed Transformer Engine does not expose MXFP8BlockScaling")
        ok, reason = _probe_availability(te, "is_mxfp8_available")
        if not ok:
            raise RuntimeError(f"MXFP8 is unavailable on this runtime: {reason}")
        format_enum = getattr(recipe, "Format", None)
        mxfp8_kwargs = {}
        fp8_format = getattr(format_enum, "E4M3", None) if format_enum is not None else None
        if fp8_format is not None:
            mxfp8_kwargs["fp8_format"] = fp8_format
        return PrecisionBackend(
            precision_recipe="fp8_full",
            runtime_backend="te_fp8",
            reason="Transformer Engine MXFP8 block scaling (Blackwell-only, full E4M3)",
            te_module=te,
            te_recipe=recipe.MXFP8BlockScaling(**mxfp8_kwargs),
            te_recipe_name="MXFP8BlockScaling",
            requires_materialized_construction=True,
            stochastic_rounding="implicit_mxfp8",
            split_accumulator="implicit_mxfp8",
            requires_full_context_window=True,
        )
    if precision_recipe == "fp4_blackwell":
        if not hasattr(recipe, "NVFP4BlockScaling"):
            raise RuntimeError("Installed Transformer Engine does not expose NVFP4BlockScaling")
        ok, reason = _probe_availability(te, "is_nvfp4_available")
        if not ok:
            raise RuntimeError(f"NVFP4 is unavailable on this runtime: {reason}")
        if split_accumulator != "auto":
            raise RuntimeError(
                "fp4_blackwell uses Transformer Engine NVFP4BlockScaling, which does not expose a public "
                "split-accumulator control."
            )
        disable_stochastic_rounding = False if stochastic_rounding in {"auto", "on"} else True
        return PrecisionBackend(
            precision_recipe="fp4_blackwell",
            runtime_backend="te_fp4",
            reason="Transformer Engine NVFP4 block scaling",
            te_module=te,
            te_recipe=recipe.NVFP4BlockScaling(
                disable_rht=False,
                disable_stochastic_rounding=disable_stochastic_rounding,
                disable_2d_quantization=False,
            ),
            te_recipe_name="NVFP4BlockScaling",
            requires_materialized_construction=True,
            stochastic_rounding="enabled" if not disable_stochastic_rounding else "disabled",
            split_accumulator="n/a_nvfp4",
            requires_full_context_window=True,
        )
    raise ValueError(f"Unsupported precision_recipe: {precision_recipe}")
