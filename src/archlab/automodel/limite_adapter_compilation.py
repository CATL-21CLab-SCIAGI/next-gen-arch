"""Opt-in regional compilation without changing native model state or geometry."""

from functools import cache
from types import MethodType, SimpleNamespace

import torch

from archlab.architectures.limite_bindings import bind_native_forward
from archlab.automodel.limite_adapter_common import make_head_part

STRICT_OPTIONS = {
    "emulate_precision_casts": True,
    "force_same_precision": True,
    "eager_numerics.division_rounding": True,
    # Separate per-module graph pools are prohibitively expensive for this
    # hook-based model. Whole-objective compilation also changes its numerics.
    "triton.cudagraphs": False,
}


@cache
def _native_norm_boundary(norm):
    # This public wrapper leaves the publisher function and namespace intact.
    # Forward/backward retain the native dtype-dependent RMS epsilon/rounding.
    return torch.compiler.disable(norm)


@cache
def _prelude_native_functions(upstream):
    return SimpleNamespace(
        **dict(vars(upstream), rms_norm=_native_norm_boundary(upstream.rms_norm))
    )


def _preserve_block_norms(model):
    """Bind native input/MLP norms as eager boundaries for outer compilation."""
    norm = model.model.adapters[0].upstream.rms_norm
    for layer in model.model.base.layers:
        forward = bind_native_forward(
            layer.forward.__func__,
            (("rms_norm", _native_norm_boundary(norm)),),
            "native_norm_boundary",
        )
        layer.forward = MethodType(forward, layer)
    for adapter in model.model.adapters:
        adapter.upstream = _prelude_native_functions(adapter.upstream)


def configure_compilation(model, mode="none", *, backend="inductor"):
    """Compile native regions once and retain original parameters.

    The hook boundary, native MUDD and data preprocessing remain eager. The
    strict-blocks mode also fuses outer residual operations, with native eager
    input/MLP RMS norms to preserve their exact dtype-dependent epsilon and
    forward/backward rounding. Shared functions retain one binding per region.
    """
    if mode not in ("none", "strict-regional", "strict-blocks"):
        raise ValueError("invalid native compilation mode")
    contract = {"mode": mode}
    if mode == "none":
        return None, contract
    if model.model.adapter_config["variant"] != "normal":
        raise ValueError("regional compilation is qualified only for normal attention")
    torch.compiler.config.recompile_limit = 128
    torch.compiler.config.accumulated_recompile_limit = 1024
    options = dict(STRICT_OPTIONS) if backend == "inductor" else None
    functions = {}

    def compile_region(module):
        forward = module.forward.__func__
        if forward not in functions:
            functions[forward] = torch.compile(
                forward, backend=backend, fullgraph=False, dynamic=False, options=options
            )
        module.forward = MethodType(functions[forward], module)

    if mode == "strict-blocks":
        _preserve_block_norms(model)
        for layer in model.model.base.layers:
            compile_region(layer)
        for adapter in model.model.adapters:
            compile_region(adapter)
    else:
        for layer in model.model.base.layers:
            compile_region(layer.self_attn)
            compile_region(layer.mlp)
        for adapter in model.model.adapters:
            compile_region(adapter.native)

    part = torch.compile(
        make_head_part(model), backend=backend, fullgraph=True, dynamic=False, options=options
    )
    contract.update(
        backend=backend,
        options=options,
        recompile_limit=128,
        accumulated_recompile_limit=1024,
        native_attention_layers=len(model.model.base.layers),
        inserted_attention_layers=len(model.model.adapters),
        native_mlp_layers=len(model.model.base.layers),
        outer_residuals_compiled=mode == "strict-blocks",
        block_norm_boundary=(
            "native eager input/MLP RMSNorm forward and backward; dtype-dependent default epsilon"
            if mode == "strict-blocks" else "unchanged regional boundaries"
        ),
        head="native sigmoid softcap and cross entropy; unchanged checkpoint chunks",
        precision="native parameter/activation dtypes; explicit casts retained; reductions may round differently",
    )
    return part, contract
