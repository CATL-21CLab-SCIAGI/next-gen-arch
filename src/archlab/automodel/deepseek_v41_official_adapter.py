"""Attach the reviewed simplicial branch to AutoModel's V4.1 attention write.

The official block calls ``attn_hc.expand`` after attention and before predicting
the FFN coefficients. Extending that instance method leaves the decoder forward,
carried pre-mix, shared attention state, and original checkpoint keys intact.
No upstream source or container package is modified.
"""

from __future__ import annotations

import inspect
from types import MethodType
from functools import partial

import torch
from torch import nn

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter
from archlab.architectures.deepseek_v41_linsimp_adapter import (
    V41LinSimpAdapter,
    V41LinearAttentionAdapter,
    linear_adapter_parameter_count,
    linsimp_adapter_parameter_count,
)
from archlab.architectures.deepseek_v41_normal_adapter import (
    V41NormalAttentionAdapter, normal_adapter_parameter_count,
)

ADAPTER_MARKER = ".simplicial_adapter."
PRODUCTION_LAYER_INDICES = (4, 9, 14, 19, 24, 29, 34, 39)


def _expand_with_adapter(self, output, residual, mix):
    # Keep the original expansion's argument order, precision and orientation.
    streams = self._archlab_original_expand(output, residual, mix)
    if self.adapter_enabled:
        adapted = self.simplicial_adapter(streams)
        if self.identity_observer is not None:
            self.identity_observer(streams, adapted)
        return adapted
    return streams


def _unwrapped_layer(layer):
    # PyTorch's CheckpointWrapper delegates reads, but an attribute assignment on
    # the wrapper would register a branch that the decoder never executes.
    while hasattr(layer, "_checkpoint_wrapped_module"):
        layer = layer._checkpoint_wrapped_module
    return layer


def _check_text_window(module, args, kwargs, *, allow_right_padding=False):
    tokens = kwargs.get("input_ids", args[0] if args else None)
    if tokens is None or tokens.ndim != 2 or min(tokens.shape) < 1:
        raise ValueError("simplicial training requires [batch, sequence] text input_ids")
    unsupported = ("inputs_embeds", "image_mask", "past_key_values", "pixel_values",
                   "cu_seqlens", "seq_lens", "padding_mask")
    if any(kwargs.get(name) is not None for name in unsupported) or kwargs.get("use_cache"):
        raise ValueError("simplicial training supports independent text windows without cache or packing")
    mask = kwargs.get("attention_mask")
    if mask is not None:
        if not allow_right_padding:
            raise ValueError("pass text windows without an attention_mask; tail padding must be unsupervised")
        if mask.shape != tokens.shape or mask.dtype != torch.bool:
            raise ValueError("right-padding masks must be boolean and match input_ids")
        if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("only contiguous valid prefixes and right padding are supported")
    positions = kwargs.get("position_ids")
    if positions is not None:
        expected = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        if positions.shape != tokens.shape or not torch.equal(positions, expected):
            raise ValueError("simplicial training requires positions 0..sequence_length-1")


def install_official_adapters(
    model: nn.Module,
    config: V41AdapterConfig = V41AdapterConfig(),
    *,
    layer_indices: tuple[int, ...] = PRODUCTION_LAYER_INDICES,
    seed: int = 42,
    backend: str = "triton",
    device: torch.device | str = "cpu",
    variant: str = "simplicial",
    allow_right_padding: bool = False,
) -> dict[int, nn.Module]:
    """Freeze the original model and install FP32 adapter masters exactly once.

    Call after loading/sharding the frozen base and before its first forward.
    The default layer selection and geometry give 167,816,256 trainable weights.
    Smaller explicit configurations/selections exist only for numerical tests.
    Returned adapters retain the integer keys used by the existing optimizer and
    adapter-only checkpoint code. Their registration path is
    ``model.layers.N.attn_hc.simplicial_adapter``. The normal control retains
    this registration path for the common checkpoint/insertion boundary; its
    contract and tensor keys identify the distinct 1-simplicial architecture.
    """
    adapters_by_variant = {
        "simplicial": (V41SimplicialAdapter, config.parameter_count()),
        "normal": (V41NormalAttentionAdapter, normal_adapter_parameter_count(config)),
        "linear": (V41LinearAttentionAdapter, linear_adapter_parameter_count(config)),
        "linsimp": (V41LinSimpAdapter, linsimp_adapter_parameter_count(config)),
    }
    if variant not in adapters_by_variant:
        raise ValueError("choose simplicial, normal, linear, or linsimp adapter control")
    adapter_type, expected_parameters = adapters_by_variant[variant]
    if getattr(model, "_archlab_v41_simplicial_installed", False):
        raise ValueError("V4.1 simplicial adapters are already installed")
    text_config = model.config.text_config
    if (config.width, config.streams) != (text_config.hidden_size, text_config.hc_mult):
        raise ValueError("adapter and official backbone residual geometry differ")
    layers = model.model.layers
    if not isinstance(layers, nn.ModuleDict):
        raise TypeError("expected the official V4.1 decoder ModuleDict")
    if (not layer_indices or len(set(layer_indices)) != len(layer_indices)
            or any(type(index) is not int or str(index) not in layers for index in layer_indices)):
        raise ValueError("adapter layers must be distinct existing zero-based decoder indices")
    selected = {index: _unwrapped_layer(layers[str(index)]) for index in layer_indices}
    for layer in selected.values():
        connection = getattr(layer, "attn_hc", None)
        if not isinstance(connection, nn.Module) or not hasattr(layer, "ffn_hc"):
            raise TypeError("expected the official V4.1 attention/FFN hyper-connections")
        if not isinstance(inspect.getattr_static(type(connection), "expand", None), staticmethod):
            raise TypeError("expected the official static attention expansion interface")
        if hasattr(connection, "simplicial_adapter") or "expand" in connection.__dict__:
            raise ValueError("attention expansion has already been extended")
        if getattr(connection, "streams", None) != config.streams:
            raise ValueError("attention connection and adapter stream geometry differ")

    # Construct everything before touching the live model. Adapter initialization
    # restores the RNG; its master precision must not inherit a BF16 global default.
    adapters = {}
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        for index in selected:
            with torch.device("cpu"):
                adapter = adapter_type(config, seed=seed + index, backend=backend)
            adapters[index] = adapter.to(device=device, dtype=torch.float32)
    finally:
        torch.set_default_dtype(old_dtype)

    original = dict(model.named_parameters())
    model.requires_grad_(False)
    for index, layer in selected.items():
        connection = layer.attn_hc
        connection._archlab_original_expand = connection.expand
        connection.simplicial_adapter = adapters[index]
        connection.simplicial_adapter.train(model.training)
        connection.adapter_enabled = True
        connection.identity_observer = None
        connection.expand = MethodType(_expand_with_adapter, connection)
    after = dict(model.named_parameters())
    if any(after.get(name) is not parameter for name, parameter in original.items()):
        raise RuntimeError("adapter insertion changed an original parameter or checkpoint key")
    trainable = [parameter for parameter in after.values() if parameter.requires_grad]
    if (sum(parameter.numel() for parameter in trainable) != len(adapters) * expected_parameters
            or any(parameter.dtype != torch.float32 for parameter in trainable)):
        raise RuntimeError("adapter parameter budget or FP32 master precision changed")
    model.model.register_forward_pre_hook(partial(_check_text_window, allow_right_padding=allow_right_padding), with_kwargs=True)
    model._archlab_v41_simplicial_installed = True
    return adapters
