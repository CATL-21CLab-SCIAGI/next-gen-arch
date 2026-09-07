"""Add simplicial modules without rewriting NeMo's pretrained decoder.

Upstream contract: NVIDIA-NeMo/Automodel a4ce87c003f08b74d68684d3627f6e6048bc0140,
components/models/qwen3_8_flash_next/layers.py. The decoder calls the MoE
HyperConnection's ``mix`` immediately after completing its attention residual.
Only that read is extended; its original parameters, keys and combine remain.
"""

from __future__ import annotations

import torch
from torch import nn

from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.models.qwen3_8_flash_next.layers import (
    Qwen3_8_FlashNextHyperConnection,
)

from archlab.architectures.simplicial_adapter import (
    SimplicialAdapterConfig,
    SimplicialResidualAdapter,
)

UPSTREAM_COMMIT = "a4ce87c003f08b74d68684d3627f6e6048bc0140"
ADAPTER_MARKER = ".simplicial_adapter."


class AdditiveMoERead(Qwen3_8_FlashNextHyperConnection):
    """Reuse an existing HC's parameters and equations, with an added input branch."""

    def __init__(self, original: Qwen3_8_FlashNextHyperConnection, adapter: nn.Module):
        # Do not call the upstream initializer: even original Parameter objects
        # must survive installation, including their FSDP ownership and metadata.
        nn.Module.__init__(self)
        if type(original) is not Qwen3_8_FlashNextHyperConnection or not original.use_combine:
            raise TypeError("expected an unmodified upstream MoE HyperConnection")
        for name in ("hidden_size", "hc_count", "flat_hidden_size", "lowrank_size", "use_combine"):
            setattr(self, name, getattr(original, name))
        for name in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
            self.add_module(name, getattr(original, name))
        self.simplicial_adapter = adapter
        self.adapter_enabled = True
        self.identity_observer = None

    def mix(self, hidden_states: torch.Tensor):
        if self.adapter_enabled:
            adapted = self.simplicial_adapter(hidden_states)
            if self.identity_observer is not None:
                self.identity_observer(hidden_states, adapted)
            hidden_states = adapted
        return super().mix(hidden_states)


def _check_batch(module: nn.Module, args: tuple, kwargs: dict) -> None:
    """The first experiment uses unpadded fixed windows and ordinary text positions."""
    tokens = kwargs.get("input_ids", args[0] if args else None)
    if tokens is None or tokens.ndim != 2:
        raise ValueError("additive training requires [batch, sequence] raw text input_ids")
    unsupported = ("inputs_embeds", "past_key_values", "pixel_values", "pixel_values_videos",
                   "cu_seqlens", "seq_lens", "_qwen3_8_flash_next_cp_context", "padding_mask")
    if any(kwargs.get(key) is not None for key in unsupported) or kwargs.get("use_cache"):
        raise ValueError("this additive integration supports unpadded text with no cache or CP")
    if kwargs.get("attention_mask") is not None:
        raise ValueError("pass unpadded text windows without an attention_mask")
    positions = kwargs.get("position_ids")
    if positions is not None:
        expected = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        if positions.shape != tokens.shape or not torch.equal(positions, expected):
            raise ValueError("only ordinary text positions 0..sequence_length-1 are supported")


def install_simplicial_modules(
    model: nn.Module,
    config: SimplicialAdapterConfig,
    *,
    seed: int = 42,
    backend: str = "triton",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, SimplicialResidualAdapter]:
    """Install once, before the first sharded forward; return only new modules.

    When using EP/FSDP, load and shard the frozen base first, then call this and
    independently fully_shard each returned adapter before any model forward.
    This avoids teaching the pretrained loader to ignore arbitrary missing keys.
    The distributed probe must verify this ordering before production use.
    """
    if getattr(model, "_archlab_simplicial_installed", False):
        raise ValueError("simplicial modules are already installed")
    text_config = model.config.text_config
    if (config.hidden_size, config.residual_streams) != (text_config.hidden_size, text_config.hc_count):
        raise ValueError("adapter and pretrained residual geometry differ")
    layers = model.model.language_model.layers
    # CheckpointWrapper delegates attribute reads, but NOT attribute writes.
    # Installing on the wrapper would register parameters that never execute.
    unwrapped = {name: unwrap_checkpoint_wrapper(layer) for name, layer in layers.items()}
    selected = {name: layer for name, layer in unwrapped.items() if layer.layer_type == "full_attention"}
    if not selected:
        raise ValueError("no full-attention blocks found")
    if any(type(layer.mlp_hyper_connection) is not Qwen3_8_FlashNextHyperConnection
           for layer in selected.values()):
        raise TypeError("unexpected upstream MoE HyperConnection type")
    original = dict(model.named_parameters())
    result = {}
    for name, layer in selected.items():
        with torch.device("cpu"):
            adapter = SimplicialResidualAdapter(config, seed=seed + int(name), backend=backend)
        adapter = adapter.to(device=device, dtype=dtype)
        layer.mlp_hyper_connection = AdditiveMoERead(layer.mlp_hyper_connection, adapter)
        result[name] = adapter
    after = dict(model.named_parameters())
    if any(after.get(name) is not parameter for name, parameter in original.items()):
        raise RuntimeError("adapter insertion changed an original parameter or checkpoint key")
    for name, parameter in after.items():
        parameter.requires_grad_(ADAPTER_MARKER in name)
    model.register_forward_pre_hook(_check_batch, with_kwargs=True)
    model._archlab_simplicial_installed = True
    return result
