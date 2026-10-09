"""Opt-in reuse of the publisher's exact unmasked training context.

The cache contains execution tensors only, never parameters or model buffers.
Preparing it outside capture preserves the native global-attention ``None``
mask, even when Transformers changes its mask-building policy during capture.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NativeTrainingContext:
    key: tuple
    positions: torch.Tensor
    masks: dict[str, torch.Tensor | None]


class NativeTrainingContextCache:
    """One exact native context, disabled until the execution adapter opts in."""

    def __init__(self):
        self.enabled = False
        self.entry: NativeTrainingContext | None = None
        self.locked = False

    def enable(self, enabled=True):
        self.enabled = bool(enabled)
        if not enabled:
            self.invalidate()
            self.locked = False

    def invalidate(self):
        self.entry = None

    def lock(self, locked=True):
        """Reject cold keys before capture; also permits CPU boundary tests."""
        self.locked = bool(locked)

    def prepare(
        self,
        ids: torch.Tensor,
        key: tuple,
        dtype: torch.dtype,
        build_masks: Callable,
    ) -> NativeTrainingContext:
        if not self.enabled:
            raise RuntimeError("enable static native training context before preparation")
        if self.entry is not None and self.entry.key == key:
            return self.entry
        capturing = self.locked or (ids.is_cuda and torch.cuda.is_current_stream_capturing())
        if capturing:
            raise RuntimeError(
                "prepare native training context before CUDA capture; cache key changed"
            )
        positions = torch.arange(ids.shape[1], device=ids.device)[None].expand(ids.shape[0], -1)
        # Native mask construction only reads shape/device/dtype; retain the
        # actual token embedding lookup and its original forward hooks.
        shape = torch.empty((*ids.shape, 0), device=ids.device, dtype=dtype)
        masks = build_masks(shape, positions, None, None)
        self.entry = NativeTrainingContext(key, positions, masks)
        return self.entry


def native_training_context_key(ids, base, training, trainable_mode):
    """Track the native mask geometry and execution mode without tensor reads."""
    config = base.config
    return (
        tuple(ids.shape),
        ids.device.type,
        ids.device.index,
        base.embed_tokens.weight.dtype,
        training,
        base.training,
        trainable_mode,
        config._attn_implementation,
        config.sliding_window,
        tuple(config.layer_types),
        tuple(config.global_layers),
    )
