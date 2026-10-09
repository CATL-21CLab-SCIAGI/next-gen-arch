"""Checkpointed replay of the unchanged publisher Limite layers.

The publisher's native QKV, QK norm, RoPE/NoPE, value embeddings, XSA,
gates, residuals and MUDD functions are retained. Only uncached, unpadded
attention reductions use container-owned cuDNN SDPA at the native full shape.
One boolean local mask is shared across layers; no sequence-squared integer
intermediates are constructed. Non-reentrant checkpoints keep decoder
intermediates out of the retained autograd graph. Tiling and FA4 remain explicit
experimental alternatives because their full-model rounding differs.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import cache, wraps
from types import MethodType

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from archlab.architectures.limite_bindings import bind_native_forward


@dataclass(frozen=True)
class NativeReplayMask:
    """A causal key span including the query; None denotes global attention."""

    window_span: int | None
    allowed: torch.Tensor | None = None


@cache
def _official_flash_attention():
    from archlab.architectures.fa4_attention import fa4_runtime_contract

    fa4_runtime_contract(validate=True)
    from flash_attn.cute.interface import flash_attn_func

    return flash_attn_func


def _native_sdpa(q, k, v, *, scaling, mask):
    # cuDNN is the publisher's resolved CUDA SDPA backend. Restrict dispatch so
    # an unsupported runtime fails rather than silently choosing another
    # rounding path or retaining a quadratic math-attention activation.
    context = sdpa_kernel(SDPBackend.CUDNN_ATTENTION) if q.is_cuda else nullcontext()
    with context:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=mask is None,
            scale=scaling, dropout_p=0.0, enable_gqa=True,
        )


def _shared_window_mask(length, window_span, device):
    if window_span is None or window_span >= length:
        return None
    positions = torch.arange(length, device=device)
    queries, keys = positions[:, None], positions[None, :]
    # Peak temporary storage is two boolean masks. In particular, subtract
    # the scalar window before broadcasting; an NxN int64 age tensor would
    # consume 128 GiB at the publisher's 131072-token context.
    allowed = keys <= queries
    return allowed.logical_and_(keys > queries - window_span)


def _sdpa_tile(q, k, v, *, scaling, window_span, query_start=0, key_start=0):
    mask = None
    if window_span is not None:
        queries = torch.arange(query_start, query_start + q.shape[2], device=q.device)
        keys = torch.arange(key_start, key_start + k.shape[2], device=q.device)
        age = queries[:, None] - keys[None, :]
        mask = (age >= 0) & (age < window_span)
    return _native_sdpa(q, k, v, scaling=scaling, mask=mask)


def _bounded_sdpa(q, k, v, *, scaling, window_span, query_tile=1024):
    if window_span is None or window_span >= q.shape[2]:
        return _sdpa_tile(q, k, v, scaling=scaling, window_span=None)
    outputs = []
    for start in range(0, q.shape[2], query_tile):
        end = min(start + query_tile, q.shape[2])
        key_start = max(0, start - window_span + 1)
        arguments = (q[:, :, start:end], k[:, :, key_start:end], v[:, :, key_start:end])
        kwargs = dict(scaling=scaling, window_span=window_span,
                      query_start=start, key_start=key_start)
        # Recompute each tile separately during backward. The outer decoder
        # checkpoint alone would retain all tile attention intermediates when
        # it reconstructs the layer's backward graph.
        if torch.is_grad_enabled() and any(t.requires_grad for t in arguments):
            outputs.append(checkpoint(_sdpa_tile, *arguments, use_reentrant=False, **kwargs))
        else:
            outputs.append(_sdpa_tile(*arguments, **kwargs))
    return torch.cat(outputs, dim=2)


def replay_attention(q, k, v, *, scaling, window_span, backend="sdpa_native", native_mask=None):
    """Native GQA reduction with a shared descriptor and unchanged full shape.

    Inputs use the publisher [batch, head, token, channel] interface. The
    The explicit SDPA option is a small-shape numerical oracle. Bounded SDPA
    and FA4 are experimental: their full-model parity checks fail despite
    passing individual-layer tests.
    """
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("native replay requires four-dimensional Q/K/V")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:] or q.shape[1] % k.shape[1]:
        raise ValueError("native replay requires matching uncached GQA sequences")
    if window_span is not None and (isinstance(window_span, bool) or window_span < 1):
        raise ValueError("native replay window must include the query")
    if backend == "sdpa_native":
        mask = native_mask
        if mask is None:
            mask = _shared_window_mask(q.shape[2], window_span, q.device)
        return _native_sdpa(q, k, v, scaling=scaling, mask=mask).transpose(1, 2).contiguous()
    if backend == "sdpa_bounded":
        return _bounded_sdpa(q, k, v, scaling=scaling, window_span=window_span).transpose(1, 2).contiguous()
    if backend == "sdpa":
        mask = None
        if window_span is not None:
            positions = torch.arange(q.shape[2], device=q.device)
            age = positions[:, None] - positions[None, :]
            mask = (age >= 0) & (age < window_span)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=mask is None,
            scale=scaling, dropout_p=0.0, enable_gqa=True,
        ).transpose(1, 2).contiguous()
    if backend != "fa4":
        raise ValueError("native replay attention backend must be sdpa_native, sdpa_bounded, fa4 or sdpa")
    if not q.is_cuda or any(t.dtype != torch.bfloat16 or t.device != q.device for t in (q, k, v)):
        raise ValueError("official FA4 native replay requires CUDA BF16 Q/K/V")
    window = (None, None) if window_span is None else (window_span - 1, 0)
    output, _ = _official_flash_attention()(
        *(t.transpose(1, 2).contiguous() for t in (q, k, v)),
        softmax_scale=scaling, causal=True, window_size=window,
    )
    return output


class _ReplayInterface:
    def __init__(self, original, backend):
        self.original, self.backend = original, backend

    def get_interface(self, name, fallback):
        original = self.original.get_interface(name, fallback)

        def reduction(module, q, k, v, mask, *, scaling=None, dropout=0.0, **kwargs):
            if not isinstance(mask, NativeReplayMask):
                return original(module, q, k, v, mask, scaling=scaling, dropout=dropout, **kwargs)
            expected = None if module.is_global else module.window_span
            if dropout or mask.window_span != expected or kwargs.get("output_attentions", False):
                raise ValueError("native replay attention contract changed")
            return replay_attention(q, k, v, scaling=scaling, window_span=expected,
                                    backend=self.backend, native_mask=mask.allowed), None

        return reduction


@cache
def _bound_attention(forward, backend):
    original = forward.__globals__["ALL_ATTENTION_FUNCTIONS"]
    return bind_native_forward(
        forward, (("ALL_ATTENTION_FUNCTIONS", _ReplayInterface(original, backend)),),
        f"native_replay_{backend}",
    )


def _checkpoint_layer(layer):
    original = layer.forward

    @wraps(original)
    def forward(self, *args, **kwargs):
        mask = kwargs.get("attention_mask", args[5] if len(args) > 5 else None)
        if self.training and torch.is_grad_enabled() and isinstance(mask, NativeReplayMask):
            return checkpoint(original, *args, use_reentrant=False, **kwargs)
        return original(*args, **kwargs)

    layer.forward = MethodType(forward, layer)


def enable_native_replay(model, *, attention_backend="sdpa_native", checkpoint_layers=True):
    """Configure a learner after decode binding, preserving all state keys.

    Actor/decode calls retain their existing bound reduction. Uncached replay
    requires padding to have been removed by the execution adapter. This avoids
    silently changing masked tokens or allocating an enormous fallback mask.
    """
    backbone = model.model
    if not getattr(model, "archlab_native_checkpoint", False) or hasattr(backbone, "adapters"):
        raise ValueError("native replay requires an unmodified publisher model topology")
    if getattr(model, "archlab_native_replay", False):
        raise ValueError("native replay already configured")
    if attention_backend not in ("sdpa_native", "sdpa_bounded", "fa4", "sdpa") or not isinstance(checkpoint_layers, bool):
        raise ValueError("invalid native replay execution configuration")
    for layer in backbone.layers:
        attention = layer.self_attn
        if attention.attention_dropout:
            raise ValueError("native replay requires the publisher zero-dropout contract")
        attention.forward = MethodType(
            _bound_attention(attention.forward.__func__, attention_backend), attention,
        )
        if checkpoint_layers:
            _checkpoint_layer(layer)
    original = backbone.forward

    @wraps(original)
    def forward(self, *args, **kwargs):
        if kwargs.get("use_cache") is not False:
            return original(*args, **kwargs)
        if len(args) > 1 or kwargs.get("past_key_values") is not None:
            raise ValueError("native replay requires uncached keyword inputs")
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None or ids.ndim != 2 or kwargs.get("inputs_embeds") is not None:
            raise ValueError("native replay requires two-dimensional token IDs")
        if not 0 < ids.shape[1] <= self.config.max_position_embeddings:
            raise ValueError("native replay sequence exceeds the publisher context")
        mask = kwargs.get("attention_mask")
        if mask is not None and (
            not isinstance(mask, torch.Tensor) or mask.shape != ids.shape or not bool((mask == 1).all())
        ):
            raise ValueError("native replay requires unpadded token rows")
        positions = torch.arange(ids.shape[1], device=ids.device)[None].expand(ids.shape[0], -1)
        supplied = kwargs.get("position_ids")
        if supplied is not None and (supplied.shape != positions.shape or not torch.equal(supplied, positions)):
            raise ValueError("native replay requires contiguous positions starting at zero")
        window_span = int(self.config.sliding_window)
        shared_mask = (_shared_window_mask(ids.shape[1], window_span, ids.device)
                       if attention_backend == "sdpa_native" else None)
        kwargs = dict(kwargs, position_ids=positions, attention_mask={
            "full_attention": NativeReplayMask(None),
            "sliding_attention": NativeReplayMask(window_span, shared_mask),
        })
        return original(*args, **kwargs)

    backbone.forward = MethodType(forward, backbone)
    model.archlab_native_replay = dict(
        attention_backend=attention_backend, checkpoint_layers=checkpoint_layers,
        causal_mask=("implicit global; one shared boolean local mask" if attention_backend == "sdpa_native"
                     else "implicit global; bounded local query tiles" if attention_backend == "sdpa_bounded" else "implicit"),
        local_query_tile=1024 if attention_backend == "sdpa_bounded" else None,
        window="publisher span including current token",
        publisher_functions="owned bindings; unchanged bytecode and model state keys",
    )
    return model
