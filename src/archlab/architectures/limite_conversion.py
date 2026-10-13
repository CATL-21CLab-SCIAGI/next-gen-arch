"""Replace Limite's native attention reductions with joint 2-simplicial attention.

The publisher owns Q/K2/V2/O, QK normalization, local RoPE/global NoPE,
value embeddings, XSA, attention gates, residuals, MLPs and MUDD. The new
short axis is K1 = 1 + delta_K1(x), V1 = 1 + delta_V1(x). Its zero-initialized
linear deltas preserve ordinary attention mathematically: identical short
keys duplicate each long-key score, and joint softmax cancels that duplication.
The constant short axis is not normalized or rotated.

This boundary accepts unpadded training, or one full prefill followed by
single-token DynamicCache decoding. Serving and packed documents require
separate integration and qualification. The explicit reference backend is a
CPU/small-test oracle, never a production fallback.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from functools import cache

import torch
from torch import nn
from torch.nn import functional as F

from archlab.architectures.limite_bindings import bind_native_forward
from archlab.architectures.simplicial_attention import reference_simplicial


@dataclass(frozen=True)
class LimiteConversionConfig:
    short_window: int = 16
    backend: str = "packed"

    def __post_init__(self):
        if type(self.short_window) is not int or not 1 <= self.short_window <= 16:
            raise ValueError("conversion short_window must be an integer in [1, 16]")
        if self.backend not in {"reference", "packed", "tilelang"}:
            raise ValueError("conversion backend must be reference, packed or tilelang")


def _validate_mask(mask, q, key_length, long_window):
    """Reject padding/custom masks instead of silently ignoring them in a kernel."""
    if mask is None:
        return
    length = q.shape[1]
    if (not isinstance(mask, torch.Tensor) or mask.ndim != 4
            or mask.shape[0] not in (1, q.shape[0]) or mask.shape[1:] != (1, length, key_length)):
        raise ValueError("conversion requires an unpadded causal attention mask")
    positions = torch.arange(key_length, device=q.device)
    age = positions[-length:, None] - positions[None, :]
    allowed = ((age >= 0) & (age < long_window))[None, None].expand_as(mask)
    if mask.dtype == torch.bool:
        valid = torch.equal(mask, allowed)
    elif mask.is_floating_point():
        # Upstream uses either -inf or the dtype's most negative finite value.
        blocked = torch.isneginf(mask) | (mask == torch.finfo(mask.dtype).min)
        valid = bool(torch.all(torch.where(allowed, mask == 0, blocked)))
    else:
        valid = False
    if not valid:
        raise ValueError("conversion does not support padding or custom attention masks")


def _joint_reduce(module, q, k, v, mask, *, scaling, dropout=0.0, **kwargs):
    if dropout or kwargs.get("output_attentions", False):
        raise ValueError("conversion requires zero dropout and no materialized attention weights")
    q, k, v = [value.transpose(1, 2).contiguous() for value in (q, k, v)]
    config = module._archlab_conversion_config
    short = module._archlab_conversion_short
    decode = module._archlab_conversion_decode
    long_window = k.shape[1] if module.is_global else module.window_span
    _validate_mask(mask, q, k.shape[1], long_window)
    short_window = min(config.short_window, long_window)
    if config.backend == "tilelang":
        from archlab.architectures.tilelang_attention import tilelang_attention, tilelang_decode

        if decode:
            alignment = ({} if module.is_global else dict(
                query_position=module._archlab_conversion_position, long_window=long_window,
            ))
            output = tilelang_decode(q, k, v, scaling=scaling, short=short, **alignment)
        else:
            output = tilelang_attention(
                q, k, v, scaling=scaling, short=short, short_window=short_window,
                long_window=long_window, dynamic_length=True, shared_gradient_dtype=torch.float64,
            )
    else:
        # Existing simplicial kernels apply D**-0.5; retain Limite's own scale.
        query_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        scaled_query = q.to(query_dtype) * (scaling * q.shape[-1] ** 0.5)
        if config.backend == "reference":
            if decode:
                groups = q.shape[2] // k.shape[2]
                k1, k2, v1, v2 = [
                    value.to(query_dtype).repeat_interleave(groups, dim=2)
                    for value in (short[0], k, short[1], v)
                ]
                scores = torch.einsum("bhd,bjhd,bkhd->bhjk", q[:, 0].to(query_dtype), k1, k2) * scaling
                probabilities = scores.flatten(-2).softmax(-1).reshape_as(scores)
                output = torch.einsum("bhjk,bjhd,bkhd->bhd", probabilities, v1, v2)[:, None]
            else:
                output = reference_simplicial(scaled_query, short[0].to(query_dtype),
                                             k.to(query_dtype), short[1].to(query_dtype), v.to(query_dtype),
                                             short_window, long_window)
        else:
            from archlab.architectures.simplicial_packed import (
                packed_simplicial_attention,
                packed_simplicial_decode,
            )

            if decode:
                output = packed_simplicial_decode(
                    scaled_query.float(), short[0].float(), k.float(), short[1].float(), v.float(),
                )
            else:
                output = packed_simplicial_attention(
                    scaled_query.float(), short[0], k, short[1], v, short_window, long_window,
                )
    return output.to(v.dtype), None


class _JointInterface:
    def get_interface(self, name, fallback):
        return _joint_reduce


@cache
def _native_joint_forward(forward):
    return bind_native_forward(
        forward, (("ALL_ATTENTION_FUNCTIONS", _JointInterface()),), "limite_conversion",
    )


class LimiteSimplicialAttention(nn.Module):
    """One replacement attention, retaining every transferred native parameter.

    ``native`` contains the publisher's projections and gates. Its ordinary
    attention reduction is never called: the local binding substitutes the
    joint reduction while retaining the publisher's surrounding computation.
    The decoder still owns exactly one attention residual update and one MLP.
    """

    def __init__(self, native, config: LimiteConversionConfig):
        super().__init__()
        if native.attention_dropout != 0:
            raise ValueError("conversion requires native zero attention dropout")
        if config.backend != "reference" and native.head_dim != 128:
            raise ValueError("GPU conversion kernels require native head_dim=128")
        self.native = native
        self.conversion_config = config
        # Native attention config is otherwise shared with the complete model.
        self.native.config = copy.deepcopy(native.config)
        self.native.config._attn_implementation = "archlab_limite_conversion"
        self.native._archlab_conversion_config = config
        self.native._archlab_conversion_short = None
        self.native._archlab_conversion_decode = False
        self.native._archlab_conversion_position = None
        weight = native.qkv_proj.weight
        self.short_kv_delta = nn.Parameter(weight.new_zeros(2 * native.kv_size, weight.shape[1]))
        self._joint_forward = _native_joint_forward(type(native).forward)

    @property
    def qkv_proj(self):
        # The publisher decoder reads this dtype when folding residual scales.
        return self.native.qkv_proj

    def forward(self, hidden_states, value_embeds, cosine, sine, attention_mask,
                past_key_values, use_cache, output_attentions):
        if not use_cache and past_key_values is not None:
            raise ValueError("conversion does not read past_key_values with use_cache=False")
        cached_length = 0
        short_cache = None
        if use_cache:
            from transformers import DynamicCache

            if not isinstance(past_key_values, DynamicCache) or torch.is_grad_enabled():
                raise ValueError("conversion cached decoding requires DynamicCache without gradients")
            cached_length = past_key_values.get_seq_length(self.native.layer_idx)
            if cached_length and hidden_states.shape[1] != 1:
                raise ValueError("conversion requires one full prefill then single-token cached decoding")
            if not hasattr(past_key_values, "archlab_conversion_short"):
                past_key_values.archlab_conversion_short = {}
            short_cache = past_key_values.archlab_conversion_short
            if cached_length and self.native.layer_idx not in short_cache:
                raise ValueError("conversion cache lacks its short-axis history")
            if self.conversion_config.backend == "tilelang" and not self.native.is_global:
                # Package storage includes the masked keys preceding the BQ4
                # query group. The reducer still attends to window_span keys.
                from transformers.cache_utils import DynamicSlidingWindowLayer

                index = self.native.layer_idx
                if index < len(past_key_values.layers) and past_key_values.layers[index].is_sliding:
                    layer = past_key_values.layers[index]
                    capacity = self.native.window_span + 3
                    if layer.sliding_window != capacity:
                        if cached_length:
                            raise ValueError("conversion cache lacks its masked query-group history")
                        past_key_values.layers[index] = DynamicSlidingWindowLayer(sliding_window=capacity)
        delta = F.linear(hidden_states, self.short_kv_delta.to(hidden_states.dtype))
        shape = (*hidden_states.shape[:2], self.native.num_kv_heads, self.native.head_dim)
        short = tuple(
            1 + branch.reshape(shape) for branch in delta.split(self.native.kv_size, dim=-1)
        )
        if cached_length:
            short = tuple(torch.cat((old, new), dim=1) for old, new in zip(
                short_cache[self.native.layer_idx], short, strict=True,
            ))
            window = min(self.conversion_config.short_window,
                         short[0].shape[1] if self.native.is_global else self.native.window_span)
            short = tuple(value[:, -window:] for value in short)
        self.native._archlab_conversion_short = short
        self.native._archlab_conversion_decode = bool(cached_length)
        self.native._archlab_conversion_position = cached_length if cached_length else None
        try:
            result = self._joint_forward(
                self.native, hidden_states, value_embeds, cosine, sine, attention_mask,
                past_key_values, use_cache, output_attentions,
            )
            if short_cache is not None:
                window = min(self.conversion_config.short_window,
                             short[0].shape[1] if self.native.is_global else self.native.window_span)
                short_cache[self.native.layer_idx] = tuple(value[:, -window:].detach() for value in short)
            return result
        finally:
            self.native._archlab_conversion_short = None
            self.native._archlab_conversion_decode = False
            self.native._archlab_conversion_position = None


def replace_limite_attention(model, config: LimiteConversionConfig):
    """Convert every native layer in place; do not insert layers or freeze weights.

    Source snapshot provenance belongs to the experiment/export receipt. The
    caller must load the original verified base, rather than an adapter lineage.
    Existing parameter objects/dtypes are transferred without reinitialization.
    """
    backbone = model.model
    if hasattr(backbone, "adapters") or not hasattr(backbone, "layers"):
        raise ValueError("conversion requires the original native Limite backbone")
    if hasattr(backbone, "conversion_config") or any(
        isinstance(layer.self_attn, LimiteSimplicialAttention) for layer in backbone.layers
    ):
        raise ValueError("Limite attention was already converted")
    if len(backbone.layers) != backbone.config.num_hidden_layers:
        raise ValueError("native layer inventory differs from its configuration")
    # Validate all sites before changing the first layer.
    for layer in backbone.layers:
        attention = layer.self_attn
        if attention.attention_dropout != 0:
            raise ValueError("conversion requires native zero attention dropout")
        if config.backend != "reference" and attention.head_dim != 128:
            raise ValueError("GPU conversion kernels require native head_dim=128")
    for layer in backbone.layers:
        layer.self_attn = LimiteSimplicialAttention(layer.self_attn, config)
    backbone.conversion_config = config
    model.train(model.training)
    return model


def conversion_metadata(model):
    """Serializable architecture inventory, without asserting source provenance."""
    backbone = model.model
    config = backbone.conversion_config
    if any(not isinstance(layer.self_attn, LimiteSimplicialAttention) for layer in backbone.layers):
        raise ValueError("conversion metadata requires every native attention to be replaced")
    return {
        "format": "archlab-limite-conversion-v1",
        "intervention": "replace_attention",
        "mechanism": "joint-softmax-2-simplicial",
        "config": asdict(config),
        "layers": list(range(len(backbone.layers))),
        "short_axis": "K1=1+linear_delta_K1;V1=1+linear_delta_V1;no_norm;no_rope",
        "initialization": "zero_short_deltas;inherited_native_attention_parameters",
        "tilelang_shared_gradient_dtype": "float64" if config.backend == "tilelang" else None,
        "cached_decode_qualified": False,
        "cached_decode_contract": "unpadded_DynamicCache;full_prefill_then_single_token;no_gradients",
        "native_geometry": {
            "hidden_size": backbone.config.hidden_size,
            "query_heads": backbone.config.num_attention_heads,
            "kv_heads": backbone.config.num_key_value_heads,
            "head_dim": backbone.config.head_dim,
            "global_layers": list(backbone.config.global_layers),
            "local_window_span": backbone.config.sliding_window,
            "softmax_scale": backbone.config.attention_softmax_scale,
        },
    }
