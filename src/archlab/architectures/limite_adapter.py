"""Native Limite attention preludes before its 48 frozen attention/MLP blocks.

The normal branch calls the unchanged publisher attention. The simplicial
branch substitutes only its attention reduction, keeping native QKV/O, scaling,
QK normalization, RoPE/NoPE, value embeddings, XSA and attention gates.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import asdict, dataclass
from functools import cache
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from archlab.architectures.limite_bindings import bind_native_forward
from archlab.architectures.limite_context import (
    NativeTrainingContextCache,
    native_training_context_key,
)


@dataclass(frozen=True)
class LimiteAdapterConfig:
    variant: str = "normal"
    short_window: int = 16
    seed: int = 42
    attention_backend: str = "native"

    def __post_init__(self):
        if self.variant not in ("normal", "simplicial") or self.short_window < 1:
            raise ValueError("invalid adapter variant/window")
        if self.attention_backend not in ("native", "tilelang"):
            raise ValueError("invalid adapter attention backend")


class _AdapterInterface:
    def __init__(self, reduction):
        self.reduction = reduction

    def get_interface(self, name, fallback):
        return self.reduction


@cache
def _attention_forward_with_reduction(forward, reduction):
    """Bind a local reduction without duplicating Dynamo's function namespace.

    Dynamo associates generated globals with a function's code. Sharing the
    publisher's code between separately copied namespaces can therefore reuse a
    compiled function name that is absent from another adapter's globals. Each
    reduction gets distinct code, and all its adapters share one namespace.
    Only the attention interface changes; the publisher's bytecode is unchanged.
    """
    return bind_native_forward(
        forward,
        (("ALL_ATTENTION_FUNCTIONS", _AdapterInterface(reduction)),),
        f"{forward.__module__}_{reduction.__module__}_{reduction.__qualname__}",
    )


def _simplicial_reduce(module, q, k, v, mask, *, scaling, **kwargs):
    # Native interface uses [B,H,N,D], while the tested kernel uses [B,N,H,D].
    q, k, v = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
    k1, v1 = module._short_values
    q = q.float() * (scaling * q.shape[-1] ** 0.5)
    if q.shape[1] == 1:
        from archlab.architectures.simplicial_packed import packed_simplicial_decode

        out = packed_simplicial_decode(q, k1.float(), k.float(), v1.float(), v.float())
    else:
        from archlab.architectures.simplicial_packed import packed_simplicial_attention

        long = q.shape[1] if module.is_global else module.window_span
        out = packed_simplicial_attention(
            q,
            k1,
            k,
            v1,
            v,
            min(module._short_window, long),
            long,
        )
    return out.to(v.dtype), None


def _tilelang_reduce(module, q, k, v, mask, *, scaling, **kwargs):
    from archlab.architectures.tilelang_attention import tilelang_attention, tilelang_decode

    q, k, v = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
    short = getattr(module, "_short_values", None)
    long = k.shape[1] if module.is_global else module.window_span
    dynamic_length = getattr(module, "_archlab_dynamic_attention", False)
    if q.shape[1] == 1 and k.shape[1] != 1:
        out = tilelang_decode(
            q, k, v, scaling=scaling, short=short,
            lengths=getattr(module, "_archlab_decode_lengths", None),
        )
    elif getattr(module, "_archlab_normal_backward", "tilelang") == "fa4" and not dynamic_length:
        from archlab.architectures.fa4_attention import native_bf16_gqa_attention
        from archlab.architectures.tilelang_attention import normal_attention_forward

        out = native_bf16_gqa_attention(
            q, k, v, scaling=scaling, long_window=long, forward=normal_attention_forward
        )
    else:
        out = tilelang_attention(
            q,
            k,
            v,
            scaling=scaling,
            long_window=long,
            short=short,
            short_window=getattr(module, "_short_window", 1),
            normal_kernel=getattr(module, "_archlab_gqa_kernel", "shared"),
            dynamic_length=dynamic_length,
        )
    return out.to(v.dtype), None


def enable_runtime_sequence_attention(model):
    """Reuse unpadded prefill/replay kernels across RL sequence lengths.

    Fixed-window finetuning retains its qualified static specialization. This
    changes launch metadata only: masks, BF16 operands and accumulation stay
    with the existing shared attention implementation.
    """
    if model.model.adapter_config["attention_backend"] != "tilelang":
        raise ValueError("runtime sequence attention requires the TileLang backend")
    for adapter in model.model.adapters:
        adapter.native._archlab_dynamic_attention = True


def set_normal_attention_kernel(model, kernel):
    """Select execution kernels without changing saved adapter geometry."""
    if kernel not in ("shared", "gqa"):
        raise ValueError("normal attention kernel must be shared or gqa")
    if kernel != "gqa" and getattr(model.model, "normal_backward", "tilelang") == "fa4":
        raise ValueError("select TileLang backward before leaving the FA4 GQA configuration")
    config = model.model.adapter_config
    if kernel != "shared" and (
        config["variant"] != "normal" or config["attention_backend"] != "tilelang"
    ):
        raise ValueError("gqa kernels require the normal TileLang variant")
    for adapter in model.model.adapters:
        adapter.native._archlab_gqa_kernel = kernel
    model.model.normal_kernel = kernel
    model.archlab_normal_attention_kernel = kernel


def set_normal_attention_backward(model, backward):
    """Select a qualified native reduction while retaining the generic GQA API."""
    if backward not in ("tilelang", "fa4"):
        raise ValueError("normal attention backward must be tilelang or fa4")
    backbone = model.model
    if backward == "fa4":
        if (
            backbone.adapter_config["variant"] != "normal"
            or backbone.adapter_config["attention_backend"] != "tilelang"
            or backbone.normal_kernel != "gqa"
        ):
            raise ValueError("FA4 backward requires the normal TileLang GQA configuration")
        from archlab.architectures.fa4_attention import fa4_runtime_contract

        model.archlab_normal_attention_backward_contract = fa4_runtime_contract()
    else:
        model.archlab_normal_attention_backward_contract = {"backend": "tilelang"}
    for adapter in backbone.adapters:
        adapter.native._archlab_normal_backward = backward
    backbone.normal_backward = backward
    model.archlab_normal_attention_backward = backward


class AttentionPrelude(nn.Module):
    def __init__(self, native_layer, config, index):
        super().__init__()
        self.config, self.index = config, index
        cls = type(native_layer)
        self.upstream = sys.modules[cls.__module__]
        native_config = copy.deepcopy(native_layer.config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + index)
            self.native = cls(native_config, index)
            nn.init.normal_(self.native.qkv_proj.weight, std=0.02)
            nn.init.zeros_(self.native.o_proj.weight)
            # Same RNG stream for both variants; the control omits only K1/V1.
            short = nn.Linear(native_config.hidden_size, 2 * self.native.kv_size, bias=False)
            nn.init.normal_(short.weight, std=0.02)
        if self.native.attention_dropout != 0:
            raise ValueError("comparison requires native zero dropout")
        if config.variant == "simplicial":
            self.short_kv = short
            self.register_buffer("_inference_short_weight", None, persistent=False)
            self.native._short_window = config.short_window
        if config.variant == "simplicial" or config.attention_backend == "tilelang":
            # Select only this adapter's reduction; publisher and frozen modules
            # retain their original code and attention implementation.
            reduction = (
                _tilelang_reduce if config.attention_backend == "tilelang" else _simplicial_reduce
            )
            bound = _attention_forward_with_reduction(cls.forward, reduction)
            self.native.forward = MethodType(bound, self.native)
            self.native.config._attn_implementation = "archlab_adapter"
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Publisher eval folds in the stored matrix dtype. Adapter masters are
        # FP32, but native projection arithmetic casts to BF16 activations.
        # Cache that exact cast once per rollout instead of on every token.
        if not mode:
            for name in (
                "_inference_qkv_weight",
                "_inference_o_weight",
                "_inference_ve_gate",
                "_inference_attn_gate",
            ):
                value = getattr(self.native, name)
                if value is not None:
                    setattr(self.native, name, value.to(torch.bfloat16))
        if self.config.variant == "simplicial":
            self._inference_short_weight = (
                None if mode else self.short_kv.weight.detach().to(torch.bfloat16)
            )
        return self

    def forward(self, hidden, context):
        u = self.upstream
        n = self.native
        x = u.rms_norm(hidden)
        cosine, sine = context["rotary"]
        cache = context["cache"]
        n._archlab_decode_lengths = (
            cache.decode_lengths(n.is_global) if hasattr(cache, "decode_lengths") else None
        )
        if self.config.variant == "simplicial":
            short_weight = (
                self.short_kv.weight.to(x.dtype) if self.training else self._inference_short_weight
            )
            short = F.linear(x, short_weight)
            k1, v1 = [
                v.view(*x.shape[:2], n.num_kv_heads, n.head_dim) for v in short.split(n.kv_size, -1)
            ]
            k1 = u.rms_norm(k1)
            if n.applies_rope:
                k1 = u.apply_rotary(k1, cosine, sine)
            if cache is not None:
                if hasattr(cache, "update_short"):
                    k1, v1 = cache.update_short(k1, v1, self.index)
                else:
                    stored = cache.archlab_short.get(self.index)
                    if stored is not None:
                        if x.shape[1] != 1:
                            raise ValueError("cached continuation must have one token")
                        k1, v1 = [
                            torch.cat((old, new), 1)[:, -self.config.short_window :]
                            for old, new in zip(stored, (k1, v1), strict=True)
                        ]
                    cache.archlab_short[self.index] = [
                        t[:, -self.config.short_window :].detach() for t in (k1, v1)
                    ]
            n._short_values = (k1, v1)
        mask = context["masks"]["full_attention" if n.is_global else "sliding_attention"]
        delta, _ = n(x, context["values"], cosine, sine, mask, cache, cache is not None, False)
        return (hidden.float() + delta.float()).to(hidden.dtype)


class PreludeBackbone(nn.Module):
    def __init__(self, base, config):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleList(
            [AttentionPrelude(layer.self_attn, config, i) for i, layer in enumerate(base.layers)]
        )
        self.adapter_config = dict(
            **asdict(config),
            native_shape=dict(
                width=base.config.hidden_size,
                query_heads=base.config.num_attention_heads,
                kv_heads=base.config.num_key_value_heads,
                head_dim=base.config.head_dim,
                local_span=base.config.sliding_window,
                global_layers=base.config.global_layers,
                softmax_scale=base.config.attention_softmax_scale,
            ),
            placement="embedding prelude before native input norm; subsequent preludes before next MUDD/taps",
        )
        self._context = None
        self._bounds = None
        self._native_context_cache = NativeTrainingContextCache()
        self.trainable_mode = "adapter"
        self.normal_kernel = "shared"
        self.normal_backward = "tilelang"
        self._handles = [base.embed_tokens.register_forward_hook(self._embedding)]
        for i, layer in enumerate(base.layers[:-1]):
            self._handles.append(layer.register_forward_hook(self._after(i + 1)))

    @property
    def config(self):
        return self.base.config

    @property
    def embed_tokens(self):
        return self.base.embed_tokens

    def _apply_adapter(self, index, hidden):
        if self._bounds is None:
            return self.adapters[index](hidden, self._context)
        if self._context["cache"] is not None:
            raise ValueError("generation requires unpadded prompt groups")
        outputs = []
        for row, (left, right) in enumerate(self._bounds):
            x = hidden[row : row + 1, left:right]
            positions = torch.arange(right - left, device=x.device)[None]
            rotary = self.base.rotary_emb(positions)
            values = self._context["values"]
            context = dict(
                rotary=rotary,
                cache=None,
                values=values[row : row + 1, left:right] if values is not None else None,
                masks=self._masks(x, positions, None, None),
            )
            outputs.append(
                F.pad(self.adapters[index](x, context), (0, 0, left, hidden.shape[1] - right))
            )
        return torch.cat(outputs)

    def _embedding(self, module, args, output):
        return self._apply_adapter(0, output)

    def _after(self, index):
        def apply(module, args, output):
            return (self._apply_adapter(index, output[0]), *output[1:])

        return apply

    def train(self, mode=True):
        if self.training != mode:
            self._native_context_cache.invalidate()
        super().train(mode)
        self.base.train(mode if self.trainable_mode == "full" else False)
        return self

    def enable_static_training_context(self, enabled=True):
        """Opt in without changing parameter objects, state keys or hooks."""
        self._native_context_cache.enable(enabled)

    def prepare_static_training_context(self, ids):
        if not self.training:
            raise RuntimeError("static native context requires training mode")
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2:
            raise ValueError("static native context requires two-dimensional input_ids")
        key = native_training_context_key(ids, self.base, self.training, self.trainable_mode)
        return self._native_context_cache.prepare(
            ids, key, self.base.embed_tokens.weight.dtype, self._masks
        )

    def lock_static_training_context(self, locked=True):
        self._native_context_cache.lock(locked)

    def _masks(self, hidden, positions, mask, cache):
        if hasattr(cache, "decode_masks"):
            return cache.decode_masks()
        u = self.adapters[0].upstream
        kw = dict(
            config=self.base.config,
            inputs_embeds=hidden,
            attention_mask=mask,
            past_key_values=cache,
            position_ids=positions,
        )
        if cache is not None and hidden.shape[1] == 1:
            return dict(full_attention=None, sliding_attention=None)
        return dict(
            full_attention=u.create_causal_mask(**kw),
            sliding_attention=u.create_sliding_window_causal_mask(**kw),
        )

    def forward(self, *args, **kwargs):
        from transformers import DynamicCache

        ids = kwargs.get("input_ids", args[0] if args else None)
        use_cache = kwargs.get("use_cache", True)
        cache = kwargs.get("past_key_values")
        native_cache = None
        if use_cache:
            if cache is None:
                cache = DynamicCache(config=self.base.config)
                kwargs["past_key_values"] = cache
            if not hasattr(cache, "archlab_native_preludes"):
                cache.archlab_native_preludes = DynamicCache(config=self.base.config)
                cache.archlab_native_preludes.archlab_short = {}
            native_cache = cache.archlab_native_preludes
        mask = kwargs.get("attention_mask")
        positions = kwargs.get("position_ids")
        static_context = None
        if (
            self._native_context_cache.enabled
            and self.training
            and use_cache is False
            and cache is None
            and mask is None
            and positions is None
            and kwargs.get("inputs_embeds") is None
            and len(args) <= 1
            and isinstance(ids, torch.Tensor)
            and ids.ndim == 2
        ):
            static_context = self.prepare_static_training_context(ids)
            positions = static_context.positions
        if positions is None:
            if isinstance(mask, torch.Tensor):
                positions = (mask.long().cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1] :]
            else:
                offset = cache.get_seq_length() if cache is not None else 0
                positions = torch.arange(offset, offset + ids.shape[1], device=ids.device)[
                    None
                ].expand(ids.shape[0], -1)
        # Shapes/dtypes suffice for upstream mask construction; avoid a duplicate
        # token embedding lookup and keep the actual native lookup unchanged.
        if static_context is None:
            shape = torch.empty(
                (*ids.shape, 0), dtype=self.base.embed_tokens.weight.dtype, device=ids.device
            )
            masks = self._masks(shape, positions, mask, native_cache)
        else:
            masks = static_context.masks
        self._context = dict(
            rotary=self.base.rotary_emb(positions),
            values=self.base._value_embeddings(ids),
            cache=native_cache,
            masks=masks,
        )
        self._bounds = None
        known_unpadded = (
            cache is not None
            and bool(getattr(cache, "archlab_no_padding", False))
            and ids.shape[1] == 1
        )
        if (
            isinstance(mask, torch.Tensor)
            and mask.ndim == 2
            and not known_unpadded
            and not bool(mask.all())
        ):
            self._bounds = []
            for row in mask:
                p = row.nonzero().flatten()
                if not len(p):
                    raise ValueError("empty row")
                left, right = int(p[0]), int(p[-1]) + 1
                if not bool(row[left:right].all()):
                    raise ValueError("noncontiguous mask")
                self._bounds.append((left, right))
        if cache is not None and self._bounds is None:
            cache.archlab_no_padding = True
        if len(args) <= 1 and (cache is None or hasattr(cache, "decode_masks")):
            # The publisher accepts precomputed masks. Reuse the same native
            # mapping and positions for uncached passes. Cached generation
            # keeps the publisher's original unpadded-cache classification.
            kwargs = dict(kwargs, attention_mask=masks, position_ids=positions)
        return self.base(*args, **kwargs)


def install_adapters(model, config):
    if isinstance(model.model, PreludeBackbone):
        raise ValueError("adapters already installed")
    model.requires_grad_(False)
    device = next(model.parameters()).device
    model.model = PreludeBackbone(model.model, config).to(device=device)
    model.train()
    return model


def adapter_state(model):
    return {k: v.detach().cpu() for k, v in model.model.adapters.state_dict().items()}


def set_trainable_mode(model, mode):
    """Preserve native parameter dtypes while selecting adapter or full updates."""
    if mode not in ("adapter", "full"):
        raise ValueError("invalid Limite trainable mode")
    model.requires_grad_(mode == "full")
    model.model.adapters.requires_grad_(True)
    model.model.trainable_mode = mode
    # Native evaluation caches contain detached folded projections. Rebuild or
    # remove them after loading changed backbone weights and switching scope.
    model.train()
    return model


def backbone_state(model):
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if not name.startswith("model.adapters.")
    }
