"""Candidate vLLM cache bridge for native Limite attention replacement.

Two ordinary package Attention layers own each layer's long and short paged
K/V. Their outputs are discarded; the existing TileLang joint reduction owns
the conversion arithmetic. This narrow eager path admits one complete new
prompt or one decode token, with no request state outside vLLM's caches.
GPU numerical and request lifecycle qualification remain separate.
"""
from __future__ import annotations

from copy import copy

import torch

from archlab.architectures.limite_bindings import bind_native_forward


def validate_conversion_serving(vllm_config):
    """Reject settings whose paging or execution semantics are not implemented."""
    parallel = vllm_config.parallel_config
    if any(getattr(parallel, name, 1) != 1 for name in (
        "tensor_parallel_size", "pipeline_parallel_size",
        "decode_context_parallel_size", "prefill_context_parallel_size",
    )):
        raise ValueError("conversion serving requires TP=PP=DCP=PCP=1")
    scheduler = vllm_config.scheduler_config
    if scheduler.max_num_seqs != 1 or scheduler.enable_chunked_prefill:
        raise ValueError("conversion serving requires max_num_seqs=1 and no chunked prefill")
    if not vllm_config.model_config.enforce_eager:
        raise ValueError("conversion serving requires enforce_eager=True")
    cache = vllm_config.cache_config
    if cache.enable_prefix_caching or cache.cache_dtype not in {"auto", "bfloat16", "float16"}:
        raise ValueError("conversion serving requires no prefix caching and unquantized K/V")
    if vllm_config.quant_config is not None or vllm_config.speculative_config is not None:
        raise ValueError("conversion serving does not admit quantization or speculative decoding")
    attention = vllm_config.attention_config
    backend = getattr(attention.backend, "name", attention.backend)
    if backend != "FLASH_ATTN" or attention.flash_attn_version != 2:
        raise ValueError("conversion serving requires explicit FLASH_ATTN version 2")
    config = vllm_config.model_config.hf_config
    if config.adapter or config.conversion != {"short_window": 16, "backend": "tilelang"}:
        raise ValueError("conversion serving requires the W16 TileLang attention replacement")


def stock_attention_factory(vllm_config):
    """Construct upstream cache owners with exact per-layer native windows."""
    from vllm.model_executor.layers.attention import Attention

    validate_conversion_serving(vllm_config)
    # Attention's None per-layer window otherwise inherits the model-level
    # window, incorrectly narrowing Limite's global NoPE layers.
    cache_config = copy(vllm_config.cache_config)
    cache_config.sliding_window = None

    def factory(native, *, window, prefix):
        attention = Attention(
            num_heads=native.num_heads, head_size=native.head_dim,
            scale=native.scaling, num_kv_heads=native.num_kv_heads,
            cache_config=cache_config, per_layer_sliding_window=window, prefix=prefix,
        )
        if (attention.get_attn_backend().get_name() != "FLASH_ATTN"
                or attention.impl.vllm_flash_attn_version != 2):
            raise ValueError("conversion cache owner must actually use FlashAttention-2")
        return attention

    return factory


def _attention_context(attention):
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention.attention import get_attention_context

    raw = get_forward_context().attn_metadata
    if raw is None:
        # Profiling has no request cache. The package helper requires a slot
        # mapping even in this case, so do not call it on dummy forwards.
        return None, attention, attention.kv_cache, None
    if isinstance(raw, list):
        raise ValueError("conversion serving does not accept speculative attention metadata")
    return get_attention_context(attention.layer_name)


def _validate_context(context, native, tokens, dtype, device):
    metadata, _, cache, slots = context
    if metadata is None:
        return None
    if not all(hasattr(metadata, name) for name in (
        "query_start_loc", "seq_lens", "block_table", "num_actual_tokens",
        "causal", "use_cascade", "common_prefix_len",
    )):
        raise ValueError("conversion serving requires FlashAttention metadata")
    starts = metadata.query_start_loc
    lengths = metadata.seq_lens
    table = metadata.block_table
    integer_dtypes = {torch.int32, torch.int64}
    if (starts.ndim != 1 or starts.numel() != 2 or starts.dtype not in integer_dtypes
            or starts.tolist() != [0, tokens]
            or lengths.ndim != 1 or lengths.numel() != 1 or lengths.dtype not in integer_dtypes
            or table.ndim != 2 or table.shape[0] != 1 or table.dtype not in integer_dtypes
            or metadata.num_actual_tokens != tokens
            or not metadata.causal or metadata.use_cascade or metadata.common_prefix_len):
        raise ValueError("conversion serving requires one causal uncached prompt or decode token")
    length = int(lengths[0].item())
    if length < tokens or (length != tokens and tokens != 1):
        raise ValueError("conversion serving does not accept chunked or cached multi-token prefill")
    if (cache.ndim != 5 or cache.shape[0] != 2 or min(cache.shape) < 1
            or cache.shape[3:] != (native.num_kv_heads, native.head_dim)
            or cache.shape[2] % 16 or cache.dtype != dtype or cache.device != device
            or table.device != device or dtype not in {torch.bfloat16, torch.float16}):
        raise ValueError("conversion requires logical FlashAttention [2,blocks,block,Hkv,D] K/V")
    if (not isinstance(slots, torch.Tensor) or slots.ndim != 1 or slots.numel() != tokens
            or slots.dtype not in integer_dtypes or slots.device != device):
        raise ValueError("conversion serving requires the package's per-layer slot mapping")
    last_block = (length - 1) // cache.shape[2]
    if last_block >= table.shape[1]:
        raise ValueError("conversion block table does not cover the current sequence")
    block = int(table[0, last_block].item())
    # Upstream reserves physical block zero as the null placeholder.
    if not 0 < block < cache.shape[1]:
        raise ValueError("conversion current token has no allocated cache block")
    expected_slot = block * cache.shape[2] + (length - 1) % cache.shape[2]
    if int(slots[-1].item()) != expected_slot:
        raise ValueError("conversion slot mapping disagrees with the current block table")
    return length


def _cache_window(cache, metadata, length, window):
    """Gather chronological logical slots; indexing also respects HND strides."""
    positions = torch.arange(max(0, length - window), length, device=cache.device)
    block_size = cache.shape[2]
    blocks = metadata.block_table[0, positions // block_size].long()
    if bool(((blocks <= 0) | (blocks >= cache.shape[1])).any()):
        raise ValueError("conversion window contains an unallocated cache block")
    offsets = positions % block_size
    return tuple(cache[axis, blocks, offsets].unsqueeze(0).contiguous() for axis in (0, 1))


def _prefill(q, k, v, short, *, scaling, long_window, short_window):
    from archlab.architectures.tilelang_attention import tilelang_attention

    return tilelang_attention(q, k, v, scaling=scaling, short=short,
                              short_window=short_window, long_window=long_window,
                              dynamic_length=True)


def _decode(q, k, v, short, *, scaling):
    from archlab.architectures.tilelang_attention import tilelang_decode

    return tilelang_decode(q, k, v, scaling=scaling, short=short)


def _paged_joint_attention(native, query, key, value, attention_mask, *, scaling,
                           dropout=0.0, **kwargs):
    if (query.ndim != 4 or query.shape[0] != 1 or query.shape[2] < 1 or attention_mask is not None
            or scaling != native.scaling or dropout or kwargs.get("output_attentions", False)):
        raise ValueError("conversion serving requires one unmasked native-scale inference stream")
    q, k, v = (tensor.transpose(1, 2).contiguous() for tensor in (query, key, value))
    short = native._archlab_conversion_short
    if (q.shape[2:] != (native.num_heads, native.head_dim)
            or k.shape != v.shape or k.shape[:2] != q.shape[:2]
            or k.shape[2:] != (native.num_kv_heads, native.head_dim)
            or short is None or any(t.shape != k.shape for t in short)
            or any(t.dtype != q.dtype or t.device != q.device for t in (k, v, *short))):
        raise ValueError("conversion serving received incompatible native projections")
    long_owner = native.archlab_conversion_long_attention
    short_owner = native.archlab_conversion_short_attention
    long_context, short_context = (_attention_context(owner) for owner in (long_owner, short_owner))
    if long_context[1] is not long_owner or short_context[1] is not short_owner:
        raise ValueError("conversion attention context has a different cache owner")
    lengths = [_validate_context(context, native, q.shape[1], q.dtype, q.device)
               for context in (long_context, short_context)]
    if lengths[0] != lengths[1]:
        raise ValueError("conversion long and short cache metadata disagree")
    length = lengths[0]
    long_window = (length or q.shape[1]) if native.is_global else native.window_span
    short_window = min(native._archlab_conversion_config.short_window, long_window)
    if length is not None:
        # Public Attention.forward owns cache writes and request slot mapping.
        # Its normal-attention result is deliberately discarded at this stage.
        flat_q = q.flatten(0, 1)
        long_owner(flat_q, k.flatten(0, 1), v.flatten(0, 1))
        short_owner(flat_q, short[0].flatten(0, 1), short[1].flatten(0, 1))
    if length is None or length == q.shape[1]:
        output = _prefill(q, k, v, short, scaling=scaling,
                          long_window=long_window, short_window=short_window)
    else:
        k, v = _cache_window(long_context[2], long_context[0], length, long_window)
        short = _cache_window(short_context[2], short_context[0], length, short_window)
        output = _decode(q, k, v, short, scaling=scaling)
    return output.to(value.dtype), None


class _JointInterface:
    def get_interface(self, name, fallback):
        return _paged_joint_attention


_JOINT_INTERFACE = _JointInterface()


def bind_conversion_attention(policy, attention_factory, *, prefix=""):
    """Replace only each converted reduction; package caches remain modules."""
    from archlab.architectures.limite_conversion import LimiteSimplicialAttention

    attentions = []
    for index, layer in enumerate(policy.model.layers):
        replacement = layer.self_attn
        if (not isinstance(replacement, LimiteSimplicialAttention)
                or replacement.conversion_config.short_window != 16
                or replacement.conversion_config.backend != "tilelang"):
            raise ValueError("vLLM conversion binding requires W16 TileLang native replacements")
        native = replacement.native
        for axis, window in (("long", native.window_span), ("short", 16)):
            attention = attention_factory(
                native, window=window, prefix=f"{prefix}layers.{index}.conversion.{axis}_attention",
            )
            setattr(native, f"archlab_conversion_{axis}_attention", attention)
            attentions.append(attention)
        replacement._joint_forward = bind_native_forward(
            type(native).forward, (("ALL_ATTENTION_FUNCTIONS", _JOINT_INTERFACE),),
            "archlab_vllm_conversion",
        )
        native.config._attn_implementation = "archlab_paged_conversion"
    return tuple(attentions)
