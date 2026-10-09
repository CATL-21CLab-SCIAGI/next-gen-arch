"""Use the existing PyTorch GQA API only at native masked decode reductions."""

from functools import cache
from types import MethodType

import torch
from torch.nn import functional as F

from archlab.architectures.limite_bindings import bind_native_forward


class _DecodeInterface:
    def __init__(self, original):
        self.original = original

    def get_interface(self, name, fallback):
        original = self.original.get_interface(name, fallback)
        if name != "sdpa":
            return original

        def reduction(module, q, k, v, mask, *, scaling=None, dropout=0.0, **kwargs):
            if not module.training and getattr(module, "_archlab_decode_gqa", False) and q.shape[2] == 1:
                if (getattr(module, '_archlab_decode_backend', 'sdpa') == 'flash_attn_kvcache'
                        and mask is not None and mask.dtype == torch.bool
                        and mask.shape == (1, 1, 1, k.shape[2])):
                    from flash_attn import flash_attn_with_kvcache

                    # Native graph masks describe either a global prefix or a
                    # chronological local suffix. FA's cache API skips unused
                    # storage using device lengths, without expanding KV heads.
                    count = mask.sum(-1, dtype=torch.int32).reshape(-1)
                    if module.is_global:
                        lengths, leftpad = count, None
                    else:
                        lengths = torch.full_like(count, k.shape[2])
                        leftpad = lengths - count
                    result = flash_attn_with_kvcache(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                        cache_seqlens=lengths.expand(q.shape[0]).contiguous(),
                        cache_leftpad=(leftpad.expand(q.shape[0]).contiguous() if leftpad is not None else None),
                        softmax_scale=scaling, causal=False,
                    )
                    return result, None
                result = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=dropout,
                    scale=scaling, is_causal=False, enable_gqa=True,
                )
                return result.transpose(1, 2).contiguous(), None
            return original(module, q, k, v, mask, scaling=scaling, dropout=dropout, **kwargs)

        return reduction


@cache
def _bound_forward(forward):
    interface = _DecodeInterface(forward.__globals__["ALL_ATTENTION_FUNCTIONS"])
    return bind_native_forward(
        forward, (("ALL_ATTENTION_FUNCTIONS", interface),), "native_masked_decode_gqa",
    )


def set_native_decode_gqa(model, enabled=True, *, backend='sdpa'):
    """Retain publisher prefill/training, geometry and framework implementations."""
    if backend not in ('sdpa', 'flash_attn_kvcache'):
        raise ValueError('unsupported native decode backend')
    backbone = getattr(model.model, "base", model.model)
    for layer in backbone.layers:
        attention = layer.self_attn
        attention._archlab_decode_gqa = bool(enabled)
        attention._archlab_decode_backend = backend
        attention.forward = MethodType(_bound_forward(type(attention).forward), attention)
    model.archlab_native_decode_gqa = bool(enabled)
