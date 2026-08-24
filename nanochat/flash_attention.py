"""
Unified Flash Attention interface with automatic FA3/FA4/SDPA switching.

Exports `flash_attn` with a training API compatible with FA3. The module selects:
- Flash Attention 3 on Hopper
- Flash Attention 4 on Blackwell
- PyTorch SDPA as the fallback everywhere else

For KV-cache inference we keep using FA3 when available and fall back to SDPA
otherwise. FA4 KV-cache support is intentionally not wired in until a stable API
is validated for this repo.
"""

import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _get_cuda_capability():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def _get_cuda_arch_family():
    capability = _get_cuda_capability()
    if capability is None:
        return "none"
    major, _ = capability
    if major == 9:
        return "hopper"
    if major >= 10:
        return "blackwell"
    return "other"


def _load_flash_attention_3():
    """Try to load Flash Attention 3 on Hopper (sm90)."""
    if _get_cuda_arch_family() != "hopper":
        return None
    try:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel("varunneal/flash-attention-3").flash_attn_interface
    except Exception:
        return None


def _load_flash_attention_4():
    """Try to load Flash Attention 4 on Blackwell."""
    if _get_cuda_arch_family() != "blackwell":
        return None
    try:
        from flash_attn.cute import flash_attn_func as fa4_flash_attn_func
        return fa4_flash_attn_func
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
_fa4 = _load_flash_attention_4()
HAS_FA3 = _fa3 is not None
HAS_FA4 = _fa4 is not None
HAS_FLASH_ATTENTION = HAS_FA3 or HAS_FA4

# Override for testing: set to "fa3", "fa4", "sdpa", or None (auto)
_override_impl = None


def _resolve_attention_backend():
    """Resolve the training attention backend for the current process."""
    if _override_impl == "fa3":
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return "fa3"
    if _override_impl == "fa4":
        assert HAS_FA4, "Cannot override to FA4: not available on this hardware"
        return "fa4"
    if _override_impl == "sdpa":
        return "sdpa"

    from nanochat.common import COMPUTE_DTYPE
    if COMPUTE_DTYPE != torch.bfloat16:
        return "sdpa"
    if HAS_FA4:
        return "fa4"
    if HAS_FA3:
        return "fa3"
    return "sdpa"


def _resolve_attention_backend_reason():
    """Human-readable explanation of the selected backend."""
    capability = _get_cuda_capability()
    arch = _get_cuda_arch_family()

    if _override_impl is not None:
        return f"override={_override_impl}"

    from nanochat.common import COMPUTE_DTYPE

    if COMPUTE_DTYPE != torch.bfloat16:
        if HAS_FA4:
            return f"Flash Attention 4 is available on {arch}, but COMPUTE_DTYPE={COMPUTE_DTYPE}; using SDPA"
        if HAS_FA3:
            return f"Flash Attention 3 is available on {arch}, but COMPUTE_DTYPE={COMPUTE_DTYPE}; using SDPA"
        return f"COMPUTE_DTYPE={COMPUTE_DTYPE}; using SDPA"
    if HAS_FA4:
        return "Flash Attention 4 available on Blackwell"
    if HAS_FA3:
        return "Flash Attention 3 available on Hopper"
    if arch == "blackwell":
        return "Blackwell detected, but flash_attn.cute.flash_attn_func is unavailable"
    if arch == "hopper":
        return "Hopper detected, but Flash Attention 3 kernels are unavailable"
    if arch == "none":
        return "CUDA unavailable"
    if capability is not None:
        major, minor = capability
        return f"CUDA SM {major}{minor} has no configured Flash Attention backend"
    return "Falling back to SDPA"


def _resolve_use_fa3():
    return _resolve_attention_backend() == "fa3"


def _resolve_use_fa4():
    return _resolve_attention_backend() == "fa4"


def describe_attention_backend(backend=None):
    backend = _resolve_attention_backend() if backend is None else backend
    return {
        "fa3": "Flash Attention 3",
        "fa4": "Flash Attention 4",
        "sdpa": "PyTorch SDPA",
    }.get(backend, backend)


ATTENTION_BACKEND = _resolve_attention_backend()
ATTENTION_BACKEND_REASON = _resolve_attention_backend_reason()
USE_FA3 = ATTENTION_BACKEND == "fa3"
USE_FA4 = ATTENTION_BACKEND == "fa4"


def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are in (B, H, T, D).
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    if Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash attention for training (no KV cache).

    Args:
        q, k, v: tensors of shape (B, T, H, D)
        causal: whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.
    """
    backend = _resolve_attention_backend()
    if backend == "fa3":
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    if backend == "fa4":
        return _fa4(q, k, v, causal=causal, window_size=window_size)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash attention with KV cache for inference.

    FA3 updates k_cache/v_cache in place. FA4 is intentionally not used here yet,
    so non-FA3 backends use the SDPA implementation below.
    """
    backend = _resolve_attention_backend()
    if backend == "fa3":
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()

    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)
    return y_sdpa.transpose(1, 2)


flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
