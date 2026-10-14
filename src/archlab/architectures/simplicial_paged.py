"""Batched 2-simplicial attention over paged K/V caches (serving form).

Each query token carries its own sequence and absolute position, so one launch
serves prefill, chunked prefill, decode and mixed batches. Short-window (K1/V1)
and long-window (K2/V2) keys are gathered chronologically through a block
table (one per axis: vLLM may place the two caches in different KV-cache groups); the arithmetic and joint online softmax match ``simplicial_decode``.
Launch shapes depend only on tensor metadata, so the call is CUDA-graph safe.
No backward is exposed; training uses ``simplicial_attention``.

Cache layout per branch: [2 (K, V), blocks, block_size, kv_heads, head_dim]
with arbitrary strides (the vLLM FlashAttention logical layout).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged(Q, SC, LC, STABLE, LTABLE, SEQ, POS, OUT, scale,
           sq_t, sq_h, sc_a, sc_b, sc_s, sc_h, lc_a, lc_b, lc_s, lc_h, sst_s, lst_s, so_t, so_h,
           SW, LW, BS,
           HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, H: tl.constexpr,
           T: tl.constexpr, SWP: tl.constexpr):
    token, g = tl.program_id(0), tl.program_id(1)
    seq, pos = tl.load(SEQ + token), tl.load(POS + token)
    h, d, t = tl.arange(0, H), tl.arange(0, D), tl.arange(0, T)
    heads = g * G + h
    q = tl.load(Q + token * sq_t + heads[:, None] * sq_h + d[None, :], h[:, None] < G, other=0)
    acc = tl.full((H, D), 0, tl.float32)
    m = tl.full((H,), -float("inf"), tl.float32)
    z = tl.full((H,), 0, tl.float32)
    short_lo = tl.maximum(pos - SW + 1, 0)
    long_lo = tl.maximum(pos - LW + 1, 0)
    for i in range(SWP):
        j = short_lo + i
        if j <= pos:
            block = tl.load(STABLE + seq * sst_s + j // BS)
            base = block * sc_b + (j % BS) * sc_s + g * sc_h + d
            k1 = tl.load(SC + base)
            v1 = tl.load(SC + sc_a + base).to(tl.float32)
            a = (q.to(tl.float32) * k1[None, :].to(tl.float32)).to(q.dtype)
            for start in range(long_lo, pos + 1, T):
                k = start + t
                valid = k <= pos
                kb = tl.load(LTABLE + seq * lst_s + k // BS, valid, other=0)
                off = kb[:, None] * lc_b + (k[:, None] % BS) * lc_s + g * lc_h + d[None, :]
                k2 = tl.load(LC + off, valid[:, None], other=0)
                v2 = tl.load(LC + lc_a + off, valid[:, None], other=0)
                s = tl.dot(a, tl.trans(k2.to(q.dtype)), input_precision="tf32x3") * scale
                s = tl.where(valid[None, :], s, -float("inf"))
                nm = tl.maximum(m, tl.max(s, 1))
                p = tl.exp(s - nm[:, None])
                alpha = tl.exp(m - nm)
                z = z * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v2.to(q.dtype),
                                                    input_precision="tf32x3") * v1[None, :]
                m = nm
    tl.store(OUT + token * so_t + heads[:, None] * so_h + d[None, :], acc / z[:, None], h[:, None] < G)


def simplicial_paged_attention(q, short_cache, long_cache, block_table, token_seq, token_pos, *,
                               short_window, long_window, scale, long_block_table=None):
    """q [tokens, query_heads, D]; returns FP32 [tokens, query_heads, D].

    ``long_window`` bounds the K2/V2 axis (pass a value >= the longest sequence
    for global layers). ``block_table`` indexes the short cache and, unless
    ``long_block_table`` is given, also the long cache. Block zero may be the
    null block; every slot inside a token's windows must be allocated.
    """
    long_block_table = block_table if long_block_table is None else long_block_table
    if q.ndim != 3 or short_cache.ndim != 5 or long_cache.ndim != 5:
        raise ValueError("expected q [T,H,D] and caches [2,blocks,block,Hk,D]")
    tokens, heads, dim = q.shape
    if (short_cache.shape[0] != 2 or long_cache.shape[0] != 2
            or short_cache.shape[2:] != long_cache.shape[2:]
            or short_cache.shape[-1] != dim or heads % short_cache.shape[3]
            or dim not in (16, 32, 64, 128, 256)):
        raise ValueError("incompatible paged simplicial geometry")
    if not 1 <= short_window <= long_window:
        raise ValueError("require 1 <= short_window <= long_window")
    if (q.device.type != "cuda" or q.dtype not in (torch.float32, torch.bfloat16)
            or any(x.device != q.device for x in (short_cache, long_cache, block_table, long_block_table,
                                                   token_seq, token_pos))
            or any(x.dtype not in (torch.int32, torch.int64)
                   for x in (block_table, long_block_table, token_seq, token_pos))
            or token_seq.shape != (tokens,) or token_pos.shape != (tokens,)
            or block_table.ndim != 2 or long_block_table.ndim != 2):
        raise ValueError("invalid paged simplicial inputs")
    if (q.stride(-1) != 1 or short_cache.stride(-1) != 1 or long_cache.stride(-1) != 1
            or block_table.stride(-1) != 1 or long_block_table.stride(-1) != 1):
        raise ValueError("innermost dimensions must be contiguous")
    kv_heads = short_cache.shape[3]
    group = heads // kv_heads
    if group > 128:
        raise ValueError("at most 128 query heads per KV group")
    out = torch.empty((tokens, heads, dim), device=q.device, dtype=torch.float32)
    if tokens == 0:
        return out
    _paged[(tokens, kv_heads)](
        q, short_cache, long_cache, block_table, long_block_table, token_seq, token_pos, out, float(scale),
        q.stride(0), q.stride(1),
        short_cache.stride(0), short_cache.stride(1), short_cache.stride(2), short_cache.stride(3),
        long_cache.stride(0), long_cache.stride(1), long_cache.stride(2), long_cache.stride(3),
        block_table.stride(0), long_block_table.stride(0), out.stride(0), out.stride(1),
        int(short_window), int(long_window), int(short_cache.shape[2]),
        HK=kv_heads, D=dim, G=group, H=max(16, triton.next_power_of_2(group)), T=32,
        SWP=triton.next_power_of_2(int(short_window)))
    return out
