"""Batched 2-simplicial attention over paged K/V caches (serving form).

Each query token carries its own sequence and absolute position, so one launch
serves prefill, chunked prefill, decode and mixed batches. Short-window (K1/V1)
and long-window (K2/V2) keys are gathered through one block table per axis
(vLLM may place the two caches in different KV-cache groups). The arithmetic is
the TileLang joint softmax of ``tilelang_attention``: BF16 query-short-key
products and probabilities, FP32 scores, normalizers and outputs. Long-axis
splits fill the GPU for decode; their log2 partials are merged exactly.
Launch shapes depend only on tensor metadata, so the call is CUDA-graph safe.
No backward is exposed.

Cache layout per branch: [2 (K, V), blocks, block_size, kv_heads, head_dim]
with arbitrary strides (the vLLM FlashAttention logical layout).
"""

import math

import tilelang
import tilelang.language as T
import torch

_SHORT_SLOTS = 16


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _paged_kernel(HQ, HK, D, W1, W2, BS, scale, GP, BN=64, threads=32, stages=1):
    """Decode form: one program per (token, KV head, long-axis split).

    The G = HQ / HK query heads sharing a KV head are packed as GP >= G groups of 16
    virtual rows, so every gathered K2/V2 tile serves all of them (rows of padding heads
    are masked and never written).
    """
    S = _SHORT_SLOTS
    G = HQ // HK
    M = GP * S
    tokens, splits, requests, short_width, long_width = T.dynamic(
        "tokens, splits, requests, short_width, long_width")
    short_blocks, long_blocks = T.dynamic("short_blocks, long_blocks")
    sb1, ss1, sh1, sb2, ss2, sh2 = T.dynamic("sb1, ss1, sh1, sb2, ss2, sh2", "int64")

    @T.prim_func
    def main(
        Q: T.Tensor([tokens, HQ, D], T.bfloat16),
        K1: T.StridedTensor([short_blocks, BS, HK, D], [sb1, ss1, sh1, 1], T.bfloat16),
        V1: T.StridedTensor([short_blocks, BS, HK, D], [sb1, ss1, sh1, 1], T.bfloat16),
        K2: T.StridedTensor([long_blocks, BS, HK, D], [sb2, ss2, sh2, 1], T.bfloat16),
        V2: T.StridedTensor([long_blocks, BS, HK, D], [sb2, ss2, sh2, 1], T.bfloat16),
        SHORT_TABLE: T.Tensor([requests, short_width], T.int32),
        LONG_TABLE: T.Tensor([requests, long_width], T.int32),
        SEQ: T.Tensor([tokens], T.int32),
        POS: T.Tensor([tokens], T.int32),
        OUT: T.Tensor([splits, tokens, HQ, D], T.float32),
        MAX: T.Tensor([splits, tokens, HQ], T.float32),
        SUM: T.Tensor([splits, tokens, HQ], T.float32),
    ):
        with T.Kernel(tokens, HK, splits, threads=threads) as (t, g, split):
            a = T.alloc_shared([M, D], T.bfloat16)
            vlo = T.alloc_fragment([M, D], T.float32)
            kv = T.alloc_shared([BN, D], T.bfloat16)
            vv = T.alloc_shared([BN, D], T.bfloat16)
            score = T.alloc_fragment([M, BN], T.float32)
            prob = T.alloc_fragment([M, BN], T.bfloat16)
            acc = T.alloc_fragment([M, D], T.float32)
            rowmax = T.alloc_fragment([M], T.float32)
            rowmax2 = T.reshape(rowmax, [GP, S])
            rowz = T.alloc_fragment([M], T.float32)
            rowz2 = T.reshape(rowz, [GP, S])
            mx = T.alloc_fragment([GP], T.float32)
            old = T.alloc_fragment([GP], T.float32)
            current_max = T.alloc_fragment([GP], T.float32)
            z = T.alloc_fragment([GP], T.float32)
            zz = T.alloc_fragment([GP], T.float32)
            output = T.alloc_fragment([GP, D], T.float32)
            result = T.reshape(vlo, [GP, S, D])
            seq = SEQ[t]
            pos = POS[t]
            split_length = T.ceildiv(long_width * BS, splits)
            lo = T.max(T.max(pos - W2 + 1, 0), split * split_length)
            hi = T.min(pos, (split + 1) * split_length - 1)
            for r, d in T.Parallel(M, D):
                j = T.max(pos - r % S, 0)
                block = T.cast(SHORT_TABLE[seq, j // BS], "int64")
                valid = r // S < G and r % S < W1 and pos - r % S >= 0
                a[r, d] = T.if_then_else(
                    valid, Q[t, g * G + T.min(r // S, G - 1), d].astype(T.float32)
                    * K1[block, j % BS, g, d].astype(T.float32), 0)
                vlo[r, d] = T.if_then_else(valid, V1[block, j % BS, g, d], 0)
            T.clear(acc)
            T.clear(z)
            T.fill(mx, -T.infinity(T.float32))
            for tile in T.Pipelined(T.if_then_else(hi >= lo, T.ceildiv(hi - lo + 1, BN), 0), num_stages=stages):
                for k, d in T.Parallel(BN, D):
                    key = T.min(lo + tile * BN + k, hi)
                    block = T.cast(LONG_TABLE[seq, key // BS], "int64")
                    kv[k, d] = K2[block, key % BS, g, d]
                    vv[k, d] = V2[block, key % BS, g, d]
                T.clear(score)
                T.gemm(a, kv, score, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for r, k in T.Parallel(M, BN):
                    score[r, k] = T.if_then_else(
                        r // S < G and r % S < W1 and pos - r % S >= 0 and lo + tile * BN + k <= hi,
                        score[r, k] * scale * 1.4426950408889634,
                        -T.infinity(T.float32),
                    )
                T.reduce_max(score, rowmax, dim=1)
                T.reduce_max(rowmax2, current_max, dim=1)
                T.copy(mx, old)
                for i in T.Parallel(GP):
                    mx[i] = T.if_then_else(i < G, T.max(mx[i], current_max[i]), 0)
                for r, k in T.Parallel(M, BN):
                    score[r, k] = T.exp2(score[r, k] - mx[r // S])
                T.reduce_sum(score, rowz, dim=1)
                T.reduce_sum(rowz2, zz, dim=1)
                for i in T.Parallel(GP):
                    z[i] = z[i] * T.exp2(old[i] - mx[i]) + zz[i]
                for r, d in T.Parallel(M, D):
                    acc[r, d] *= T.exp2(old[r // S] - mx[r // S])
                T.copy(score, prob)
                T.gemm(prob, vv, acc, policy=T.GemmWarpPolicy.FullRow)
            for r, d in T.Parallel(M, D):
                vlo[r, d] *= acc[r, d]
            T.reduce_sum(result, output, dim=1)
            for i, d in T.Parallel(GP, D):
                if i < G:
                    OUT[split, t, g * G + i, d] = output[i, d]
            for i in T.Parallel(GP):
                if i < G:
                    MAX[split, t, g * G + i] = mx[i]
                    SUM[split, t, g * G + i] = z[i]

    return main


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _paged_tiled_kernel(HQ, HK, D, W1, W2, BS, scale, BQ=4, BN=64, threads=128, stages=1):
    """Prefill form: one program per BQ consecutive query tokens of one request.

    The BQ x 16 virtual-query rows share each gathered K2/V2 tile (as in the dense training kernel).
    Program g finds its request in TILES (exclusive prefix sum of per-request tile counts); programs
    past the last tile have no rows. Queries of a request are consecutive tokens with consecutive
    positions, as vLLM packs them.
    """
    S = _SHORT_SLOTS
    M = BQ * S
    tokens, programs, requests, bounds, short_width, long_width = T.dynamic(
        "tokens, programs, requests, bounds, short_width, long_width")
    short_blocks, long_blocks = T.dynamic("short_blocks, long_blocks")
    sb1, ss1, sh1, sb2, ss2, sh2 = T.dynamic("sb1, ss1, sh1, sb2, ss2, sh2", "int64")

    @T.prim_func
    def main(
        Q: T.Tensor([tokens, HQ, D], T.bfloat16),
        K1: T.StridedTensor([short_blocks, BS, HK, D], [sb1, ss1, sh1, 1], T.bfloat16),
        V1: T.StridedTensor([short_blocks, BS, HK, D], [sb1, ss1, sh1, 1], T.bfloat16),
        K2: T.StridedTensor([long_blocks, BS, HK, D], [sb2, ss2, sh2, 1], T.bfloat16),
        V2: T.StridedTensor([long_blocks, BS, HK, D], [sb2, ss2, sh2, 1], T.bfloat16),
        SHORT_TABLE: T.Tensor([requests, short_width], T.int32),
        LONG_TABLE: T.Tensor([requests, long_width], T.int32),
        STARTS: T.Tensor([bounds], T.int32),
        TILES: T.Tensor([bounds], T.int32),
        POS: T.Tensor([tokens], T.int32),
        GRID: T.Tensor([programs], T.int32),
        OUT: T.Tensor([tokens, HQ, D], T.float32),
    ):
        with T.Kernel(programs, HQ, threads=threads) as (g, h):
            a = T.alloc_shared([M, D], T.bfloat16)
            vlo = T.alloc_fragment([M, D], T.float32)
            kv = T.alloc_shared([BN, D], T.bfloat16)
            vv = T.alloc_shared([BN, D], T.bfloat16)
            score = T.alloc_fragment([M, BN], T.float32)
            prob = T.alloc_fragment([M, BN], T.bfloat16)
            acc = T.alloc_fragment([M, D], T.float32)
            rowmax = T.alloc_fragment([M], T.float32)
            rowmax2 = T.reshape(rowmax, [BQ, S])
            rowz = T.alloc_fragment([M], T.float32)
            rowz2 = T.reshape(rowz, [BQ, S])
            mx = T.alloc_fragment([BQ], T.float32)
            old = T.alloc_fragment([BQ], T.float32)
            current_max = T.alloc_fragment([BQ], T.float32)
            z = T.alloc_fragment([BQ], T.float32)
            zz = T.alloc_fragment([BQ], T.float32)
            output = T.alloc_fragment([BQ, D], T.float32)
            result = T.reshape(vlo, [BQ, S, D])
            owner = T.alloc_local([1], T.int32)
            kvh = h // (HQ // HK)
            owner[0] = 0
            for r in T.serial(requests):
                owner[0] = T.if_then_else(TILES[r + 1] <= g, r + 1, owner[0])
            seq = T.min(owner[0], requests - 1)
            first = STARTS[seq] + (g - TILES[seq]) * BQ
            count = T.if_then_else(g < TILES[requests], T.min(BQ, STARTS[seq + 1] - first), 0)
            base = POS[T.min(first, tokens - 1)]
            lo = T.max(base - W2 + 1, 0)
            hi = base + count - 1
            for r, d in T.Parallel(M, D):
                pos = base + r // S
                j = T.max(pos - r % S, 0)
                block = T.cast(SHORT_TABLE[seq, j // BS], "int64")
                valid = r // S < count and r % S < W1 and pos - r % S >= 0
                a[r, d] = T.if_then_else(
                    valid, Q[T.min(first + r // S, tokens - 1), h, d].astype(T.float32)
                    * K1[block, j % BS, kvh, d].astype(T.float32), 0)
                vlo[r, d] = T.if_then_else(valid, V1[block, j % BS, kvh, d], 0)
            T.clear(acc)
            T.clear(z)
            T.fill(mx, -T.infinity(T.float32))
            for tile in T.Pipelined(T.if_then_else(count > 0, T.ceildiv(hi - lo + 1, BN), 0), num_stages=stages):
                for k, d in T.Parallel(BN, D):
                    key = T.min(lo + tile * BN + k, hi)
                    block = T.cast(LONG_TABLE[seq, key // BS], "int64")
                    kv[k, d] = K2[block, key % BS, kvh, d]
                    vv[k, d] = V2[block, key % BS, kvh, d]
                T.clear(score)
                T.gemm(a, kv, score, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for r, k in T.Parallel(M, BN):
                    pos = base + r // S
                    key = lo + tile * BN + k
                    score[r, k] = T.if_then_else(
                        r // S < count and r % S < W1 and pos - r % S >= 0 and key <= pos and key > pos - W2,
                        score[r, k] * scale * 1.4426950408889634,
                        -T.infinity(T.float32),
                    )
                T.reduce_max(score, rowmax, dim=1)
                T.reduce_max(rowmax2, current_max, dim=1)
                T.copy(mx, old)
                for i in T.Parallel(BQ):
                    mx[i] = T.if_then_else(i < count, T.max(mx[i], current_max[i]), 0)
                for r, k in T.Parallel(M, BN):
                    score[r, k] = T.exp2(score[r, k] - mx[r // S])
                T.reduce_sum(score, rowz, dim=1)
                T.reduce_sum(rowz2, zz, dim=1)
                for i in T.Parallel(BQ):
                    z[i] = z[i] * T.exp2(old[i] - mx[i]) + zz[i]
                for r, d in T.Parallel(M, D):
                    acc[r, d] *= T.exp2(old[r // S] - mx[r // S])
                T.copy(score, prob)
                T.gemm(prob, vv, acc, policy=T.GemmWarpPolicy.FullRow)
            for r, d in T.Parallel(M, D):
                vlo[r, d] *= acc[r, d]
            T.reduce_sum(result, output, dim=1)
            for i, d in T.Parallel(BQ, D):
                if i < count:
                    OUT[first + i, h, d] = output[i, d] / z[i]

    return main


def _decode_tile(group):
    """Head-packing for the split kernel: 16 rows per warp keeps the FP32 accumulators in registers."""
    padded = 1 << max(group - 1, 0).bit_length()
    return dict(GP=padded, threads=32 * padded, **_DECODE)


_DECODE = dict(BN=64, stages=1)
# Swept on B300 at Limite geometry (10/2 heads, D=128): 2.6x faster than BQ=4/128 threads; BQ=16 is no better.
_TILE_QUERIES = 8
_TILE = dict(BN=64, threads=256, stages=2)


def _tiled_paged_attention(q, short_cache, long_cache, block_table, long_block_table, token_pos,
                           query_start_loc, *, short_window, long_window, scaling):
    tokens, heads, dim = q.shape
    requests = min(query_start_loc.numel() - 1, block_table.shape[0])
    block_table, long_block_table = block_table[:requests], long_block_table[:requests]
    tiles = torch.zeros(requests + 1, device=q.device, dtype=torch.int32)
    lengths = query_start_loc[1:requests + 1] - query_start_loc[:requests]
    tiles[1:] = torch.cumsum((lengths + _TILE_QUERIES - 1) // _TILE_QUERIES, 0)
    starts = query_start_loc[:requests + 1].contiguous()
    # Host-known upper bound on the tile count: no device synchronization, fixed per captured shape.
    grid = torch.empty(-(-tokens // _TILE_QUERIES) + requests, device=q.device, dtype=torch.int32)
    out = torch.empty((tokens, heads, dim), device=q.device, dtype=torch.float32)
    kernel = _paged_tiled_kernel(heads, short_cache.shape[3], dim, int(short_window), int(long_window),
                                 int(short_cache.shape[2]), float(scaling), BQ=_TILE_QUERIES, **_TILE)
    kernel(q, short_cache[0], short_cache[1], long_cache[0], long_cache[1], block_table, long_block_table,
           starts, tiles, token_pos, grid, out)
    return out


def tilelang_paged_attention(q, short_cache, long_cache, block_table, token_seq, token_pos, *,
                             short_window, long_window, scaling, long_block_table=None, splits=None,
                             query_start_loc=None):
    """q [tokens, query_heads, D] BF16; returns FP32 [tokens, query_heads, D].

    ``long_window`` bounds the K2/V2 axis (pass a value >= the longest sequence
    for global layers). ``block_table`` indexes the short cache and, unless
    ``long_block_table`` is given, also the long cache. Block zero may be the
    null block; every slot inside a token's windows must be allocated.
    With ``query_start_loc`` (vLLM's packed request offsets) a batch holding at
    least ``_TILE_QUERIES`` tokens per request uses the query-tiled prefill kernel;
    decode-dominated batches keep the long-axis split kernel.
    """
    long_block_table = block_table if long_block_table is None else long_block_table
    if q.ndim != 3 or short_cache.ndim != 5 or long_cache.ndim != 5:
        raise ValueError("expected q [T,H,D] and caches [2,blocks,block,Hk,D]")
    tokens, heads, dim = q.shape
    if (short_cache.shape[0] != 2 or long_cache.shape[0] != 2
            or short_cache.shape[2:] != long_cache.shape[2:]
            or short_cache.shape[-1] != dim or heads % short_cache.shape[3] or dim not in (64, 128)):
        raise ValueError("incompatible paged simplicial geometry")
    if not 1 <= short_window <= min(long_window, _SHORT_SLOTS):
        raise ValueError("require 1 <= short_window <= min(long_window, 16)")
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("invalid attention scale")
    tables = (block_table, long_block_table, token_seq, token_pos)
    if (not q.is_cuda or any(x.dtype != torch.bfloat16 or x.device != q.device
                             for x in (q, short_cache, long_cache))
            or any(x.device != q.device or x.dtype != torch.int32 for x in tables)
            or token_seq.shape != (tokens,) or token_pos.shape != (tokens,)
            or block_table.ndim != 2 or long_block_table.ndim != 2
            or block_table.shape[0] != long_block_table.shape[0]):
        raise ValueError("paged TileLang attention requires CUDA BF16 caches and int32 metadata")
    if any(not x.is_contiguous() for x in (q, *tables)) or short_cache.stride(-1) != 1 or long_cache.stride(-1) != 1:
        raise ValueError("queries and metadata must be contiguous; cache rows must be dense")
    if tokens == 0:
        return torch.empty((0, heads, dim), device=q.device, dtype=torch.float32)
    if query_start_loc is not None and tokens >= _TILE_QUERIES * (query_start_loc.numel() - 1):
        if query_start_loc.dtype != torch.int32 or query_start_loc.device != q.device or splits is not None:
            raise ValueError("tiled prefill takes int32 query_start_loc on the query device and no splits")
        return _tiled_paged_attention(q, short_cache, long_cache, block_table, long_block_table, token_pos,
                                      query_start_loc, short_window=short_window, long_window=long_window,
                                      scaling=scaling)
    kv_heads, block_size = short_cache.shape[3], short_cache.shape[2]
    # Depends only on host ints fixed per captured graph (token count, block-table width).
    capacity = long_block_table.shape[1] * block_size
    if splits is None:
        programs = 4 * torch.cuda.get_device_properties(q.device).multi_processor_count
        splits = max(1, min(-(-programs // (tokens * kv_heads)), -(-capacity // 512)))
    partial = torch.empty((splits, tokens, heads, dim), device=q.device, dtype=torch.float32)
    maxima = torch.empty((splits, tokens, heads), device=q.device, dtype=torch.float32)
    sums = torch.empty((splits, tokens, heads), device=q.device, dtype=torch.float32)
    group = heads // kv_heads
    kernel = _paged_kernel(heads, kv_heads, dim, int(short_window), int(long_window), int(block_size), float(scaling),
                           **_decode_tile(group))
    kernel(q, short_cache[0], short_cache[1], long_cache[0], long_cache[1], block_table, long_block_table,
           token_seq, token_pos, partial, maxima, sums)
    # Empty splits carry max=-inf and sum=0; every token's own key keeps one split nonempty.
    weight = torch.exp2(maxima - maxima.amax(0))
    return (partial * weight[..., None]).sum(0) / (sums * weight).sum(0)[..., None]
