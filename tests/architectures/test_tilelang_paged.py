import math

import pytest
import torch

from archlab.architectures.simplicial_attention import reference_simplicial

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DIM, BLOCK, SCALE = 128, 16, 0.1


def _paged_batch(lengths, kv_heads, hnd=False, seed=0):
    """Contiguous per-sequence K/V plus the same values scattered into shuffled physical blocks."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    blocks_per = [math.ceil(n / BLOCK) for n in lengths]
    total = 1 + sum(blocks_per)  # block 0 is the null placeholder
    physical = (torch.randperm(total - 1, generator=generator) + 1).tolist()
    table = torch.zeros(len(lengths), max(blocks_per), dtype=torch.int32)
    shape = (2, total, kv_heads, BLOCK, DIM) if hnd else (2, total, BLOCK, kv_heads, DIM)
    caches = [torch.zeros(shape, dtype=torch.bfloat16) for _ in range(2)]  # short, long
    contiguous = []
    for s, n in enumerate(lengths):
        table[s, :blocks_per[s]] = torch.tensor(physical[:blocks_per[s]], dtype=torch.int32)
        physical = physical[blocks_per[s]:]
        k1, k2, v1, v2 = (torch.randn(1, n, kv_heads, DIM, generator=generator).bfloat16() for _ in range(4))
        contiguous.append((k1, k2, v1, v2))
        for position in range(n):
            block, offset = int(table[s, position // BLOCK]), position % BLOCK
            for cache, (k, v) in zip(caches, ((k1, v1), (k2, v2)), strict=True):
                view = cache.transpose(2, 3) if hnd else cache
                view[0, block, offset] = k[0, position]
                view[1, block, offset] = v[0, position]
    if hnd:  # logical [2, blocks, block, Hk, D] with HND physical strides
        caches = [cache.transpose(2, 3) for cache in caches]
    return [c.cuda() for c in caches], table.cuda(), contiguous


def _layout(lengths):
    seq = torch.tensor(sum(([s] * n for s, n in enumerate(lengths)), []), dtype=torch.int32, device="cuda")
    pos = torch.tensor(sum((list(range(n)) for n in lengths), []), dtype=torch.int32, device="cuda")
    return seq, pos


@pytest.mark.parametrize("heads,kv_heads", [(4, 2), (10, 2), (8, 1)])
@pytest.mark.parametrize("short_window,long_window", [(16, 64), (16, 10_000), (4, 4), (24, 64), (32, 10_000)])
@pytest.mark.parametrize("hnd", [False, True])
def test_matches_oracle_for_prefill_and_decode_tokens(heads, kv_heads, short_window, long_window, hnd):
    from archlab.architectures.tilelang_paged import tilelang_paged_attention

    lengths = [1, 23, 140]
    (short, long), table, contiguous = _paged_batch(lengths, kv_heads, hnd)
    generator = torch.Generator(device="cpu").manual_seed(1)
    seqs, positions, queries, expected = [], [], [], []
    for s, (n, (k1, k2, v1, v2)) in enumerate(zip(lengths, contiguous, strict=True)):
        q = torch.randn(1, n, heads, DIM, generator=generator).bfloat16()
        oracle = reference_simplicial(q.double(), k1.double(), k2.double(), v1.double(), v2.double(),
                                      short_window, long_window, scale=SCALE)
        chosen = range(n) if s != 1 else [n - 1]  # sequence 1 is a decode step; others are prefills
        for position in chosen:
            seqs.append(s)
            positions.append(position)
            queries.append(q[0, position])
            expected.append(oracle[0, position])
    out = tilelang_paged_attention(
        torch.stack(queries).cuda(), short, long, table,
        torch.tensor(seqs, dtype=torch.int32, device="cuda"), torch.tensor(positions, dtype=torch.int32, device="cuda"),
        short_window=short_window, long_window=long_window, scaling=SCALE)
    torch.testing.assert_close(out.cpu(), torch.stack(expected).float(), rtol=3e-2, atol=3e-2)


def test_token_order_and_batch_composition_do_not_change_results():
    from archlab.architectures.tilelang_paged import tilelang_paged_attention

    heads, kv_heads, lengths = 4, 2, [50, 70]
    (short, long), table, _ = _paged_batch(lengths, kv_heads, seed=3)
    q = torch.randn(120, heads, DIM, device="cuda").bfloat16()
    seq, pos = _layout(lengths)
    kwargs = dict(short_window=16, long_window=33, scaling=SCALE, splits=3)  # fixed split count: bitwise comparison
    full = tilelang_paged_attention(q, short, long, table, seq, pos, **kwargs)
    order = torch.randperm(120, device="cuda")
    shuffled = tilelang_paged_attention(q[order].contiguous(), short, long, table,
                                        seq[order].contiguous(), pos[order].contiguous(), **kwargs)
    torch.testing.assert_close(shuffled, full[order], rtol=0, atol=0)
    single = tilelang_paged_attention(q[60:61], short, long, table, seq[60:61], pos[60:61], **kwargs)
    torch.testing.assert_close(single, full[60:61], rtol=0, atol=0)


def test_cuda_graph_replay_matches_eager():
    from archlab.architectures.tilelang_paged import tilelang_paged_attention

    heads, kv_heads = 4, 2
    (short, long), table, _ = _paged_batch([40, 90], kv_heads, seed=5)
    q = torch.randn(2, heads, DIM, device="cuda", dtype=torch.bfloat16)
    seq = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    pos = torch.tensor([39, 89], dtype=torch.int32, device="cuda")
    kwargs = dict(short_window=16, long_window=64, scaling=SCALE)
    eager = tilelang_paged_attention(q, short, long, table, seq, pos, **kwargs)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tilelang_paged_attention(q, short, long, table, seq, pos, **kwargs)  # warm up compile
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        captured = tilelang_paged_attention(q, short, long, table, seq, pos, **kwargs)
    pos.copy_(torch.tensor([20, 60], dtype=torch.int32, device="cuda"))  # metadata changes between replays
    graph.replay()
    expected = tilelang_paged_attention(q, short, long, table, seq, pos, **kwargs)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    assert not torch.equal(captured, eager)


def test_separate_short_and_long_block_tables():
    from archlab.architectures.tilelang_paged import tilelang_paged_attention

    heads, kv_heads, lengths = 4, 2, [30, 75]
    (short, long), table, _ = _paged_batch(lengths, kv_heads, seed=7)
    perm_seed = torch.Generator(device="cpu").manual_seed(11)
    remap = torch.zeros(long.shape[1], dtype=torch.long)
    remap[1:] = torch.randperm(long.shape[1] - 1, generator=perm_seed) + 1
    moved = torch.zeros_like(long)
    moved[:, remap.cuda()] = long
    moved_table = remap.cuda()[table.long()].to(torch.int32)
    q = torch.randn(sum(lengths), heads, DIM, device="cuda").bfloat16()
    seq, pos = _layout(lengths)
    kwargs = dict(short_window=16, long_window=40, scaling=SCALE)
    shared = tilelang_paged_attention(q, short, long, table, seq, pos, **kwargs)
    separate = tilelang_paged_attention(q, short, moved, table, seq, pos, long_block_table=moved_table, **kwargs)
    torch.testing.assert_close(separate, shared, rtol=0, atol=0)


def test_long_axis_splits_agree():
    from archlab.architectures.tilelang_paged import tilelang_paged_attention

    heads, kv_heads = 8, 2
    (short, long), table, _ = _paged_batch([300, 37], kv_heads, seed=13)
    q = torch.randn(2, heads, DIM, device="cuda").bfloat16()
    seq = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    pos = torch.tensor([299, 36], dtype=torch.int32, device="cuda")
    kwargs = dict(short_window=16, long_window=10_000, scaling=SCALE)
    one = tilelang_paged_attention(q, short, long, table, seq, pos, splits=1, **kwargs)
    for splits in (2, 7, 19, None):
        many = tilelang_paged_attention(q, short, long, table, seq, pos, splits=splits, **kwargs)
        torch.testing.assert_close(many, one, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("heads,kv_heads", [(4, 2), (10, 2), (8, 1)])
@pytest.mark.parametrize("short_window,long_window", [(16, 64), (16, 10_000), (4, 4), (8, 37), (24, 64), (32, 10_000)])
def test_tiled_prefill_matches_oracle_and_split_kernel(heads, kv_heads, short_window, long_window):
    """Packed requests: full prefill, chunked prefill over a cached prefix, a decode step, an empty request."""
    from archlab.architectures.tilelang_paged import tilelang_paged_attention

    lengths = [130, 203, 41, 9]
    queried = [range(0, 130), range(150, 203), range(40, 41), range(0, 0)]
    (short, long), table, contiguous = _paged_batch(lengths, kv_heads, seed=17)
    generator = torch.Generator(device="cpu").manual_seed(19)
    seqs, positions, queries, expected, starts = [], [], [], [], [0]
    for s, ((k1, k2, v1, v2), chosen) in enumerate(zip(contiguous, queried, strict=True)):
        q = torch.randn(1, lengths[s], heads, DIM, generator=generator).bfloat16()
        oracle = reference_simplicial(q.double(), k1.double(), k2.double(), v1.double(), v2.double(),
                                      short_window, long_window, scale=SCALE)
        for position in chosen:
            seqs.append(s)
            positions.append(position)
            queries.append(q[0, position])
            expected.append(oracle[0, position])
        starts.append(starts[-1] + len(chosen))
    args = (torch.stack(queries).cuda(), short, long, table,
            torch.tensor(seqs, dtype=torch.int32, device="cuda"), torch.tensor(positions, dtype=torch.int32, device="cuda"))
    kwargs = dict(short_window=short_window, long_window=long_window, scaling=SCALE)
    tiled = tilelang_paged_attention(
        *args, query_start_loc=torch.tensor(starts, dtype=torch.int32, device="cuda"), **kwargs)
    torch.testing.assert_close(tiled.cpu(), torch.stack(expected).float(), rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(tiled, tilelang_paged_attention(*args, **kwargs), rtol=2e-2, atol=2e-2)


def test_block_offsets_beyond_int32_elements():
    """Production caches hold > 2**31 elements: block * block_stride must not overflow (vLLM eval crash)."""
    from archlab.architectures.tilelang_paged import tilelang_paged_attention

    heads, kv_heads, lengths = 4, 1, [40, 70]
    (short, long), table, _ = _paged_batch(lengths, kv_heads, seed=3)
    q = torch.randn(sum(lengths), heads, DIM, device="cuda", dtype=torch.bfloat16)
    seq, pos = _layout(lengths)
    kwargs = dict(short_window=16, long_window=60, scaling=SCALE)
    expected = tilelang_paged_attention(q, short, long, table, seq, pos, **kwargs)
    shift = 2 ** 31 // (BLOCK * kv_heads * DIM) + 7  # first moved block's offset exceeds int32
    big = []
    for cache in (short, long):
        moved = torch.zeros((2, shift + cache.shape[1], BLOCK, kv_heads, DIM), dtype=cache.dtype, device="cuda")
        moved[:, shift:] = cache
        big.append(moved)
    moved_table = torch.where(table > 0, table + shift, table)
    out = tilelang_paged_attention(q, big[0], big[1], moved_table, seq, pos, **kwargs)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
