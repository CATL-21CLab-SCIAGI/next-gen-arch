import math

import pytest
import torch

from archlab.architectures.simplicial_attention import reference_simplicial

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _paged_batch(lengths, kv_heads, dim, block_size, dtype, hnd=False, seed=0):
    """Contiguous per-sequence K/V plus the same values scattered into shuffled physical blocks."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    blocks_per = [math.ceil(n / block_size) for n in lengths]
    total = 1 + sum(blocks_per)  # block 0 is the null placeholder
    physical = (torch.randperm(total - 1, generator=generator) + 1).tolist()
    table = torch.zeros(len(lengths), max(blocks_per), dtype=torch.int32)
    shape = (2, total, kv_heads, block_size, dim) if hnd else (2, total, block_size, kv_heads, dim)
    caches = [torch.zeros(shape, dtype=dtype) for _ in range(2)]  # short, long
    contiguous = []
    for s, n in enumerate(lengths):
        table[s, :blocks_per[s]] = torch.tensor(physical[:blocks_per[s]], dtype=torch.int32)
        physical = physical[blocks_per[s]:]
        k1, k2, v1, v2 = (torch.randn(1, n, kv_heads, dim, generator=generator).to(dtype) for _ in range(4))
        contiguous.append((k1, k2, v1, v2))
        for position in range(n):
            block, offset = int(table[s, position // block_size]), position % block_size
            for cache, (k, v) in zip(caches, ((k1, v1), (k2, v2))):
                view = cache.transpose(2, 3) if hnd else cache
                view[0, block, offset] = k[0, position]
                view[1, block, offset] = v[0, position]
    if hnd:  # logical [2, blocks, block, Hk, D] with HND physical strides
        caches = [cache.transpose(2, 3) for cache in caches]
    return [c.cuda() for c in caches], table.cuda(), contiguous


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-4), (torch.bfloat16, 3e-2)])
@pytest.mark.parametrize("heads,kv_heads", [(4, 2), (10, 2), (8, 1)])
@pytest.mark.parametrize("short_window,long_window", [(16, 64), (16, 10_000), (4, 4)])
@pytest.mark.parametrize("hnd", [False, True])
def test_matches_oracle_for_prefill_and_decode_tokens(dtype, tol, heads, kv_heads, short_window, long_window, hnd):
    from archlab.architectures.simplicial_paged import simplicial_paged_attention

    dim, block_size, lengths = 64, 16, [1, 23, 140]
    (short, long), table, contiguous = _paged_batch(lengths, kv_heads, dim, block_size, dtype, hnd)
    generator = torch.Generator(device="cpu").manual_seed(1)
    seqs, positions, queries, expected = [], [], [], []
    for s, (n, (k1, k2, v1, v2)) in enumerate(zip(lengths, contiguous)):
        q = torch.randn(1, n, heads, dim, generator=generator).to(dtype)
        oracle = reference_simplicial(q.double(), k1.double(), k2.double(), v1.double(), v2.double(),
                                      short_window, long_window)
        chosen = range(n) if s != 1 else [n - 1]  # sequence 1 is a decode step; others are prefills
        for position in chosen:
            seqs.append(s); positions.append(position)
            queries.append(q[0, position]); expected.append(oracle[0, position])
    out = simplicial_paged_attention(
        torch.stack(queries).cuda(), short, long, table,
        torch.tensor(seqs, dtype=torch.int32, device="cuda"), torch.tensor(positions, dtype=torch.int32, device="cuda"),
        short_window=short_window, long_window=long_window, scale=dim ** -0.5)
    torch.testing.assert_close(out.cpu(), torch.stack(expected).float(), rtol=tol, atol=tol)


def test_token_order_and_batch_composition_do_not_change_results():
    from archlab.architectures.simplicial_paged import simplicial_paged_attention

    dim, heads, kv_heads = 64, 4, 2
    (short, long), table, contiguous = _paged_batch([50, 70], kv_heads, dim, 16, torch.float32, seed=3)
    q = torch.randn(120, heads, dim, device="cuda")
    seq = torch.tensor([0] * 50 + [1] * 70, dtype=torch.int32, device="cuda")
    pos = torch.tensor(list(range(50)) + list(range(70)), dtype=torch.int32, device="cuda")
    kwargs = dict(short_window=16, long_window=33, scale=dim ** -0.5, splits=3)  # fixed split count: bitwise comparison
    full = simplicial_paged_attention(q, short, long, table, seq, pos, **kwargs)
    order = torch.randperm(120, device="cuda")
    shuffled = simplicial_paged_attention(q[order], short, long, table, seq[order], pos[order], **kwargs)
    torch.testing.assert_close(shuffled, full[order], rtol=0, atol=0)
    single = simplicial_paged_attention(q[60:61], short, long, table, seq[60:61], pos[60:61], **kwargs)
    torch.testing.assert_close(single, full[60:61], rtol=0, atol=0)


def test_cuda_graph_replay_matches_eager():
    from archlab.architectures.simplicial_paged import simplicial_paged_attention

    dim, heads, kv_heads = 64, 4, 2
    (short, long), table, _ = _paged_batch([40, 90], kv_heads, dim, 16, torch.bfloat16, seed=5)
    q = torch.randn(2, heads, dim, device="cuda", dtype=torch.bfloat16)
    seq = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    pos = torch.tensor([39, 89], dtype=torch.int32, device="cuda")
    kwargs = dict(short_window=16, long_window=64, scale=dim ** -0.5)
    eager = simplicial_paged_attention(q, short, long, table, seq, pos, **kwargs)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        simplicial_paged_attention(q, short, long, table, seq, pos, **kwargs)  # warm up compile
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        captured = simplicial_paged_attention(q, short, long, table, seq, pos, **kwargs)
    pos.copy_(torch.tensor([20, 60], dtype=torch.int32, device="cuda"))  # metadata changes between replays
    graph.replay()
    expected = simplicial_paged_attention(q, short, long, table, seq, pos, **kwargs)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    assert not torch.equal(captured, eager)


def test_separate_short_and_long_block_tables():
    from archlab.architectures.simplicial_paged import simplicial_paged_attention

    dim, heads, kv_heads, lengths = 64, 4, 2, [30, 75]
    (short, _), short_table, contiguous = _paged_batch(lengths, kv_heads, dim, 16, torch.float32, seed=7)
    (_, long), long_table, _ = _paged_batch(lengths, kv_heads, dim, 16, torch.float32, seed=7)
    # Same values, different physical placement for the long axis.
    perm_seed = torch.Generator(device="cpu").manual_seed(11)
    order = (torch.randperm(long.shape[1] - 1, generator=perm_seed) + 1)
    remap = torch.zeros(long.shape[1], dtype=torch.long); remap[1:] = order
    moved = torch.zeros_like(long); moved[:, remap.cuda()] = long
    moved_table = remap.cuda()[long_table.long()].to(torch.int32)
    q = torch.randn(sum(lengths), heads, dim, device="cuda")
    seq = torch.tensor(sum(([s] * n for s, n in enumerate(lengths)), []), dtype=torch.int32, device="cuda")
    pos = torch.tensor(sum((list(range(n)) for n in lengths), []), dtype=torch.int32, device="cuda")
    kwargs = dict(short_window=16, long_window=40, scale=dim ** -0.5)
    shared = simplicial_paged_attention(q, short, long, short_table, seq, pos, **kwargs)
    separate = simplicial_paged_attention(q, short, moved, short_table, seq, pos, long_block_table=moved_table, **kwargs)
    torch.testing.assert_close(separate, shared, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-5), (torch.bfloat16, 2e-2)])
def test_long_axis_splits_agree(dtype, tol):
    from archlab.architectures.simplicial_paged import simplicial_paged_attention

    dim, heads, kv_heads, lengths = 64, 8, 2, [300, 37]
    (short, long), table, _ = _paged_batch(lengths, kv_heads, dim, 16, dtype, seed=13)
    q = torch.randn(2, heads, dim, device="cuda").to(dtype)
    seq = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    pos = torch.tensor([299, 36], dtype=torch.int32, device="cuda")
    kwargs = dict(short_window=16, long_window=10_000, scale=dim ** -0.5)
    one = simplicial_paged_attention(q, short, long, table, seq, pos, splits=1, **kwargs)
    for splits in (2, 7, 19, None):
        many = simplicial_paged_attention(q, short, long, table, seq, pos, splits=splits, **kwargs)
        torch.testing.assert_close(many, one, rtol=tol, atol=tol)


def test_block_offsets_beyond_int32_elements():
    """Production caches hold > 2**31 elements: block * block_stride must not overflow (vLLM eval crash)."""
    from archlab.architectures.simplicial_paged import simplicial_paged_attention

    dim, heads, kv_heads, block_size, lengths = 64, 4, 1, 16, [40, 70]
    (short, long), table, _ = _paged_batch(lengths, kv_heads, dim, block_size, torch.bfloat16, seed=3)
    q = torch.randn(sum(lengths), heads, dim, device="cuda", dtype=torch.bfloat16)
    seq = torch.tensor(sum(([s] * n for s, n in enumerate(lengths)), []), dtype=torch.int32, device="cuda")
    pos = torch.tensor(sum((list(range(n)) for n in lengths), []), dtype=torch.int32, device="cuda")
    kwargs = dict(short_window=16, long_window=60, scale=dim ** -0.5)
    expected = simplicial_paged_attention(q, short, long, table, seq, pos, **kwargs)
    shift = 2 ** 31 // (block_size * kv_heads * dim) + 7  # first moved block's offset exceeds int32
    big = []
    for cache in (short, long):
        moved = torch.zeros((2, shift + cache.shape[1], block_size, kv_heads, dim), dtype=cache.dtype, device="cuda")
        moved[:, shift:] = cache
        big.append(moved)
    moved_table = torch.where(table > 0, table + shift, table)
    out = simplicial_paged_attention(q, big[0], big[1], moved_table, seq, pos, **kwargs)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
