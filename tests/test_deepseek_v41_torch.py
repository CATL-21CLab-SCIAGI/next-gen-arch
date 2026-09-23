import pytest
import torch

from archlab.architectures.deepseek_v41_torch import (
    query_chunked_sparse_attention,
    rounded_activation,
)


def test_sparse_chunk_gradient_sink_duplicates_and_empty_rows():
    torch.manual_seed(11)
    q = torch.randn(1, 3, 2, 4, dtype=torch.float64, requires_grad=True)
    kv = torch.randn(1, 5, 4, dtype=torch.float64, requires_grad=True)
    sink = torch.randn(2, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[[0, 0, 1], [-1, -1, -1], [1, 3, 4]]])
    def fn(a, b, c):
        return query_chunked_sparse_attention(a, b, c, ids, .5, query_chunk=1)
    assert torch.autograd.gradcheck(fn, (q, kv, sink))
    actual = fn(q, kv, sink)
    expected = query_chunked_sparse_attention(q, kv, sink, ids, .5, query_chunk=3, recompute=False)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual[:, 1], torch.zeros_like(actual[:, 1]))
    with torch.no_grad():
        big_sink = query_chunked_sparse_attention(q, kv, sink + 10000, ids, .5)
        assert torch.equal(big_sink, torch.zeros_like(big_sink))


def test_sparse_prefix_and_shared_kv_accumulated_gradient():
    torch.manual_seed(19)
    q = torch.randn(1, 7, 2, 4, requires_grad=True)
    kv = torch.randn(1, 7, 4, requires_grad=True)
    sink = torch.zeros(2)
    ids = torch.arange(7)[None, None, :].expand(1, 7, 7).clone()
    ids[ids > torch.arange(7)[None, :, None]] = -1
    full = query_chunked_sparse_attention(q, kv, sink, ids, .5, query_chunk=2)
    prefix = query_chunked_sparse_attention(q[:, :3], kv[:, :3], sink, ids[:, :3, :3], .5)
    torch.testing.assert_close(full[:, :3], prefix)
    first, = torch.autograd.grad(full.sum(), kv, retain_graph=True)
    second, = torch.autograd.grad((full + full).sum(), kv)
    torch.testing.assert_close(second, first * 2)


def test_fp4_round_ties_and_identity_gradient():
    x = torch.tensor([[.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.] * 4], requires_grad=True)
    rounded = rounded_activation(x, bits=4)
    expected = torch.tensor([[0, 1, 1, 2, 2, 4, 4, 6.] * 4])
    torch.testing.assert_close(rounded, expected, atol=0, rtol=0)
    rounded.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    torch.testing.assert_close(rounded_activation(-x, bits=4), -expected, atol=0, rtol=0)


@pytest.mark.parametrize("bits,e4m3", [(8, False), (4, False), (4, True)])
def test_activation_zero_is_finite(bits, e4m3):
    value = rounded_activation(torch.zeros(2, 64), bits=bits, e4m3_scale=e4m3)
    assert torch.equal(value, torch.zeros_like(value))


def test_native_exp_rounding_uses_unnormalized_probabilities():
    torch.manual_seed(42)
    q = torch.randn(1, 3, 2, 4, requires_grad=True)
    kv = torch.randn(1, 7, 4, requires_grad=True)
    sink = torch.tensor([.1, -.2])
    ids = torch.arange(7)[None, None].expand(1, 3, 7)
    scores = torch.einsum("bshd,bkd->bshk", q, kv) * .5
    maximum = scores.amax(-1, keepdim=True)
    exp_scores = (scores - maximum).exp()
    expected = torch.einsum("bshk,bkd->bshd", exp_scores.bfloat16().float(), kv)
    expected = expected / (exp_scores.sum(-1, keepdim=True) + (sink[None, None, :, None] - maximum).exp())
    actual = query_chunked_sparse_attention(q, kv, sink, ids, .5, native_rounding=True)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert q.grad.isfinite().all() and kv.grad.isfinite().all()


def test_native_rounding_masks_and_large_sink_have_finite_backward():
    q = torch.randn(1, 3, 2, 4, requires_grad=True)
    kv = torch.randn(1, 7, 4, requires_grad=True)
    sink = torch.tensor([10000., -10000.], requires_grad=True)
    ids = torch.full((1, 3, 70), -1)
    result = query_chunked_sparse_attention(q, kv, sink, ids, .5, native_rounding=True)
    assert torch.equal(result, torch.zeros_like(result))
    result.sum().backward()
    assert q.grad.isfinite().all() and kv.grad.isfinite().all() and sink.grad.isfinite().all()
