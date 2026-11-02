"""Native-geometry joint-softmax checks, including both causal boundaries."""

import pytest
import torch

from archlab.architectures.simplicial_attention import reference_simplicial


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the GPU runtime")
@pytest.mark.parametrize(
    "short,long,n,positions",
    [
        (1, 1, 17, [0, 1, 16]),
        (16, 17, 33, [0, 1, 2, 15, 16, 17, 30, 31, 32]),
        (16, 1025, 2048, [0, 15, 16, 1024, 1025, 2047]),
        (16, 2048, 2048, [0, 16, 1025, 2047]),
    ],
)
def test_native_geometry_forward_and_all_input_gradients(short, long, n, positions):
    from archlab.architectures.simplicial_packed import packed_simplicial_attention

    torch.manual_seed(71)
    xs = [torch.randn(1, n, h, 128, device="cuda", requires_grad=True) for h in (10, 2, 2, 2, 2)]
    ref = reference_simplicial(*xs, short, long, query_positions=positions)
    dy = torch.randn_like(ref)
    expected = torch.autograd.grad(ref, xs, dy)
    actual = packed_simplicial_attention(*xs, short, long)[:, positions]
    grads = torch.autograd.grad(actual, xs, dy)
    torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-5)
    for grad, oracle in zip(grads, expected, strict=True):
        assert torch.isfinite(grad).all()
        torch.testing.assert_close(grad, oracle, atol=5e-5, rtol=5e-4)
        relative = float((grad - oracle).norm() / oracle.norm().clamp_min(1e-6))
        assert relative < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the GPU runtime")
@pytest.mark.parametrize("long", [1025, 2048])
def test_native_bf16_inputs_and_query_scale_match_fp32_oracle(long):
    from archlab.architectures.simplicial_packed import packed_simplicial_attention

    torch.manual_seed(93)
    positions = [0, 15, 16, 1024, 1025, 2047]
    xs = [
        torch.randn(1, 2048, h, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for h in (10, 2, 2, 2, 2)
    ]
    scale = 0.1 * 128**0.5
    fp32 = [x.float() for x in xs]
    ref = reference_simplicial(fp32[0] * scale, *fp32[1:], 16, long, query_positions=positions)
    dy = torch.randn_like(ref)
    expected = torch.autograd.grad(ref, xs, dy)
    actual = packed_simplicial_attention(xs[0].float() * scale, *xs[1:], 16, long)[:, positions]
    grads = torch.autograd.grad(actual, xs, dy)
    torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-5)
    for grad, oracle in zip(grads, expected, strict=True):
        assert grad.dtype == torch.bfloat16
        assert torch.isfinite(grad).all()
        relative = float(
            (grad.float() - oracle.float()).norm() / oracle.float().norm().clamp_min(1e-6)
        )
        assert relative < 5e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the GPU runtime")
@pytest.mark.parametrize("n", [1, 17, 1025, 1026, 8192])
def test_split_decode_matches_joint_softmax(n):
    from archlab.architectures.simplicial_packed import packed_simplicial_decode

    torch.manual_seed(83)
    q = torch.randn(2, 1, 10, 128, device="cuda")
    k1, v1 = [torch.randn(2, min(16, n), 2, 128, device="cuda") for _ in range(2)]
    k2, v2 = [torch.randn(2, n, 2, 128, device="cuda") for _ in range(2)]
    keys1, keys2, values1, values2 = [x.repeat_interleave(5, dim=2) for x in (k1, k2, v1, v2)]
    scores = torch.einsum("bhd,bjhd,bkhd->bhjk", q[:, 0], keys1, keys2) / 128**0.5
    probabilities = scores.flatten(-2).softmax(-1).reshape_as(scores)
    expected = torch.einsum("bhjk,bjhd,bkhd->bhd", probabilities, values1, values2)[:, None]
    actual = packed_simplicial_decode(q, k1, k2, v1, v2)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
