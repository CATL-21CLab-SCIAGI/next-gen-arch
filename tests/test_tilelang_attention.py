"""Independent native-geometry oracles for the fresh BF16 TileLang contract."""

import pytest
import torch

from archlab.architectures.simplicial_attention import reference_simplicial

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires B300 runtime")


@pytest.mark.parametrize("simplicial", [False, True])
@pytest.mark.parametrize("active", [1, 17, 1025])
def test_preallocated_decode_masks_unused_long_and_short_slots(simplicial, active):
    from archlab.architectures.tilelang_attention import tilelang_decode

    torch.manual_seed(53)
    q = torch.randn(4, 1, 10, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(4, 2048, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    k[:, active:].fill_(100)
    v[:, active:].fill_(100)
    short = [torch.randn(4, 16, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)] if simplicial else None
    count = min(active, 16)
    if short:
        for value in short:
            value[:, :16 - count].fill_(100)
    lengths = torch.tensor([active, count, 0], device="cuda", dtype=torch.int32)
    with torch.no_grad():
        expected = tilelang_decode(q, k[:, :active], v[:, :active], scaling=.1, short=tuple(x[:, -count:] for x in short) if short else None)
        actual = tilelang_decode(q, k, v, scaling=.1, short=tuple(short) if short else None, lengths=lengths)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def relative(actual, expected):
    return float(
        (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-6)
    )


def normal_reference(q, k, v, positions, window):
    k, v = [x.repeat_interleave(q.shape[2] // k.shape[2], dim=2) for x in (k, v)]
    scores = torch.einsum("bihd,bjhd->bhij", q[:, positions], k) * 0.1
    queries = torch.tensor(positions, device=q.device)[:, None]
    keys = torch.arange(k.shape[1], device=q.device)[None]
    scores = scores.masked_fill(
        ~((keys <= queries) & (keys > queries - window))[None, None], -float("inf")
    )
    return torch.einsum("bhij,bjhd->bihd", scores.softmax(-1), v)


@pytest.mark.parametrize(
    "variant,normal_kernel", [("normal", "shared"), ("normal", "gqa"), ("simplicial", "shared")]
)
@pytest.mark.parametrize(
    "short,long,n,positions",
    [
        (1, 1, 1, [0]),
        (1, 1, 17, [0, 1, 16]),
        (16, 17, 33, [0, 1, 15, 16, 17, 31, 32]),
        (3, 5, 37, list(range(37))),
        (16, 1025, 2048, [0, 15, 16, 1024, 1025, 2047]),
        (16, 2048, 2048, [0, 15, 16, 1025, 2047]),
    ],
)
def test_forward_and_all_gradients(variant, normal_kernel, short, long, n, positions, dynamic_length=False):
    from archlab.architectures.tilelang_attention import tilelang_attention

    torch.manual_seed(93)
    xs = [
        torch.randn(1, n, h, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for h in (10, 2, 2, 2, 2)
    ]
    fp = [x.float() for x in xs]
    oracle = (
        reference_simplicial(
            fp[0] * (0.1 * 128**0.5), *fp[1:], short, long, query_positions=positions
        )
        if variant == "simplicial"
        else normal_reference(fp[0], fp[2], fp[4], positions, long)
    )
    dy = torch.randn_like(oracle)
    expected = torch.autograd.grad(oracle, xs, dy, allow_unused=True)
    out = tilelang_attention(
        xs[0],
        xs[2],
        xs[4],
        scaling=0.1,
        long_window=long,
        short=(xs[1], xs[3]) if variant == "simplicial" else None,
        short_window=short,
        normal_kernel=normal_kernel,
        dynamic_length=dynamic_length,
    )
    actual = out[:, positions]
    grads = torch.autograd.grad(actual, xs, dy, allow_unused=True)
    assert relative(actual, oracle) < 0.005
    for grad, ref in zip(grads, expected, strict=True):
        if ref is None:
            assert grad is None
        else:
            assert torch.isfinite(grad).all()
            assert relative(grad, ref) < 0.01
    if long == short == 1:
        assert torch.equal(actual, oracle)
        assert not grads[0].count_nonzero()


@pytest.mark.parametrize(
    "variant,normal_kernel", [("normal", "shared"), ("normal", "gqa"), ("simplicial", "shared")]
)
@pytest.mark.parametrize(
    "short,long,n,positions",
    [(3, 5, 37, list(range(37))),
     (16, 1025, 1026, [0, 15, 16, 1024, 1025]),
     (16, 1111, 1111, [0, 15, 16, 1024, 1110])],
)
def test_runtime_sequence_forward_and_all_gradients(variant, normal_kernel, short, long, n, positions):
    test_forward_and_all_gradients(variant, normal_kernel, short, long, n, positions, dynamic_length=True)


@pytest.mark.parametrize("variant", ["normal", "simplicial"])
@pytest.mark.parametrize("n", [1, 17, 1025, 1026, 8192])
def test_decode_joint_softmax(variant, n):
    from archlab.architectures.tilelang_attention import tilelang_decode

    torch.manual_seed(83)
    q = torch.randn(2, 1, 10, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(2, n, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    short = (
        tuple(
            torch.randn(2, min(16, n), 2, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        )
        if variant == "simplicial"
        else None
    )
    kk, vv = [x.float().repeat_interleave(5, dim=2) for x in (k, v)]
    if short:
        k1, v1 = [x.float().repeat_interleave(5, dim=2) for x in short]
        scores = torch.einsum("bhd,bjhd,bkhd->bhjk", q[:, 0].float(), k1, kk) * 0.1
        probs = scores.flatten(-2).softmax(-1).reshape_as(scores)
        oracle = torch.einsum("bhjk,bjhd,bkhd->bhd", probs, v1, vv)[:, None]
    else:
        scores = torch.einsum("bhd,bkhd->bhk", q[:, 0].float(), kk) * 0.1
        oracle = torch.einsum("bhk,bkhd->bhd", scores.softmax(-1), vv)[:, None]
    with torch.no_grad():
        actual = tilelang_decode(q, k, v, scaling=0.1, short=short)
    assert relative(actual, oracle) < 0.005


@pytest.mark.parametrize("window", [1025, 10240])
def test_gqa_full_rl_context_forward_and_gradients(window):
    from archlab.architectures.tilelang_attention import tilelang_attention

    torch.manual_seed(831)
    positions = [0, 1, 16, 1024, 1025, 8191, 10239]
    inputs = [
        torch.randn(1, 10240, heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for heads in (10, 2, 2)
    ]
    expected = normal_reference(*(value.float() for value in inputs), positions, window)
    dy = torch.randn_like(expected)  # General FP32 dO, including non-BF16-exact values.
    expected_gradients = torch.autograd.grad(expected, inputs, dy)
    out = tilelang_attention(
        *inputs, scaling=0.1, long_window=window, normal_kernel="gqa"
    )
    actual = out[:, positions]
    actual_gradients = torch.autograd.grad(actual, inputs, dy)
    assert torch.isfinite(out).all()
    assert relative(actual, expected) < 0.005
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.isfinite(actual_gradient).all()
        assert relative(actual_gradient, expected_gradient) < 0.01
