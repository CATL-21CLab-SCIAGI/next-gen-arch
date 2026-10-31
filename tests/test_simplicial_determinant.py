"""Signed determinant math, independent gradients, and bounded CUDA qualification."""

import json

import pytest
import torch

from archlab.architectures.simplicial_attention import (
    determinant_operands,
    determinant_simplicial_attention,
    oriented_determinant_simplicial_attention,
    reference_determinant_simplicial,
    reference_simplicial,
    simplicial_attention,
    validate_inputs,
)


def _inputs(score_dim, value_dim, *, n=4, heads=4, kv_heads=2, dtype=torch.float64):
    generator = torch.Generator().manual_seed(731)
    return [
        (torch.randn(1, n, h, d, generator=generator, dtype=dtype) * .5).requires_grad_()
        for h, d in ((heads, score_dim), (kv_heads, score_dim), (kv_heads, score_dim),
                     (kv_heads, value_dim), (kv_heads, value_dim))
    ]


def _expanded_reference(q, k1, k2, v1, v2, short, long, *, scale=None):
    operands = determinant_operands(q, k1, k2)
    return reference_simplicial(*operands, v1, v2, short, long,
                               scale=q.shape[-1] ** -.5 if scale is None else scale)


@pytest.mark.parametrize("score_dim,value_dim,scale", [(3, 2, 1.), (6, 5, .37), (8, 8, None),
                                                        (256, 256, None)])
def test_sarrus_attention_matches_linalg_det_and_all_five_gradients(score_dim, value_dim, scale):
    inputs = _inputs(score_dim, value_dim)
    actual = _expanded_reference(*inputs, 2, 3, scale=scale)
    expected = reference_determinant_simplicial(*inputs, 2, 3, scale=scale)
    dy = torch.linspace(-1, 1, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    actual_grads = torch.autograd.grad(actual, inputs, dy)
    expected_grads = torch.autograd.grad(expected, inputs, dy)
    for a, b in zip((actual, *actual_grads), (expected, *expected_grads), strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
    if score_dim % 3:
        for grad in actual_grads[:3]:
            assert not grad[..., -(score_dim % 3):].count_nonzero()
        assert actual_grads[3][..., -1].count_nonzero()


def test_determinant_oracle_gradcheck():
    inputs = _inputs(3, 2, n=2, heads=1, kv_heads=1)
    assert torch.autograd.gradcheck(lambda *xs: reference_determinant_simplicial(*xs, 2, 2),
                                   inputs, fast_mode=True)


def test_chunk_rotations_preserve_signed_determinants_but_reflections_reverse_sign():
    generator = torch.Generator().manual_seed(721)
    vectors = [torch.randn(2, 7, dtype=torch.float64, generator=generator) for _ in range(3)]
    rotations, _ = torch.linalg.qr(torch.randn(2, 3, 3, dtype=torch.float64, generator=generator))
    rotations[..., -1] *= torch.linalg.det(rotations)[..., None]
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(2, dtype=torch.float64))

    def score(xs):
        a, b, c = determinant_operands(*xs)
        return (a * b * c).sum(-1)

    def transformed(matrix):
        return [torch.cat((torch.einsum("...ci,cij->...cj", x[..., :6].reshape(2, 2, 3), matrix)
                           .flatten(-2), x[..., 6:]), -1) for x in vectors]

    expected = torch.linalg.det(torch.stack([x[..., :6].reshape(2, 2, 3) for x in vectors], -2)).sum(-1)
    torch.testing.assert_close(score(vectors), expected)
    torch.testing.assert_close(score(transformed(rotations)), expected)
    rotations[..., 0] *= -1
    torch.testing.assert_close(score(transformed(rotations)), -expected)
    # Swapping two determinant rows changes orientation, too.
    torch.testing.assert_close(score([vectors[1], vectors[0], vectors[2]]), -expected)


def test_single_pair_and_future_mask_are_independent_of_score_coordinates():
    inputs = _inputs(8, 5, n=5)
    output = reference_determinant_simplicial(*inputs, 1, 1)
    torch.testing.assert_close(output, (inputs[3] * inputs[4]).repeat_interleave(2, 2))
    gradients = torch.autograd.grad(output.sum(), inputs)
    assert all(not grad.count_nonzero() for grad in gradients[:3])
    original = reference_determinant_simplicial(*inputs, 2, 3, query_positions=[0, 2])
    for branch in range(1, 5):
        altered = [x.detach().clone() for x in inputs]
        altered[branch][:, 3:] += 50
        torch.testing.assert_close(
            reference_determinant_simplicial(*altered, 2, 3, query_positions=[0, 2]), original)


def test_explicit_scale_does_not_use_padded_width_and_legacy_validator_stays_strict():
    inputs = _inputs(8, 5)
    expected = reference_determinant_simplicial(*inputs, 2, 3)
    torch.testing.assert_close(expected, _expanded_reference(*inputs, 2, 3, scale=8 ** -.5))
    with pytest.raises(ValueError, match="value head dimension"):
        validate_inputs(*inputs, 2, 3)
    with pytest.raises(ValueError, match="at least 3"):
        reference_determinant_simplicial(*_inputs(2, 5), 2, 3)
    with pytest.raises(ValueError, match="outside sequence"):
        reference_determinant_simplicial(*inputs, 2, 3, query_positions=[4])
    for scale in (float("nan"), float("inf"), torch.tensor(.1, requires_grad=True)):
        with pytest.raises(ValueError, match="finite real number"):
            reference_determinant_simplicial(*inputs, 2, 3, scale=scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires assigned CUDA runtime")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind,ds,dv,n,short,long,scale", [
    ("elementwise", 16, 16, 17, 1, 1, None),  # original seven positional arguments
    ("elementwise", 32, 32, 37, 2, 35, None),  # original multi-pair forward/backward
    ("elementwise", 256, 256, 6, 6, 6, None),  # original native head width
    ("elementwise", 32, 16, 37, 2, 35, .37),  # distinct score/value offsets, tile tail
    ("determinant", 8, 16, 37, 2, 35, .2),  # minimum padded width and partial chunk
    ("determinant", 256, 256, 6, 6, 6, None),  # native score=512/value=256, GQA 16/4
    ("determinant", 256, 256, 17, 2, 17, None),  # expanded-width 16-key tile boundary
])
def test_cuda_forward_and_backward_match_independent_oracle(kind, ds, dv, n, short, long, scale, dtype):
    core = determinant_simplicial_attention if kind == "determinant" else simplicial_attention
    oracle = reference_determinant_simplicial if kind == "determinant" else reference_simplicial
    inputs = [x.detach().to(device="cuda", dtype=dtype).requires_grad_()
              for x in _inputs(ds, dv, n=n, heads=16, kv_heads=4)]
    refs = [x.detach().cpu().double().requires_grad_() for x in inputs]
    kwargs = {} if scale is None else {"scale": scale}
    actual = core(*inputs, short, long, **kwargs)
    expected = oracle(*refs, short, long, **kwargs)
    generator = torch.Generator().manual_seed(753)
    dy = torch.randn(expected.shape, generator=generator, dtype=torch.float64).to(dtype)
    actual_grads = torch.autograd.grad(actual, inputs, dy.cuda())
    expected_grads = torch.autograd.grad(expected, refs, dy.double())
    errors = {}
    tolerance = .025 if dtype == torch.bfloat16 else 1e-4
    for name, observed, reference in zip(("output", "q", "k1", "k2", "v1", "v2"),
                                          (actual, *actual_grads), (expected, *expected_grads), strict=True):
        observed = observed.detach().cpu().double()
        reference = reference.detach()
        assert torch.isfinite(observed).all()
        difference = observed - reference
        relative = float(difference.norm() / reference.norm().clamp_min(1e-12))
        absolute = float(difference.abs().max())
        errors[name] = {"relative_l2": relative, "max_absolute": absolute}
        assert relative < tolerance or absolute < tolerance * .01, (name, errors[name])
    if short == long == 1:
        assert all(not grad.count_nonzero() for grad in actual_grads[:3])
    print(json.dumps({"kind": kind, "score_dim": ds, "value_dim": dv, "sequence": n,
                      "windows": [short, long], "scale": scale, "dtype": str(dtype),
                      "heads": 16, "kv_heads": 4, "relative_l2_limit": tolerance, "errors": errors}))


def _raw_arbitrary_window_det_oracle(q, k1, k2, v1, v2, w1, w2, *, scale=None):
    scale = (q.shape[-1] ** -0.5) if scale is None else float(scale)
    groups = q.shape[2] // k1.shape[2]
    k1, k2, v1, v2 = [x.repeat_interleave(groups, dim=2) for x in (k1, k2, v1, v2)]
    used = 3 * (q.shape[-1] // 3)
    qc, k1c, k2c = [x[..., :used].unflatten(-1, (used // 3, 3)) for x in (q, k1, k2)]
    outputs = []
    for i in range(q.shape[1]):
        j = slice(max(0, i - w1 + 1), i + 1)
        k = slice(max(0, i - w2 + 1), i + 1)
        a = qc[:, i][:, None, None].expand(-1, k1c[:, j].shape[1], k2c[:, k].shape[1], -1, -1, -1)
        b = k1c[:, j][:, :, None].expand_as(a)
        c = k2c[:, k][:, None].expand_as(a)
        scores = torch.linalg.det(torch.stack((a, b, c), dim=-2)).sum(-1).permute(0, 3, 1, 2)
        probs = (scores * scale).flatten(-2).softmax(-1).reshape_as(scores)
        outputs.append(torch.einsum("bhjk,bjhd,bkhd->bhd", probs, v1[:, j], v2[:, k]))
    return torch.stack(outputs, dim=1)


@pytest.mark.parametrize("w1,w2", [(2, 5), (5, 2), (4, 4), (7, 1)])
def test_oriented_determinant_reference_matches_raw_oracle_and_masks_outside_windows(w1, w2):
    inputs = _inputs(8, 5, n=7)
    actual = oriented_determinant_simplicial_attention(*inputs, w1, w2, backend="reference")
    expected = _raw_arbitrary_window_det_oracle(*inputs, w1, w2)
    dy = torch.linspace(-1, 1, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    actual_grads = torch.autograd.grad(actual, inputs, dy)
    expected_grads = torch.autograd.grad(expected, inputs, dy)
    for a, b in zip((actual, *actual_grads), (expected, *expected_grads), strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
    # Perturbing stream 1 at position 0 must not change query position 6 when w1 <= 6.
    if w1 <= 6:
        altered = [x.detach().clone() for x in inputs]
        altered[1][:, 0] += 25.0
        altered[3][:, 0] += 25.0
        out2 = oriented_determinant_simplicial_attention(*altered, w1, w2, backend="reference")
        torch.testing.assert_close(out2[:, 6], actual[:, 6], atol=1e-12, rtol=1e-10)
    if w2 <= 6:
        altered = [x.detach().clone() for x in inputs]
        altered[2][:, 0] += 25.0
        altered[4][:, 0] += 25.0
        out2 = oriented_determinant_simplicial_attention(*altered, w1, w2, backend="reference")
        torch.testing.assert_close(out2[:, 6], actual[:, 6], atol=1e-12, rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires assigned CUDA runtime")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("w1,w2", [(19, 3), (3, 19), (17, 5)])
def test_cuda_oriented_determinant_windows_match_raw_oracle(w1, w2, dtype):
    inputs = [x.detach().to(device="cuda", dtype=dtype).requires_grad_()
              for x in _inputs(256, 256, n=21, heads=16, kv_heads=4)]
    refs = [x.detach().cpu().double().requires_grad_() for x in inputs]
    actual = oriented_determinant_simplicial_attention(*inputs, w1, w2, backend="kernel")
    expected = _raw_arbitrary_window_det_oracle(*refs, w1, w2)
    generator = torch.Generator().manual_seed(911)
    dy = torch.randn(expected.shape, generator=generator, dtype=torch.float64).to(dtype)
    actual_grads = torch.autograd.grad(actual, inputs, dy.cuda())
    expected_grads = torch.autograd.grad(expected, refs, dy.double())
    tolerance = 0.025 if dtype == torch.bfloat16 else 1e-4
    for name, observed, reference in zip(("output", "q", "k1", "k2", "v1", "v2"),
                                         (actual, *actual_grads), (expected, *expected_grads), strict=True):
        observed = observed.detach().cpu().double()
        reference = reference.detach()
        assert torch.isfinite(observed).all()
        rel = float((observed - reference).norm() / reference.norm().clamp_min(1e-12))
        assert rel < tolerance, (name, rel)
