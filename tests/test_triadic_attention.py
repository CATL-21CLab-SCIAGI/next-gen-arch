"""Independent FP64 joint-state oracle and all-input gradient qualification."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from archlab.architectures.triadic_attention import (
    _official_runtime_identity,
    causal_conv_reference,
    triadic_gdn_attention,
    triadic_gdn_reference,
)


def joint_matrix_oracle(q, k, v, k2, q2, g, beta, *, scale=None):
    """Flatten features; apply a full joint transition matrix in FP64.

    Unlike the production recurrence's slice contractions, this directly forms
    (I-beta*kappa*kappa^T) diag(exp(g)) and advances a two-dimensional memory.
    """
    batch, length, heads, dim = q.shape
    features = k2.shape[-1]
    state = q.new_zeros(batch, heads, features * dim, v.shape[-1])
    identity = torch.eye(features * dim, dtype=q.dtype, device=q.device)
    outputs = []
    for index in range(length):
        key = (k2[:, index, :, :, None] * k[:, index, :, None, :]).flatten(-2)
        query = (q2[:, index, :, :, None] * q[:, index, :, None, :]).flatten(-2)
        decay = g[:, index].exp().repeat_interleave(dim, -1)
        transition = (
            identity - beta[:, index, :, None, None] * key[..., :, None] * key[..., None, :]
        ) * decay[..., None, :]
        state = transition @ state + (
            beta[:, index, :, None, None] * key[..., :, None] * v[:, index, :, None, :]
        )
        outputs.append((query[..., None, :] @ state).squeeze(-2))
    return torch.stack(outputs, 1) * (dim**-0.5 if scale is None else scale)


def small_inputs(features=2, *, requires_grad=False):
    generator = torch.Generator().manual_seed(213)

    def normal(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    q, k = [F.normalize(normal(1, 4, 1, 3), dim=-1) for _ in range(2)]
    v = normal(1, 4, 1, 2)
    k2, q2 = [F.normalize(F.softplus(normal(1, 4, 1, features)), dim=-1) for _ in range(2)]
    g = -F.softplus(normal(1, 4, 1, features)) * 0.2
    beta = normal(1, 4, 1).sigmoid()
    return [x.requires_grad_(requires_grad) for x in (q, k, v, k2, q2, g, beta)]


class TriadicAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_joint_matrix_output_and_all_gradients(self):
        inputs = small_inputs(requires_grad=True)
        actual = triadic_gdn_reference(*inputs, scale=0.7)
        expected = joint_matrix_oracle(*inputs, scale=0.7)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        cotangent = torch.linspace(-1, 1, actual.numel(), dtype=torch.float64).reshape_as(actual)
        actual_gradients = torch.autograd.grad(actual, inputs, cotangent, retain_graph=True)
        expected_gradients = torch.autograd.grad(expected, inputs, cotangent)
        for name, actual_gradient, expected_gradient in zip(
            ("q", "k", "v", "k2", "q2", "g", "beta"),
            actual_gradients,
            expected_gradients,
            strict=True,
        ):
            with self.subTest(input=name):
                torch.testing.assert_close(
                    actual_gradient, expected_gradient, rtol=1e-11, atol=1e-12
                )
                self.assertTrue(actual_gradient.isfinite().all())
                self.assertGreater(actual_gradient.norm().item(), 0)

    def test_all_seven_inputs_finite_difference(self):
        inputs = [x[:, :2].clone().requires_grad_(True) for x in small_inputs()]
        self.assertTrue(torch.autograd.gradcheck(triadic_gdn_reference, inputs, fast_mode=True))

    def test_feature_one_is_gdn(self):
        inputs = small_inputs(features=1)
        inputs[3] = torch.ones_like(inputs[3])
        inputs[4] = torch.ones_like(inputs[4])
        torch.testing.assert_close(
            triadic_gdn_reference(*inputs), joint_matrix_oracle(*inputs), atol=1e-12, rtol=1e-12
        )

    def test_causality_and_packed_document_reset(self):
        inputs = small_inputs()
        altered = [x.clone() for x in inputs]
        for x in altered:
            x[:, 2:] += 4
        actual = triadic_gdn_reference(*inputs)
        torch.testing.assert_close(
            actual[:, :2], triadic_gdn_reference(*altered)[:, :2], rtol=0, atol=0
        )
        offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
        packed = triadic_gdn_reference(*inputs, cu_seqlens=offsets)
        separate = torch.cat(
            [
                triadic_gdn_reference(*[x[:, left:right] for x in inputs])
                for left, right in ((0, 2), (2, 4))
            ],
            1,
        )
        torch.testing.assert_close(packed, separate, rtol=0, atol=0)

    def test_joint_erase_is_not_independent_slice_erasure(self):
        inputs = small_inputs(features=2)
        q, k, v, k2, q2, g, beta = inputs
        state = q.new_zeros(1, 1, 2, 3, 2)
        outputs = []
        for index in range(q.shape[1]):
            state = state * g[:, index].exp()[..., None, None]
            separate_predictions = torch.einsum("bhd,bhedv->bhev", k[:, index], state)
            residual = v[:, index, :, None] - k2[:, index, :, :, None] * separate_predictions
            state = state + torch.einsum(
                "bh,bhe,bhd,bhev->bhedv", beta[:, index], k2[:, index], k[:, index], residual
            )
            outputs.append(torch.einsum("bhd,bhedv,bhe->bhv", q[:, index], state, q2[:, index]))
        wrong = torch.stack(outputs, 1) * (3**-0.5)
        self.assertGreater((triadic_gdn_reference(*inputs) - wrong).abs().max().item(), 1e-3)

    def test_no_implicit_production_cpu_fallback(self):
        with self.assertRaisesRegex(ValueError, "Hopper or datacenter Blackwell"):
            triadic_gdn_attention(*small_inputs(), backend="official")
        torch.testing.assert_close(
            triadic_gdn_attention(*small_inputs(), backend="reference"),
            triadic_gdn_reference(*small_inputs()),
        )

    def test_metadata_cannot_masquerade_as_actual_dsl_version(self):
        _official_runtime_identity.cache_clear()
        with (
            patch(
                "archlab.architectures.triadic_attention.metadata.version",
                return_value="4.7.1",
            ),
            patch(
                "archlab.architectures.triadic_attention.importlib.import_module",
                return_value=SimpleNamespace(__version__="4.5.0", __file__=__file__),
            ),
            self.assertRaisesRegex(RuntimeError, "imported 4.5.0"),
        ):
            _official_runtime_identity()

    def test_invalid_packed_boundaries(self):
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            triadic_gdn_reference(
                *small_inputs(), cu_seqlens=torch.tensor([0, 2, 2, 4], dtype=torch.int32)
            )

    def test_convolution_scalar_oracle_and_document_reset(self):
        torch.manual_seed(22)
        x = torch.randn(1, 6, 3, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        offsets = torch.tensor([0, 2, 6], dtype=torch.int32)
        actual = causal_conv_reference(x, weight, 2, cu_seqlens=offsets)
        rows = []
        for left, right in ((0, 2), (2, 6)):
            for index in range(left, right):
                value = sum(
                    x[:, index - 3 + slot] * weight[:, slot]
                    for slot in range(4)
                    if index - 3 + slot >= left
                )
                rows.append(torch.cat((F.silu(value[:, :2]), value[:, 2:]), -1))
        expected = torch.stack(rows, 1)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        ga = torch.autograd.grad(actual.sum(), (x, weight), retain_graph=True)
        ge = torch.autograd.grad(expected.sum(), (x, weight))
        for actual_gradient, expected_gradient in zip(ga, ge, strict=True):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), "official kernel qualification needs a GPU")
    def test_official_gpu_output_and_all_seven_gradients(self):
        # Run on the actual container with the pinned dependency. Never skip a
        # missing/mismatched dependency on a supported GPU.
        device = torch.device("cuda")
        if torch.cuda.get_device_capability()[0] not in (9, 10):
            self.skipTest("official kernel supports Hopper/datacenter Blackwell")
        generator = torch.Generator(device=device).manual_seed(121)

        def normal(*shape):
            return torch.randn(*shape, generator=generator, device=device)

        q, k = [F.normalize(normal(1, 64, 2, 128), dim=-1).to(torch.bfloat16) for _ in range(2)]
        v = normal(1, 64, 2, 128).mul(0.5).to(torch.bfloat16)
        k2, q2 = [F.normalize(F.softplus(normal(1, 64, 2, 4)), dim=-1) for _ in range(2)]
        g = -F.softplus(normal(1, 64, 2, 4)).mul(0.1)
        beta = normal(1, 64, 2).sigmoid()
        inputs = [t.detach().requires_grad_(True) for t in (q, k, v, k2, q2, g, beta)]
        reference_inputs = [t.detach().double().requires_grad_(True) for t in inputs]
        actual = triadic_gdn_attention(*inputs)
        expected = triadic_gdn_reference(*reference_inputs)
        cotangent = normal(*actual.shape).to(torch.bfloat16)
        actual_gradients = torch.autograd.grad(actual, inputs, cotangent)
        expected_gradients = torch.autograd.grad(expected, reference_inputs, cotangent.double())

        def relative(a, b):
            return ((a.double() - b).norm() / b.norm().clamp_min(1e-12)).item()

        self.assertLess(relative(actual, expected), 0.02)
        for name, actual_gradient, expected_gradient in zip(
            ("q", "k", "v", "k2", "q2", "g", "beta"),
            actual_gradients,
            expected_gradients,
            strict=True,
        ):
            with self.subTest(input=name):
                self.assertTrue(actual_gradient.isfinite().all())
                self.assertLess(relative(actual_gradient, expected_gradient), 0.05)


if __name__ == "__main__":
    unittest.main()
