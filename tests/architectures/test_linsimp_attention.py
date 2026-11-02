"""Independent distribution, explicit pair-sum, causality, and gradient checks."""

import math
import unittest

import torch

from archlab.architectures.linsimp_attention import (
    linear_rf_attention,
    linsimp_attention,
    orthogonal_feature_bank,
    positive_features,
)


def explicit_pairs(q, k, r, v, u, omega, window, temperature):
    repeats = q.shape[2] // k.shape[2]
    k, r, v, u = [t.repeat_interleave(repeats, 2) for t in (k, r, v, u)]
    q, k, r = [torch.nn.functional.normalize(t, dim=-1) for t in (q, k, r)]
    rows = []
    for i in range(q.shape[1]):
        numerator = torch.zeros_like(q[:, i])
        denominator = torch.zeros_like(q[:, i, :, :1])
        for c in range(max(0, i - window + 1), i + 1):
            composite = positive_features(temperature * q[:, i] * r[:, c], omega)
            for j in range(i + 1):
                weight = (composite * positive_features(k[:, j], omega)).sum(-1, keepdim=True)
                numerator = numerator + weight * v[:, j] * u[:, c]
                denominator = denominator + weight
        rows.append(numerator / denominator)
    return torch.stack(rows, dim=1)


class LinSimpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gaussian_marginals_and_kernel(self):
        bank = orthogonal_feature_bank(4, 16, 16384, device="cpu", dtype=torch.float32, seed=123)
        self.assertLess(abs(bank.square().mean().item() - 1), 0.03)
        self.assertLess(bank.mean((0, 1)).abs().max().item(), 0.025)
        blocks = bank.reshape(4, -1, 16, 16)
        directions = torch.nn.functional.normalize(blocks, dim=-1)
        torch.testing.assert_close(
            directions @ directions.transpose(-1, -2),
            torch.eye(16).expand_as(blocks),
            atol=1e-6,
            rtol=1e-5,
        )
        x = torch.zeros(4, 16)
        x[:, 0] = 0.5
        estimate = (positive_features(x, bank) ** 2).sum(-1).mean().item()
        self.assertLess(abs(estimate - math.exp(0.25)), 0.025)
        self.assertTrue(
            torch.equal(
                bank,
                orthogonal_feature_bank(4, 16, 16384, device="cpu", dtype=torch.float32, seed=123),
            )
        )

    def test_joint_pair_sum_and_all_input_gradients(self):
        for window in (1, 3, 32):
            with self.subTest(window=window):
                torch.manual_seed(72)
                inputs = [torch.randn(1, 6, h, 4, requires_grad=True) for h in (4, 2, 2, 2, 2)]
                tau = torch.tensor(2.0, requires_grad=True)
                bank = orthogonal_feature_bank(4, 4, 16, device="cpu", dtype=torch.float32, seed=24)
                actual = linsimp_attention(*inputs, bank, window=window, temperature=tau)
                expected = explicit_pairs(*inputs, bank, window, tau)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                cotangent = torch.randn_like(actual)
                ga = torch.autograd.grad(actual, [*inputs, tau], cotangent, retain_graph=True)
                ge = torch.autograd.grad(expected, [*inputs, tau], cotangent)
                for a, e in zip(ga, ge, strict=False):
                    torch.testing.assert_close(a, e, atol=3e-6, rtol=4e-5)

    def test_causality_and_large_temperature(self):
        torch.manual_seed(5)
        inputs = [torch.randn(1, 7, 2, 4, requires_grad=True) for _ in range(5)]
        bank = orthogonal_feature_bank(2, 4, 16, device="cpu", dtype=torch.float32, seed=8)
        altered = [x.detach().clone() for x in inputs]
        for x in altered:
            x[:, 4:] += 50
        actual = linsimp_attention(*inputs, bank, window=3)
        future = linsimp_attention(*altered, bank, window=3)
        torch.testing.assert_close(actual[:, :4], future[:, :4], atol=0, rtol=0)
        for fn, args, kwargs in (
            (linsimp_attention, inputs, {"window": 32}),
            (linear_rf_attention, [inputs[0], inputs[1], inputs[3]], {}),
        ):
            output = fn(*args, bank, temperature=100.0, **kwargs)
            self.assertTrue(output.isfinite().all())
            first = (
                inputs[3][:, 0] * inputs[4][:, 0] if fn is linsimp_attention else inputs[3][:, 0]
            )
            torch.testing.assert_close(output[:, 0], first, atol=2e-6, rtol=2e-5)
            gradients = torch.autograd.grad(output.square().sum(), args)
            self.assertTrue(all(g.isfinite().all() for g in gradients))


if __name__ == "__main__":
    unittest.main()
