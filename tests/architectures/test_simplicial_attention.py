"""Independent CPU oracle checks; CUDA kernel checks also run in the probe."""

import unittest

import torch

from archlab.architectures.simplicial_attention import reference_simplicial


class SimplicialOracleTests(unittest.TestCase):
    def inputs(self, n=4, dtype=torch.float64):
        gen = torch.Generator().manual_seed(42)
        return [torch.randn(1, n, h, 4, generator=gen, dtype=dtype).requires_grad_()
                for h in (4, 2, 2, 2, 2)]

    def test_gradcheck(self):
        x = self.inputs()
        self.assertTrue(torch.autograd.gradcheck(lambda *x: reference_simplicial(*x, 2, 3), x))

    def test_single_pair_is_value_product(self):
        x = self.inputs()
        out = reference_simplicial(*x, 1, 1)
        expected = (x[3] * x[4]).repeat_interleave(2, dim=2)
        torch.testing.assert_close(out, expected)

    def test_both_key_axes_are_causal(self):
        x = self.inputs(6)
        expected = reference_simplicial(*x, 3, 5)[:, :3]
        for branch in range(1, 5):
            altered = [item.detach().clone() for item in x]
            altered[branch][:, 3:] += 100
            torch.testing.assert_close(reference_simplicial(*altered, 3, 5)[:, :3], expected)
        expected.sum().backward()
        for branch in x[1:]:
            self.assertEqual(torch.count_nonzero(branch.grad[:, 3:]).item(), 0)

    def test_joint_softmax_not_independent_normalization(self):
        x = self.inputs()
        q, k1, k2, v1, v2 = [t.repeat_interleave(2, 2) if i else t
                              for i, t in enumerate(x)]
        scores = torch.einsum('bihd,bjhd,bkhd->bhijk', q, k1, k2) / 2
        i, j, k = torch.meshgrid(*(torch.arange(4) for _ in range(3)), indexing='ij')
        mask = (j <= i) & (k <= i) & (j > i - 2) & (k > i - 3)
        p = scores.masked_fill(~mask, -float('inf')).flatten(-2).softmax(-1).reshape_as(scores)
        expected = torch.einsum('bhijk,bjhd,bkhd->bihd', p, v1, v2)
        torch.testing.assert_close(reference_simplicial(*x, 2, 3), expected)


if __name__ == '__main__':
    unittest.main()
