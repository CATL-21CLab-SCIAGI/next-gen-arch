"""Explicit full-width leaf-kernel gate, not a pretrained-model training test."""

import unittest

import torch

from archlab.benchmarks.simplicial_attention import correctness_case


@unittest.skipUnless(torch.cuda.is_available(), "requires an unused GPU in the frozen container")
class SimplicialHead256Tests(unittest.TestCase):
    def test_oracle_forward_backward_and_both_window_boundaries(self):
        for dtype in (torch.float32, torch.bfloat16):
            for length, short, long, positions in (
                (17, 4, 16, None),
                (129, 16, 128, [0, 1, 15, 16, 127, 128]),
                (5, 1, 1, None),
            ):
                with self.subTest(dtype=dtype, length=length, windows=(short, long)):
                    result = correctness_case(length, short, long, dtype, positions, head_dim=256)
                    self.assertTrue(result["passed"], result)


if __name__ == "__main__":
    unittest.main()
