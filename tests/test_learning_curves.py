"""Token alignment checks independent of GPU or a live training run."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from archlab.benchmarks.learning_curves import read_run, trailing_means


class CurveTests(unittest.TestCase):
    def test_trailing_mean(self):
        self.assertEqual(trailing_means([1, 3, 5, 7], 2), [1, 2, 4, 6])

    def test_dense_steps_are_averaged_into_complete_token_blocks(self):
        class FakeAccumulator:
            def __init__(self, *_args, **_kwargs):
                pass

            def Reload(self):
                return self

            def Tags(self):
                return {"scalars": ["cross entropy", "cross entropy validation", "lm loss validation"]}

            def Scalars(self, tag):
                if tag == "cross entropy":
                    # A missing step invalidates the second block, not the first.
                    return [SimpleNamespace(step=i, value=float(i), wall_time=100 + i)
                            for i in range(1, 17) if i != 12]
                return [SimpleNamespace(step=8, value=3.0 if tag == "cross entropy validation" else 99.0)]

        with tempfile.TemporaryDirectory() as directory, patch(
            "tensorboard.backend.event_processing.event_accumulator.EventAccumulator", FakeAccumulator
        ):
            result = read_run("dense", directory, "cross entropy", 1, 8)
        self.assertEqual(len(result["points"]), 1)
        self.assertEqual(result["points"][0]["tokens"], 8)
        self.assertEqual(result["points"][0]["ce"], 4.5)
        self.assertEqual(result["validation"][0]["ce"], 3.0)

    def test_fractional_optimizer_step_is_rejected(self):
        with self.assertRaises(ValueError):
            read_run("invalid", "/tmp", "lm loss", 3, 8)


if __name__ == "__main__":
    unittest.main()
