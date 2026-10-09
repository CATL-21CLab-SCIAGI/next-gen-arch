import threading

import numpy as np
import pytest

from archlab.automodel.limite_adapter_data import WindowPrefetcher


class Windows:
    def __getitem__(self, row):
        assert threading.current_thread().name.startswith("math-windows")
        return np.arange(5, dtype=np.int64) + 10 * (row % 17)


@pytest.mark.parametrize("rank,world,microbatch,accumulation", [(0, 1, 1, 1), (1, 2, 3, 2)])
def test_prefetch_preserves_global_rows_across_wraps_and_resume(
    rank, world, microbatch, accumulation
):
    def collect(start, stop):
        reader = WindowPrefetcher(
            Windows(),
            step=start,
            rank=rank,
            world_size=world,
            microbatch=microbatch,
            accumulation=accumulation,
        )
        try:
            return [reader.get(step) for step in range(start, stop)]
        finally:
            reader.close()

    uninterrupted = collect(31, 39)
    resumed = collect(35, 39)
    for step, actual in enumerate(uninterrupted, 31):
        batch = world * microbatch * accumulation
        expected = np.stack(
            [
                np.stack(
                    [
                        np.arange(5) + 10 * (row % 17)
                        for row in step * batch
                        + (acc * world + rank) * microbatch
                        + np.arange(microbatch)
                    ]
                )
                for acc in range(accumulation)
            ]
        )
        np.testing.assert_array_equal(actual, expected)
        if step >= 35:
            np.testing.assert_array_equal(actual, resumed[step - 35])


def test_prefetch_rejects_a_skipped_cursor():
    reader = WindowPrefetcher(Windows(), step=7, rank=0, world_size=1, microbatch=1, accumulation=1)
    try:
        with pytest.raises(ValueError, match="cursor"):
            reader.get(8)
        np.testing.assert_array_equal(reader.get(7)[0, 0], np.arange(5) + 70)
    finally:
        reader.close()
