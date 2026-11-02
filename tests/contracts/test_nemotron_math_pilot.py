import pytest

from archlab.preprocessing.nemotron_math_pilot import shifted_window_spans


def test_shifted_targets_have_complete_unique_coverage():
    spans = [[3, 8], [10, 19]]
    selected = []
    for start in range(0, 20, 4):
        for lo, hi in shifted_window_spans(spans, start, 4):
            selected.extend(range(start + lo + 1, start + hi + 1))
    assert selected == [*range(3, 8), *range(10, 19)]


def test_final_window_budget_masks_exactly_once():
    assert shifted_window_spans([[0, 4], [7, 10]], 0, 12, limit=5) == [[0, 3], [6, 8]]
    assert shifted_window_spans([[2, 5]], 0, 12, limit=0) == []
    assert shifted_window_spans([[2, 5]], 7, 4) == []


@pytest.mark.parametrize("spans", [[[2, 2]], [[-1, 2]], [[2, 8], [7, 10]], [[5, 8], [2, 4]]])
def test_invalid_spans_fail_closed(spans):
    with pytest.raises(ValueError):
        shifted_window_spans(spans, 0, 20)
