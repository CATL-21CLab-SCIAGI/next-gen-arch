"""Exact nested state comparisons for checkpoint qualification."""

from __future__ import annotations

import torch


def assert_state_equal(actual, expected):
    """Compare local tensor values, dtypes and state structure exactly."""
    if isinstance(expected, torch.Tensor):
        actual = actual.to_local() if hasattr(actual, "to_local") else actual
        expected = expected.to_local() if hasattr(expected, "to_local") else expected
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype:
            raise AssertionError("checkpoint tensor dtype differs")
        if not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
            raise AssertionError("checkpoint tensor values differ")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError("checkpoint state keys differ")
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError("checkpoint state sequence differs")
        for left, right in zip(actual, expected, strict=True):
            assert_state_equal(left, right)
    elif actual != expected:
        raise AssertionError("checkpoint state value differs")
