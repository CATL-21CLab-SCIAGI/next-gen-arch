from __future__ import annotations

import pytest

from archlab.megatron.export_eval import _lm_eval_model_args, expected_checkpoint_iterations


def test_checkpoint_schedule_covers_each_approximate_10b_boundary_and_final() -> None:
    assert expected_checkpoint_iterations(47_684, 4_768) == [
        4_768,
        9_536,
        14_304,
        19_072,
        23_840,
        28_608,
        33_376,
        38_144,
        42_912,
        47_680,
        47_684,
    ]


@pytest.mark.parametrize(("train_iters", "save_interval"), ((0, 1), (1, 0), (-1, 3)))
def test_checkpoint_schedule_rejects_non_positive_inputs(
    train_iters: int, save_interval: int
) -> None:
    with pytest.raises(ValueError):
        expected_checkpoint_iterations(train_iters, save_interval)


def test_lm_eval_model_args_use_harness_key_value_syntax(tmp_path) -> None:
    assert _lm_eval_model_args(tmp_path / "hf") == (f"pretrained={tmp_path / 'hf'},dtype=bfloat16")
