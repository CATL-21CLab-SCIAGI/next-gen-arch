import json

import pytest

from archlab.automodel.loop_regularization import cell_name, cells
from archlab.tracking.loop_regularization_report import (
    comparison,
    complete_wd_groups,
    interpolate,
    load_measurements,
)


def test_interpolation_is_linear_in_measured_compute_and_never_extrapolates():
    points = [dict(compute=100, loss=5), dict(compute=400, loss=2)]
    assert interpolate(points, 250) == 3.5
    assert interpolate(points, 100) == 5
    assert interpolate(points, 99) is None
    assert interpolate(points, 401) is None


def test_tuning_requires_all_six_controls_and_does_not_force_increasing_k():
    points = [
        dict(reference_depth=8, recursions=k, weight_decay=wd, compute=100 * d, loss=loss)
        for k, loss in ((1, 2), (2, 3))
        for d in (1, 2)
        for wd in (0.8,)
    ]
    assert complete_wd_groups(points) == {}
    result = comparison(points)
    assert all(cut["best_k"] == 1 for cut in result["cuts"])
    assert result["optimal_weight_decay"] == []


def test_report_excludes_incomplete_runs_and_rejects_missing_final_eval(tmp_path):
    plan = {
        "runs": [
            dict(cell=c, output=str(tmp_path / cell_name(c)), source_commit="pinned")
            for c in cells()
        ]
    }
    assert load_measurements(plan) == ([], [])
    first = plan["runs"][0]
    root = tmp_path / cell_name(first["cell"])
    root.mkdir()
    contract = dict(repeated_data_sweep=first["cell"], project_commit="pinned")
    done = dict(passed=True, step=100, supervised_tokens=1_000_000_000, training_gpu_seconds=999)
    (root / "RUN_CONTRACT.json").write_text(json.dumps(contract))
    (root / "COMPLETE.json").write_text(json.dumps(done))
    with pytest.raises(ValueError, match="final validation"):
        load_measurements(plan)
    row = dict(step=100, consumed_supervised_tokens=1_000_000_000, training_gpu_seconds=999, loss=3)
    (root / "validation.jsonl").write_text(json.dumps(row) + "\n")
    points, trajectories = load_measurements(plan)
    assert len(points) == len(trajectories) == 1
    assert points[0]["compute"] == 999
    assert points[0]["loss"] == 3
