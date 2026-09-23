import copy
import json
from collections import Counter

import pytest

from archlab.automodel.deepseek_v41_lightweight_original import (
    read_reference,
    select_cases,
    summarize,
    training_ready,
)
from archlab.evaluation.deepseek_v41_compare_metrics import score_choice_vector, score_choices


def fixture_cases():
    result = []
    for subject in range(57):
        for index in range(4):
            result.append(
                {"id": f"mmlu:{subject}:{index}", "task": "mmlu", "subject": str(subject)}
            )
    for task, count in (("arc_challenge", 40), ("piqa", 80)):
        result.extend(
            {"id": f"{task}:{index}", "task": task, "subject": None} for index in range(count)
        )
    for row in result:
        row.update(
            exact_overlap_consumed_training=False,
            exact_overlap_planned_training=False,
            scores={"normal": [0], "simplicial": [0]},
        )
    return result


def test_reference_jsonl_preserves_unicode_line_separators(tmp_path):
    rows = [{"question": "first\u2028second\u0085third"}, {"question": "next record"}]
    path = tmp_path / "reference.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    assert read_reference(path) == rows


def test_subset_is_small_subject_balanced_and_independent_of_scores_and_row_order():
    rows = fixture_cases()
    selected = select_cases(rows)
    assert Counter(row["task"] for row in selected) == {
        "mmlu": 114,
        "arc_challenge": 32,
        "piqa": 64,
    }
    assert set(Counter(row["subject"] for row in selected if row["task"] == "mmlu").values()) == {2}
    changed = copy.deepcopy(list(reversed(rows)))
    for row in changed:
        row["scores"] = {"normal": [-100], "simplicial": [100]}
    assert [r["id"] for r in select_cases(changed)] == [r["id"] for r in selected]
    with pytest.raises(ValueError, match="duplicate"):
        select_cases(rows + [rows[0]])


def test_original_scorer_retains_the_existing_normalization_exactly():
    case = {"id": "piqa:x", "task": "piqa", "choices": ["x", "long answer"], "answer": 0}
    scores = [-2.0, -3.0]
    expected = {"prediction": 0, "normalized_prediction": 1, "accuracy": 1, "accuracy_norm": 0}
    assert score_choice_vector(case, scores) == expected
    assert score_choices(case, {"normal": scores, "simplicial": scores}) == {
        "normal": expected,
        "simplicial": expected,
    }


def test_training_gate_waits_for_real_updates_and_rejects_lost_headroom(tmp_path):
    paths = [tmp_path / "normal", tmp_path / "simplicial"]
    for p in paths:
        p.mkdir()
    assert training_ready(paths) is None
    for p in paths:
        for name, value in {
            "TRAINING_ADMITTED.json": {"passed": True, "contract_digest": "test"},
            "REAL_UPDATE_VERIFIED.json": {"passed": True, "optimizer_step": 1},
            "MEMORY_QUALIFICATION.json": {"passed": True, "minimum_driver_free_gib": 64},
        }.items():
            (p / name).write_text(json.dumps(value))
    assert len(training_ready(paths)) == 2
    (paths[0] / "MEMORY_QUALIFICATION.json").write_text(
        json.dumps({"passed": True, "minimum_driver_free_gib": 63})
    )
    with pytest.raises(ValueError, match="reserve"):
        training_ready(paths)
    (paths[0] / "STOPPED.json").write_text("{}")
    with pytest.raises(RuntimeError, match="stopped"):
        training_ready(paths)


def test_no_observed_regression_is_not_reported_as_certain_equivalence():
    result = {"accuracy": 1, "accuracy_norm": 1}
    rows = [
        {"task": "mmlu", "original": result, "result": {"normal": result, "simplicial": result}}
        for _ in range(5)
    ]
    summary = summarize(rows)["mmlu"]["normal"]["accuracy"]
    assert summary["delta_percentage_points"] == 0
    assert summary["delta_95pct_interval_pp"][0] < 0 < summary["delta_95pct_interval_pp"][1]
    assert summary["significant_regression"] is False
