import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml

from archlab.evaluation.concurrent_reasoning import guard_reason, tail_metrics
from archlab.evaluation.reasoning import generate, sha, summarize


@pytest.fixture
def config():
    return yaml.safe_load(
        (Path(__file__).parents[1] / "recipes/deepseek_v41/eval_reasoning_eval_v1.yaml").read_text()
    )


def test_oracles_twins_and_disjoint_prompts(config):
    rows = generate(config)
    assert len(rows) == 280
    assert rows == generate(config)
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
        data = row["oracle"]
        if row["task"] in ("composition", "join", "retrieval"):
            value = data["x"]
            for f in data["functions"]:
                value = f[value]
            answer = value == data["y"]
        elif row["task"] == "arithmetic":
            answer = all(
                (a + b) % data["modulus"] == t
                for a, b, t in zip(data["a"], data["b"], data["target"], strict=True)
            )
        else:
            # Independent brute-force oracle, not the generator's complement lookup.
            answer = any(
                all(
                    (x + y) % data["modulus"] == t
                    for x, y, t in zip(a, b, data["target"], strict=True)
                )
                for a in data["a"]
                for b in data["b"]
            )
        assert row["expected"] == ("YES" if answer else "NO")
        assert row["facts"] in row["text"] and row["query"] in row["text"]
    for pair in pairs.values():
        assert sorted(r["expected"] for r in pair) == ["NO", "YES"]
        assert len({r["split"] for r in pair}) == 1
        if pair[0]["task"] == "modular_pair_search":
            for axis in (0, 1):
                assert Counter(p[axis] for p in pair[0]["oracle"]["b"]) == Counter(
                    p[axis] for p in pair[1]["oracle"]["b"]
                )
    calibration = {r["text"] for r in rows if r["split"] == "calibration"}
    assert not calibration & {r["text"] for r in rows if r["split"] == "test"}


def test_scoring_pairs_and_missing_data(config, tmp_path):
    rows = generate(config)
    (tmp_path / "labels.json").write_text(json.dumps(rows))
    public = [
        dict(
            id=r["id"], token_audit=dict(all_facts_in_long_window=True, query_in_short_window=True)
        )
        for r in rows
    ]
    (tmp_path / "cases.json").write_text(json.dumps(public))
    manifest = dict(
        config=config,
        cases_sha256=sha(tmp_path / "cases.json"),
        labels_sha256=sha(tmp_path / "labels.json"),
    )
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    for name in ("normal", "simplicial"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "COMPLETE.json").write_text(
            json.dumps(dict(manifest, variant=name, kind="full", matched_training_tokens=100))
        )
        results = [
            dict(id=r["id"], text=r["expected"] if name == "simplicial" else "YES.") for r in rows
        ]
        random.Random(1).shuffle(results)
        (directory / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in results))
    report = summarize(tmp_path, tmp_path / "normal", tmp_path / "simplicial")
    assert not report["calibration_controls_pass"]
    for group in report["groups"].values():
        assert group["delta_pp"] == 100
        assert group["both_twins_paired_statistics"]["simplicial_both_twins_correct"] == 1
        assert group["invalid_outputs"] == [group["n"], 0]
    p = tmp_path / "normal/predictions.jsonl"
    p.write_text(p.read_text() + "\n" + json.dumps(results[0]))
    with pytest.raises(ValueError, match="duplicate"):
        summarize(tmp_path, tmp_path / "normal", tmp_path / "simplicial")


def test_guards_and_partial_metric_line(tmp_path):
    p = tmp_path / "train-metrics.jsonl"
    p.write_text(
        "\n".join(json.dumps(dict(step=i, seconds=2, supervised_tokens=10)) for i in range(70))
        + '\n{"step":'
    )
    row = tail_metrics(p)
    assert row["step"] == 69 and row["samples"] == 64 and row["median_seconds"] == 2
    plan = dict(training_roots={"normal": str(tmp_path)})
    now = row["mtime"]
    assert guard_reason(plan, {"normal": row}, 70, 64, now) is None
    assert "stale" in guard_reason(plan, {"normal": row}, 70, 64, now + 301)
    assert "GPU" in guard_reason(plan, {"normal": row}, 54, 64, now)
    assert "host" in guard_reason(plan, {"normal": row}, 70, 31, now)
    (tmp_path / "TRAINING_COMPLETE.json").touch()
    assert "advances" in guard_reason(plan, {"normal": row}, 70, 64, now)
