import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from archlab.data.agentic_sft import PreparedAgenticSFTDataset
from archlab.data.teacher_trajectories import (
    TeacherTrajectoryDataset,
    prepare,
    task_split,
    token_row,
    tokenizer_compatibility,
)


def trajectory(path, **changes):
    values = dict(traj0_prompt_ids=np.array([10, 11]),
                  traj0_response_ids=np.array([12, 13, 90, 91, 14]),
                  traj0_response_mask=np.array([1, 1, 0, 0, 1], dtype=np.int8),
                  traj0_generation_spans=np.array([[0, 2], [4, 5]]))
    values.update(changes)
    np.savez(path, **values)


def test_observations_remain_context_and_targets_are_unshifted(tmp_path):
    path = tmp_path / "trajectory.npz"
    trajectory(path)
    row = token_row(path, max_length=7, vocab_size=100)
    assert row["input_ids"] == [10, 11, 12, 13, 90, 91, 14]
    assert row["loss_mask"] == [0, 0, 1, 1, 0, 0, 1]
    # Causal shifting happens once in upstream: observation 91 predicts generated 14.
    assert [(a, b) for a, b, target in zip(row["input_ids"][:-1], row["input_ids"][1:],
                                          row["loss_mask"][1:], strict=True) if target] == [(11, 12), (12, 13), (91, 14)]
    parquet = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([row]), parquet)
    data = TeacherTrajectoryDataset(str(parquet), None, {"max_length": 7}, processor=object())
    assert data[0]["loss_mask"].tolist() == row["loss_mask"]
    with pytest.raises(ValueError, match="supervised native EOS"):
        PreparedAgenticSFTDataset(str(parquet), None, {"max_length": 7})


def test_rejects_overlength_without_silent_prefix_training(tmp_path):
    path = tmp_path / "trajectory.npz"
    trajectory(path)
    with pytest.raises(OverflowError):
        token_row(path, max_length=6, vocab_size=100)
    with pytest.raises(ValueError, match="not shared"):
        token_row(path, max_length=7, vocab_size=100, excluded_token_ids=[90])


@pytest.mark.parametrize("changes", [
    {"traj0_response_mask": np.array([1, 1, 1, 0, 1])},
    {"traj0_response_ids": np.array([12, 13, 100, 91, 14])},
    {"traj0_generation_spans": np.array([[0, 3], [2, 5]])},
    {"traj0_response_mask": np.array([1., 1., 0., 0., 1.])},
])
def test_invalid_tokens_or_generated_spans_fail_closed(tmp_path, changes):
    path = tmp_path / "trajectory.npz"
    trajectory(path, **changes)
    with pytest.raises(ValueError):
        token_row(path, max_length=7, vocab_size=100)


def test_split_uses_task_identity_for_every_seed():
    tasks = [f"task-{i}" for i in range(100)]
    assigned = {task: task_split(task) for task in tasks}
    assert set(assigned.values()) == {"train", "validation"}
    assert {task: task_split(task) for task in reversed(tasks)} == assigned


def test_token_meanings_must_match_even_when_ids_fit(tmp_path):
    teacher, student = tmp_path / "teacher", tmp_path / "student"
    for root in (teacher, student):
        root.mkdir()
        (root / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1}}}))
    tokenizer_compatibility(teacher, student)
    (student / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"b": 0, "a": 1}}}))
    with pytest.raises(ValueError, match="mismatch"):
        tokenizer_compatibility(teacher, student)


def collection_fixture(root):
    from archlab.artifacts import sha256_file

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    teacher, student = root / "teacher", root / "student"
    for model in (teacher, student):
        write(model / "tokenizer.json", {"model": {"vocab": {"a": 0}}})
        write(model / "config.json", {"vocab_size": 100})
    tasks = [next(f"task-{i}" for i in range(100) if task_split(f"task-{i}") == split)
             for split in ("train", "validation")]
    heldout = root / "heldout.json"
    write(heldout, {"task_ids": ["external-evaluation-task"]})
    collection = root / "collection"
    manifest = collection / "data/shard-00/manifest.json"
    write(manifest, {"task_ids": tasks})
    reference = root / "reference.json"
    write(reference, {"model_bindings": {"weight": "fixed"}, "model_config_sha256": "config"})
    checksum = sha256_file(manifest)
    write(collection / "CONTRACT.json", {"model": str(teacher), "model_bindings_reference": str(reference),
                                          "tasks": 2, "seeds": [101], "runner_sha256": "runner",
                                          "harness_bindings": {"harness": "fixed"},
                                          "shard_manifest_sha256": {str(manifest.relative_to(collection)): checksum}})
    run = collection / "runs/shard-00"
    write(run / "invocation.json", {"model": str(teacher), "tasks": tasks, "seeds": [101],
                                     "manifest_sha256": checksum, "evaluator_sha256": "runner",
                                     "harness_bindings": {"harness": "fixed"},
                                     "model_bindings": {"weight": "fixed"}, "model_config_sha256": "config"})
    episodes = []
    for task in tasks:
        episodes.append(dict(task=task, seed=101, status="scored", reward=1.0, verifier_completed=True,
                             agent_completed=True, cap_reason="none", length_limited_calls=0))
        session = run / "sessions/step_0" / f"session-{task}"
        write(session / "session.json", {"prompt_uid": task + "-s101", "reward_score": 1.0,
                                          "gradable": True, "agent_completed": True})
        write(session / "trajectory.json", {"trajectories": [{"reward_info": {
            "reward": 1.0, "verifier_finished": True, "verifier_returncode": 0}}]})
        trajectory(session / "trajectory.npz")
    (run / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    return collection, student, heldout, root / "prepared"


def test_complete_collection_exports_audited_task_split(tmp_path):
    args = collection_fixture(tmp_path)
    receipt = prepare(*args, max_length=7)
    assert receipt["rows"] == {"train": 1, "validation": 1}
    assert receipt["target_tokens"] == {"train": 3, "validation": 3}
    assert receipt["rejected"] == {}
    assert receipt["collection_coverage"]["complete"] is True
    assert pq.read_table(args[-1] / "train.parquet")["loss_mask"].to_pylist() == [[0, 0, 1, 1, 0, 0, 1]]


@pytest.mark.parametrize("failure", ["incomplete", "heldout", "model_changed", "ambiguous_session"])
def test_export_fails_closed_before_writing_dataset(tmp_path, failure):
    import shutil

    collection, student, heldout, output = collection_fixture(tmp_path)
    run = collection / "runs/shard-00"
    if failure == "incomplete":
        path = run / "episodes.jsonl"
        path.write_text(path.read_text().splitlines()[0] + "\n")
    elif failure == "heldout":
        heldout.write_text(json.dumps({"task_ids": [json.loads((run / "episodes.jsonl").read_text().splitlines()[0])["task"]]}))
    elif failure == "model_changed":
        path = run / "invocation.json"
        value = json.loads(path.read_text())
        value["model_bindings"] = {"weight": "changed"}
        path.write_text(json.dumps(value))
    else:
        session = next((run / "sessions/step_0").glob("session-*"))
        shutil.copytree(session, session.with_name("session-duplicate"))
    with pytest.raises(ValueError):
        prepare(collection, student, heldout, output, max_length=7)
    assert not output.exists()


def incomplete_collection_fixture(root):
    from archlab.artifacts import sha256_file

    args = collection_fixture(root)
    collection = args[0]
    run = collection / "runs/shard-00"
    for path in (collection / "CONTRACT.json", run / "invocation.json"):
        data = json.loads(path.read_text())
        data["seeds"] = [101, 211]
        path.write_text(json.dumps(data))
    episodes_path = run / "episodes.jsonl"
    records = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    missing = [dict(shard=run.name, task=row["task"], seed=211) for row in records]
    records.extend(dict(task=row["task"], seed=211, status="missing", reward=None) for row in missing)
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    exclusions = root / "exclusions.json"
    exclusions.write_text(json.dumps({
        "format": "archlab.teacher-infrastructure-exclusions.v1",
        "collection_contract_sha256": sha256_file(collection / "CONTRACT.json"),
        "episode_sha256": {str(episodes_path.relative_to(collection)): sha256_file(episodes_path)},
        "missing_samples": missing,
    }))
    return args, exclusions


def test_explicit_failed_episode_exclusions_preserve_incomplete_coverage(tmp_path):
    args, exclusions = incomplete_collection_fixture(tmp_path)
    receipt = prepare(*args, max_length=7, exclusion_contract=exclusions)
    assert receipt["rows"] == {"train": 1, "validation": 1}
    assert receipt["target_tokens"] == {"train": 3, "validation": 3}
    coverage = receipt["collection_coverage"]
    assert (coverage["expected"], coverage["scored"], coverage["complete"]) == (4, 2, False)
    assert coverage["missing"] == json.loads(exclusions.read_text())["missing_samples"]
    assert receipt["rejected"] == {}  # Infrastructure failures never become negative examples.


@pytest.mark.parametrize("failure", ["wrong_collection", "omitted_missing", "duplicate_missing",
                                     "changed_episodes", "unattempted"])
def test_exclusions_are_exact_and_hash_bound(tmp_path, failure):
    from archlab.artifacts import sha256_file

    args, path = incomplete_collection_fixture(tmp_path)
    exclusions = json.loads(path.read_text())
    episodes = args[0] / "runs/shard-00/episodes.jsonl"
    if failure == "wrong_collection":
        exclusions["collection_contract_sha256"] = "different"
    elif failure == "omitted_missing":
        exclusions["missing_samples"].pop()
    elif failure == "duplicate_missing":
        exclusions["missing_samples"].append(exclusions["missing_samples"][0])
    elif failure == "changed_episodes":
        records = [json.loads(line) for line in episodes.read_text().splitlines()]
        records[-1]["seconds"] = 123
        episodes.write_text("".join(json.dumps(row) + "\n" for row in records))
    else:
        episodes.write_text("\n".join(episodes.read_text().splitlines()[:2]) + "\n")
        exclusions["episode_sha256"][str(episodes.relative_to(args[0]))] = sha256_file(episodes)
    path.write_text(json.dumps(exclusions))
    with pytest.raises(ValueError):
        prepare(*args, max_length=7, exclusion_contract=path)
    assert not args[-1].exists()
