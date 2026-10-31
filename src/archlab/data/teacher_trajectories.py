"""Prepare successful frozen-grid agent rollouts for generated-token-only verl SFT.

Token IDs and masks come directly from Uni-Agent's trajectory.npz. Stop tokens
omitted by serving are not synthesized. Complete episodes over the chosen length
are rejected, never truncated. All seeds of a task share one deterministic split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from archlab.artifacts import sha256_file
from archlab.data.agentic_sft import PreparedAgenticSFTDataset
from archlab.evaluation.agentic_curve import completed_records


def task_split(task: str, seed: int = 42, validation_fraction: float = 0.1) -> str:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    bucket = int(hashlib.sha256(f"{seed}:{task}".encode()).hexdigest(), 16) / 2**256
    return "validation" if bucket < validation_fraction else "train"


def tokenizer_compatibility(teacher: Path, student: Path) -> dict:
    """Require common token meanings/BPE; exclude teacher-only special IDs from data."""
    a, b = [json.loads((p / "tokenizer.json").read_text()) for p in (teacher, student)]
    for key in ("model", "normalizer", "pre_tokenizer", "post_processor", "decoder"):
        if a.get(key) != b.get(key):
            raise ValueError(f"Teacher/student tokenizer mismatch: {key}")
    added_a, added_b = [{token["id"]: token for token in item.get("added_tokens", [])} for item in (a, b)]
    if any(added_a[k] != added_b[k] for k in added_a.keys() & added_b.keys()):
        raise ValueError("Teacher/student tokenizer mismatch: shared added_tokens")
    return {"excluded_token_ids": sorted(added_a.keys() ^ added_b.keys()),
            "files": {label: {name: sha256_file(root / name) for name in
                              ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
                              if (root / name).exists()}
                      for label, root in (("teacher", teacher), ("student", student))}}


def token_row(path: Path, *, max_length: int, vocab_size: int, excluded_token_ids=()) -> dict:
    with np.load(path, allow_pickle=False) as data:
        prompt, response, mask = (data[k] for k in
                                  ("traj0_prompt_ids", "traj0_response_ids", "traj0_response_mask"))
        if any(x.ndim != 1 or x.dtype.kind not in "iu" for x in (prompt, response, mask)):
            raise ValueError("Expected one-dimensional integer trajectory arrays")
        if not len(prompt) or len(response) != len(mask) or not np.isin(mask, [0, 1]).all() or not mask.any():
            raise ValueError("Invalid trajectory mask")
        ids = np.concatenate((prompt, response)).astype(np.int64)
        if ids.min() < 0 or ids.max() >= vocab_size:
            raise ValueError("Token outside student vocabulary")
        if np.isin(ids, excluded_token_ids).any():
            raise ValueError("Trajectory contains a token not shared by both tokenizers")
        if len(ids) > max_length:
            raise OverflowError(f"Complete trajectory exceeds {max_length}: {len(ids)}")
        spans = data["traj0_generation_spans"]
        expected = np.zeros(len(response), dtype=np.int8)
        previous = 0
        if spans.ndim != 2 or spans.shape[1] != 2 or spans.dtype.kind not in "iu":
            raise ValueError("Invalid generation spans")
        for start, end in spans:
            if not previous <= start <= end <= len(response):
                raise ValueError("Overlapping or invalid generation spans")
            expected[start:end] = 1
            previous = end
        if not np.array_equal(mask, expected):
            raise ValueError("Generated-token mask and generation spans disagree")
        targets = np.concatenate((np.zeros(len(prompt), dtype=np.int64), mask.astype(np.int64)))
    return dict(input_ids=ids.tolist(), attention_mask=[1] * len(ids), loss_mask=targets.tolist(),
                position_ids=list(range(len(ids))))


class TeacherTrajectoryDataset(PreparedAgenticSFTDataset):
    """Text-only, pretokenized trajectories; no processor or synthetic EOS is used."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        super().__init__(parquet_files, tokenizer, {**dict(config), "require_supervised_eos": False},
                         processor=None, max_samples=max_samples)


def prepare(collection: Path, student: Path, heldout: Path, output: Path, *, max_length: int,
            split_seed: int = 42, validation_fraction: float = 0.1) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    contract = json.loads((collection / "CONTRACT.json").read_text())
    teacher = Path(contract["model"])
    tokenizers = tokenizer_compatibility(teacher, student)
    reference_path = Path(contract["model_bindings_reference"])
    reference = json.loads(reference_path.read_text())
    config = json.loads((student / "config.json").read_text())
    vocab_size = config.get("text_config", config)["vocab_size"]
    forbidden = set(json.loads(heldout.read_text())["task_ids"])
    if output.exists():
        raise FileExistsError(output)
    selected, index, rejected, assignments = {"train": [], "validation": []}, [], Counter(), {}
    bindings = {str(reference_path): sha256_file(reference_path)}
    for relative, checksum in contract["shard_manifest_sha256"].items():
        manifest_path = collection / relative
        if sha256_file(manifest_path) != checksum:
            raise ValueError("Collection manifest changed")
        manifest = json.loads(manifest_path.read_text())
        tasks = manifest["task_ids"]
        if forbidden.intersection(tasks) or set(assignments).intersection(tasks):
            raise ValueError("Heldout overlap or repeated task across shards")
        assignments.update({task: task_split(task, split_seed, validation_fraction) for task in tasks})
        run = collection / "runs" / manifest_path.parent.name
        invocation = json.loads((run / "invocation.json").read_text())
        if (invocation["manifest_sha256"] != checksum or invocation["seeds"] != contract["seeds"]
                or invocation["evaluator_sha256"] != contract["runner_sha256"]
                or invocation["model"] != str(teacher.resolve()) or invocation["tasks"] != tasks
                or invocation["harness_bindings"] != contract["harness_bindings"]
                or invocation["model_bindings"] != reference["model_bindings"]
                or invocation["model_config_sha256"] != reference["model_config_sha256"]):
            raise ValueError("Run invocation differs from collection contract")
        bindings[str(run / "invocation.json")] = sha256_file(run / "invocation.json")
        episodes_path = run / "episodes.jsonl"
        records = [json.loads(line) for line in episodes_path.read_text().splitlines()]
        expected, scored = completed_records(records, tasks, contract["seeds"])
        if set(scored) != expected:
            raise ValueError(f"Collection is incomplete: {run}")
        bindings[str(episodes_path)] = sha256_file(episodes_path)
        sessions = {}
        for path in sorted((run / "sessions").glob("step_*/session-*/session.json")):
            session = json.loads(path.read_text())
            if session.get("reward_score") == 1.0 and session.get("gradable") and session.get("agent_completed"):
                sessions.setdefault(session["prompt_uid"], []).append(path)
        for (task, seed), episode in scored.items():
            if episode["reward"] != 1.0:
                rejected["unsuccessful"] += 1
                continue
            if (not episode.get("verifier_completed") or not episode.get("agent_completed")
                    or episode.get("cap_reason") != "none" or episode.get("length_limited_calls", 0)):
                rejected["incomplete_or_capped"] += 1
                continue
            paths = sessions.get(f"{task}-s{seed}", [])
            if len(paths) != 1:
                raise ValueError(f"Successful episode needs one unambiguous session: {task} seed {seed}")
            path = paths[0].parent
            trajectories = json.loads((path / "trajectory.json").read_text())["trajectories"]
            if len(trajectories) != 1:
                raise ValueError("Expected one selected trajectory")
            reward = trajectories[0]["reward_info"]
            if reward.get("reward") != 1.0 or not reward.get("verifier_finished") or reward.get("verifier_returncode") != 0:
                raise ValueError("Trajectory verifier receipt disagrees with episode")
            try:
                row = token_row(path / "trajectory.npz", max_length=max_length, vocab_size=vocab_size,
                                excluded_token_ids=tokenizers["excluded_token_ids"])
            except OverflowError:
                rejected["overlength"] += 1
                continue
            split = assignments[task]
            selected[split].append(row)
            index.append(dict(task=task, seed=seed, split=split, source=str(path),
                              trajectory_sha256=sha256_file(path / "trajectory.npz"),
                              input_tokens=len(row["input_ids"]), target_tokens=sum(row["loss_mask"])))
    if len(assignments) != contract["tasks"] or not all(selected.values()):
        raise ValueError("Incomplete task membership or empty training/validation split")
    output.mkdir(parents=True)
    schema = pa.schema([(name, pa.list_(pa.int64())) for name in
                        ("input_ids", "attention_mask", "loss_mask", "position_ids")])
    for split, rows in selected.items():
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), output / f"{split}.parquet")
    (output / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index))
    receipt = dict(format="archlab.teacher-trajectory-sft.v1", collection=str(collection.resolve()),
                   collection_contract_sha256=sha256_file(collection / "CONTRACT.json"),
                   heldout_manifest_sha256=sha256_file(heldout), tokenizers=tokenizers,
                   student=str(student.resolve()), max_length=max_length, truncation="reject-complete-episode",
                   loss_mask_alignment="unshifted-target-token", stop_tokens="omitted-by-serving-not-synthesized",
                   split_seed=split_seed, validation_fraction=validation_fraction, task_assignments=assignments,
                   rows={k: len(v) for k, v in selected.items()}, rejected=dict(rejected),
                   target_tokens={k: sum(r["target_tokens"] for r in index if r["split"] == k) for k in selected},
                   source_bindings=bindings,
                   files={p.name: sha256_file(p) for p in output.iterdir()})
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("collection", "student", "heldout", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    receipt = prepare(**vars(args))
    print(json.dumps({key: receipt[key] for key in ("rows", "rejected", "target_tokens")}))


if __name__ == "__main__":
    main()
