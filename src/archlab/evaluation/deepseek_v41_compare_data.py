"""Seal shared, deterministic evaluation cases for two full V4.1 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized_hash(text):
    return hashlib.sha256(" ".join(unicodedata.normalize("NFC", text).split()).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def build_jobs(tokenizer, cases, templates):
    from jinja2 import Environment, StrictUndefined

    env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)
    jobs = []
    for case in cases:
        prompt_name = "mmlu" if case["task"] == "mmlu" else "arc_challenge"
        context = env.from_string(templates["multiple_choice"][prompt_name]).render(**case)
        options = (
            [chr(65 + i) for i in range(len(case["choices"]))]
            if case["task"] == "mmlu"
            else case["choices"]
        )
        pairs = []
        for option in options:
            prefix_text = context.rstrip()
            continuation = context[len(prefix_text) :] + " " + option
            prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
            combined = tokenizer.encode(prefix_text + continuation, add_special_tokens=False)
            if not prefix or combined[: len(prefix)] != prefix or len(combined) <= len(prefix):
                raise ValueError(f"ambiguous answer token boundary: {case['id']}")
            pairs.append((prefix, combined[len(prefix) :]))
        if all(len(t) == 1 and p == pairs[0][0] for p, t in pairs):
            jobs.append(
                {
                    "id": case["id"],
                    "input_ids": pairs[0][0],
                    "prefix_length": len(pairs[0][0]),
                    "fast_targets": [t[0] for _, t in pairs],
                }
            )
        else:
            for choice, (prefix, target) in enumerate(pairs):
                jobs.append(
                    {
                        "id": case["id"],
                        "choice": choice,
                        "input_ids": prefix + target[:-1],
                        "prefix_length": len(prefix),
                        "targets": target,
                    }
                )
    # Minimize unequal work within each distributed round without packing or
    # batching sequences together. The same ordering is used for both models.
    return sorted(jobs, key=lambda row: (len(row["input_ids"]), row["id"], row.get("choice", -1)))


def jobs_digest(jobs):
    return hashlib.sha256(
        json.dumps(jobs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare(args):
    import yaml
    from datasets import Dataset
    from transformers import PreTrainedTokenizerFast

    pool = read_jsonl(args.capability / "cases.jsonl")
    source = json.loads((args.capability / "manifest.json").read_text())
    if sha(args.capability / "cases.jsonl") != source["cases_sha256"]:
        raise ValueError("capability source changed")
    cases = []
    subjects = sorted({r["subject"] for r in pool if r["task"] == "mmlu"})
    if len(subjects) != 57:
        raise ValueError("expected all 57 MMLU subjects")
    for subject in subjects:
        candidates = [r for r in pool if r["task"] == "mmlu" and r["subject"] == subject]
        cases += random.Random(f"{args.seed}:mmlu:{subject}").sample(
            candidates, args.mmlu_per_subject
        )
    arc = [r for r in pool if r["task"] == "arc_challenge"]
    cases += random.Random(f"{args.seed}:arc").sample(arc, args.arc_count)
    piqa = Dataset.from_file(str(args.piqa))
    if len(piqa) != 1838:
        raise ValueError("use the labeled PIQA validation split")
    for i in sorted(random.Random(f"{args.seed}:piqa").sample(range(len(piqa)), args.piqa_count)):
        r = piqa[i]
        cases.append(
            {
                "id": f"piqa:{i}",
                "task": "piqa",
                "index": i,
                "subject": None,
                "question": r["goal"],
                "choices": [r["sol1"], r["sol2"]],
                "answer": r["label"],
            }
        )
    cases = sorted(cases, key=lambda r: (r["task"], r["index"]))
    checkpoints = {
        v: json.loads((getattr(args, v) / "COMPLETE.json").read_text())
        for v in ("simplicial", "normal")
    }
    if checkpoints["simplicial"]["cursor"] != checkpoints["normal"]["cursor"]:
        raise ValueError("checkpoint cursors differ")
    cursor = checkpoints["simplicial"]["cursor"]
    if cursor["step"] != 552 or cursor["supervised_tokens"] != 91035439:
        raise ValueError("use the matched 91.04M checkpoints")
    train_manifest = json.loads((args.train / "PILOT_READY.json").read_text())
    val_manifest = json.loads((args.validation / "PILOT_READY.json").read_text())
    for path, manifest in ((args.train, train_manifest), (args.validation, val_manifest)):
        if sha(path / "windows.jsonl") != manifest["windows_sha256"]:
            raise ValueError("pilot windows changed")
    train = read_jsonl(args.train / "windows.jsonl")
    random.Random(2234).shuffle(train)
    planned = {r["problem_sha256"] for r in train}
    consumed = {r["problem_sha256"] for r in train[: cursor["step"] * 32]}
    validation = read_jsonl(args.validation / "windows.jsonl")
    if (
        val_manifest["supervised_tokens"] != 1000000
        or sum(r["targets"] for r in validation) != 1000000
    ):
        raise ValueError("use the full sealed one-million-target validation pilot")
    overlap = planned & {r["problem_sha256"] for r in validation}
    if overlap:
        raise ValueError("held-out math overlaps planned training problems")
    for r in cases:
        forms = [
            r["question"],
            r["question"]
            + "\n"
            + "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(r["choices"])),
        ]
        hashes = {normalized_hash(x) for x in forms}
        r["exact_overlap_consumed_training"] = bool(hashes & consumed)
        r["exact_overlap_planned_training"] = bool(hashes & planned)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.assets, local_files_only=True)
    templates = yaml.safe_load(args.prompts.read_text())
    jobs = build_jobs(tokenizer, cases, templates)
    if max(len(r["input_ids"]) for r in jobs) > 16384:
        raise ValueError("benchmark prompt exceeds model context")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cases.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cases)
    )
    (args.output / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False) + "\n")
    manifest = {
        "format": "archlab-v41-paired-full-eval-data-v1",
        "seed": args.seed,
        "checkpoint_cursor": cursor,
        "counts": dict(Counter(r["task"] for r in cases)),
        "mmlu_per_subject": args.mmlu_per_subject,
        "jobs": len(jobs),
        "job_max_context": max(len(j["input_ids"]) for j in jobs),
        "cases_sha256": sha(args.output / "cases.jsonl"),
        "jobs_digest": jobs_digest(jobs),
        "prompts_sha256": sha(args.prompts),
        "capability_sources": source["sources"],
        "piqa_source": {"path": str(args.piqa), "sha256": sha(args.piqa), "split": "validation"},
        "validation_manifest_sha256": sha(args.validation / "PILOT_READY.json"),
        "train_manifest_sha256": sha(args.train / "PILOT_READY.json"),
        "heldout_math_windows": len(validation),
        "heldout_math_targets": 1000000,
        "heldout_math_unique_problems": len({r["problem_sha256"] for r in validation}),
        "exact_math_planned_training_overlaps": 0,
        "exact_consumed_mc_overlaps": [
            r["id"] for r in cases if r["exact_overlap_consumed_training"]
        ],
        "exact_planned_mc_overlaps": [
            r["id"] for r in cases if r["exact_overlap_planned_training"]
        ],
        "contamination_limit": "Exact normalized question hashes only; paraphrases and base pretraining are not audited.",
        "benchmark_scope": "deterministic subsets; not full public benchmark scores",
        "multiple_choice_protocol": "zero-shot plain-text continuation likelihood, no chat, raw and character-normalized accuracy",
        "math_scope": "teacher-forced assistant targets; no generative solve-accuracy claim",
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "capability",
        "piqa",
        "train",
        "validation",
        "assets",
        "prompts",
        "simplicial",
        "normal",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--mmlu-per-subject", type=int, default=20)
    p.add_argument("--arc-count", type=int, default=128)
    p.add_argument("--piqa-count", type=int, default=256)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
