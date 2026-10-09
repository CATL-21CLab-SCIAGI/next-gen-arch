"""Audited, problem-disjoint splits and upstream symbolic math rewards."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8192)
def gold_expression(answer):
    from math_verify import parse

    return parse(r"\boxed{" + answer + "}", fallback_mode="no_fallback", parsing_timeout=2)


def reasoning_complete(text):
    """An earlier surplus closing tag cannot close a later reasoning block."""
    depth = 0
    for tag in re.finditer(r"</?think>", text):
        if tag.group() == "<think>":
            depth += 1
        else:
            depth = max(0, depth - 1)
    return depth == 0


def math_reward(completion, answer):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    if isinstance(completion, list):
        completion = completion[-1]["content"]
    if not reasoning_complete(completion):
        return 0.0
    completion = completion.rsplit("</think>", 1)[-1]
    prediction = parse(
        completion,
        extraction_config=[
            LatexExtractionConfig(try_extract_without_anchor=False, boxed_match_priority=0),
            ExprExtractionConfig(try_extract_without_anchor=False),
        ],
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=2,
    )
    if not prediction:
        from archlab.rl.rewards import canonical_math_answer

        scalar = canonical_math_answer(completion.strip().rstrip("."))
        if scalar is not None:
            prediction = gold_expression(scalar)
        else:
            # Base-model answers need not be boxed. Restrict upstream expression
            # extraction to an explicit concluding paragraph ending in math;
            # never mine numbers from an intermediate calculation in the body.
            lines = completion.strip().splitlines()
            for index in range(len(lines) - 1, -1, -1):
                conclusion = lines[index].strip().rstrip(".")
                suffix = "\n".join(lines[index + 1 :])
                if re.search(r"[0-9]|\\boxed", suffix):
                    break
                if (
                    re.search(
                        r"\b(therefore|thus|hence|answer|result|sum|option|equals)\b",
                        conclusion,
                        re.I,
                    )
                    and conclusion
                    and conclusion[-1] in "0123456789})$]"
                ):
                    prediction = parse(
                        conclusion,
                        extraction_config=[ExprExtractionConfig()],
                        fallback_mode="no_fallback",
                        extraction_mode="first_match",
                        parsing_timeout=2,
                    )
                    break
    return float(
        bool(prediction) and verify(gold_expression(answer), prediction, timeout_seconds=2)
    )


def split_rows(rows, heldout_count=128):
    from math_verify import verify

    grouped = {}
    for row in rows:
        grouped.setdefault(row["problem_sha256"], []).append(row)
    eligible, exclusions = [], []
    for key, group in sorted(grouped.items()):
        first = group[0]
        gold = gold_expression(first["expected_answer"])
        reason = None
        if not gold or not verify(gold, gold, timeout_seconds=2):
            reason = "gold_not_symbolically_verifiable"
        elif any(
            not verify(gold, gold_expression(r["expected_answer"]), timeout_seconds=2)
            for r in group[1:]
        ):
            reason = "duplicate_problem_conflicting_or_unverifiable_answers"
        if reason:
            exclusions.extend(dict(uuid=r["uuid"], reason=reason) for r in group)
            continue
        eligible.append(
            dict(
                uuid=first["uuid"],
                problem_sha256=key,
                prompt=first["prompt_text"],
                expected_answer=first["expected_answer"],
                input_ids=first["input_ids"],
            )
        )
        exclusions.extend(
            dict(uuid=r["uuid"], reason="duplicate_problem", retained_uuid=first["uuid"])
            for r in group[1:]
        )
    # SHA ordering is fixed before observing model outputs or rewards.
    eligible.sort(
        key=lambda r: hashlib.sha256(
            ("limite-rl-holdout-v1:" + r["problem_sha256"]).encode()
        ).hexdigest()
    )
    if len(eligible) <= heldout_count:
        raise ValueError("insufficient eligible training data")
    return eligible[heldout_count:], eligible[:heldout_count], exclusions


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from archlab.rl.nemotron_data import sha256_file

    ready = json.loads((args.data / "DATA_READY.json").read_text())
    if ready["manifest_sha256"] != sha256_file(args.data / "MANIFEST.json"):
        raise ValueError("dataset completion marker mismatch")
    manifest = json.loads((args.data / "MANIFEST.json").read_text())
    spec = next(f for f in manifest["files"] if f["path"] == "train.parquet")
    if sha256_file(args.data / "train.parquet") != spec["sha256"]:
        raise ValueError("dataset payload changed")
    rows = pq.read_table(args.data / "train.parquet").to_pylist()
    train, heldout, exclusions = split_rows(rows)
    args.output.mkdir(exist_ok=False, parents=True)
    for name, values in [("train", train), ("heldout", heldout), ("exclusions", exclusions)]:
        (args.output / (name + ".jsonl")).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values)
        )
    report = dict(
        source_rows=len(rows),
        train=len(train),
        heldout=len(heldout),
        excluded=len(exclusions),
        verifier="math-verify 0.8.0; symbolic only; no judge fallback",
        source_manifest_sha256=ready["manifest_sha256"],
        files={p.name: sha256_file(p) for p in args.output.glob("*.jsonl")},
    )
    (args.output / "SPLIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
