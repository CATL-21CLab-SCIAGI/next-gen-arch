"""Join independently evaluated scratch variants after both complete."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + 4 * 3600
    while True:
        failures = [
            str(p) for v in ("normal", "simplicial") for p in (args.root / v).glob("*failure.json")
        ]
        if failures:
            raise RuntimeError(f"evaluation failed: {failures}")
        if all((args.root / v / "COMPLETE.json").exists() for v in ("normal", "simplicial")):
            break
        if not args.watch:
            raise RuntimeError("both evaluations must complete first")
        if time.monotonic() > deadline:
            raise TimeoutError("evaluation did not complete in four hours")
        time.sleep(30)
    variants = ("normal", "simplicial")
    complete = {v: json.loads((args.root / v / "COMPLETE.json").read_text()) for v in variants}
    contracts = {v: json.loads((args.root / v / "RUN_CONTRACT.json").read_text()) for v in variants}
    if complete["normal"]["cursor"] != complete["simplicial"]["cursor"]:
        raise ValueError("checkpoint cursors differ")
    for key in (
        "cursor",
        "validation_targets",
        "benchmark_cases_sha256",
        "benchmark_jobs_digest",
        "prompts_sha256",
        "evaluator_sha256",
    ):
        if contracts["normal"][key] != contracts["simplicial"][key]:
            raise ValueError(f"evaluation contracts differ: {key}")
    rows = {
        v: {
            r["id"]: r
            for r in [
                json.loads(x)
                for x in (args.root / v / "multiple-choice.jsonl").read_text().splitlines()
            ]
        }
        for v in variants
    }
    if rows["normal"].keys() != rows["simplicial"].keys():
        raise ValueError("evaluated case IDs differ")
    from archlab.evaluation.deepseek_v41_compare_metrics import mc_summary

    paired = []
    for key, n in rows["normal"].items():
        s = rows["simplicial"][key]
        if (n["task"], n["answer"]) != (s["task"], s["answer"]):
            raise ValueError("case labels differ")
        paired.append(
            {
                "id": key,
                "task": n["task"],
                "result": {"normal": n["result"], "simplicial": s["result"]},
            }
        )
    mc = mc_summary(paired)
    for value in mc.values():
        value.pop("exact_consumed_training_overlaps", None)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "cursor": complete["normal"]["cursor"],
        "heldout_fineweb": {v: complete[v]["heldout_fineweb"] for v in variants},
        "multiple_choice": mc,
        "benchmark_overlap_with_scratch_training": "not assessed",
        "cpu_weight_offload": False,
    }
    result["normal_minus_simplicial_ce"] = (
        result["heldout_fineweb"]["normal"]["cross_entropy"]
        - result["heldout_fineweb"]["simplicial"]["cross_entropy"]
    )
    (args.root / "COMPARISON.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        f"Both scratch variants evaluated at **{result['cursor']['supervised_tokens']:,} supervised training tokens**, step {result['cursor']['step']}. Each used one eight-GPU node, with checksum-verified weights resident on GPU and matching benchmark inputs.",
        "",
        "| Metric | Normal attention | 2-simplicial |",
        "|---|---:|---:|",
    ]
    for key in (
        "cross_entropy",
        "perplexity",
        "top1_token_accuracy",
        "top5_token_accuracy",
        "mean_predictive_entropy",
    ):
        values = [result["heldout_fineweb"][v][key] for v in variants]
        lines.append(f"| FineWeb-Edu {key} | {values[0]:.6f} | {values[1]:.6f} |")
    for task in mc:
        n, s = [complete[v]["multiple_choice"][task] for v in variants]
        lines.append(
            f"| {task} accuracy ({n['questions']} questions) | {100 * n['accuracy']:.2f}% | {100 * s['accuracy']:.2f}% |"
        )
        if task != "mmlu":
            lines.append(
                f"| {task} character-normalized accuracy | {100 * n['accuracy_norm']:.2f}% | {100 * s['accuracy_norm']:.2f}% |"
            )
    lines += [
        "",
        "FineWeb-Edu uses the complete sealed 1M-target held-out set. Multiple-choice results reuse the fixed zero-shot continuation-likelihood protocol and question selection from the prior evaluation. These small benchmark subsets do not establish a broad quality ranking; benchmark overlap with scratch training was not assessed. Paired uncertainty statistics are in [COMPARISON.json](COMPARISON.json).",
        "",
    ]
    (args.root / "REPORT.md").write_text("\n".join(lines))
    (args.root / "COMPLETE.json").write_text(
        json.dumps(
            {"passed": True, "cursor": result["cursor"], "comparison": "COMPARISON.json"}, indent=2
        )
        + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
