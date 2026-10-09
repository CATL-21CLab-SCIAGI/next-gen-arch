"""Read-only warmup, data-continuity, and expert-coverage checks for scratch runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from archlab.reporting.runs import jsonl_rows


def rows(path):
    return jsonl_rows(path, missing_ok=True)


def snapshot(root, version, min_step):
    out = {"utc": datetime.now(timezone.utc).isoformat(), "variants": {}, "errors": []}
    all_rows = {}
    for variant in ["normal", "simplicial"]:
        path = root / f"production-{variant}-{version}"
        data = rows(path / "train-metrics.jsonl")
        all_rows[variant] = data
        failures = [p.name for p in path.glob("*failure.json")]
        if failures:
            out["errors"].append(f"{variant}: {failures}")
        if not data:
            out["variants"][variant] = {"state": "starting", "healthy": False}
            continue
        _initial = json.loads((path / "rank-00-resume.json").read_text())["cursor"]
        step = data[0]["step"] - 1
        tokens = data[0]["consumed_supervised_tokens"] - data[0]["supervised_tokens"]
        for row in data:
            if (
                row["step"] != step + 1
                or row["phase_step"] != row["step"]
                or row["window_cursor"] != row["step"] * 64
                or row["consumed_supervised_tokens"] != tokens + row["supervised_tokens"]
            ):
                out["errors"].append(f"{variant}: cursor discontinuity at{row['step']}")
            for field in ["loss", "gradient_norm_before_clip", "seconds", "learning_rate"]:
                if not math.isfinite(row[field]) or row[field] <= 0:
                    out["errors"].append(f"{variant}: invalid{field} at{row['step']}")
            if (
                row["updated_parameter_tensors"] != {"normal": 566, "simplicial": 590}[variant]
                or row["changed_local_elements"] <= 0
            ):
                out["errors"].append(f"{variant}: incomplete update at{row['step']}")
            step, tokens = row["step"], row["consumed_supervised_tokens"]
        tail = data[-20:]
        last = data[-1]
        seconds = sum(x["wall_seconds"] for x in tail)
        cv = statistics.mean(x["router_load_cv_mean"] for x in tail)
        max_cv = statistics.mean(x["router_load_cv_max"] for x in tail)
        coverage = 1 - last["router_unused_fraction_window"]
        worst = 1 - last["router_worst_unused_fraction_window"]
        completed = sorted((path / "checkpoints").glob("step-*/COMPLETE.json"))
        checkpoint = completed[-1] if completed else path / "checkpoints/MISSING/COMPLETE.json"
        healthy = (
            last["step"] >= min_step
            and len(tail) == 20
            and cv < 2.5
            and max_cv < 4
            and last["router_usage_window_updates"] == 20
            and coverage > 0.99
            and worst > 0.95
            and checkpoint.exists()
            and not failures
        )
        out["variants"][variant] = {
            "healthy": healthy,
            "step": step,
            "tokens": tokens,
            "loss": last["loss"],
            "mean_loss20": statistics.mean(x["loss"] for x in tail),
            "mean_router_cv20": cv,
            "mean_max_layer_router_cv20": max_cv,
            "expert_coverage20": coverage,
            "worst_layer_expert_coverage20": worst,
            "unused_experts_per_batch_mean20": statistics.mean(
                x["router_dead_fraction_mean"] for x in tail
            ),
            "median_update_seconds20": statistics.median(x["wall_seconds"] for x in tail),
            "supervised_tokens_per_second20": sum(x["supervised_tokens"] for x in tail) / seconds,
            "input_tokens_per_second20": sum(x["input_tokens"] for x in tail) / seconds,
            "gradient_norm_range20": [
                min(x["gradient_norm_before_clip"] for x in tail),
                max(x["gradient_norm_before_clip"] for x in tail),
            ],
            "max_memory_allocated_gib": last["max_memory_allocated_gib"],
            "latest_checkpoint_complete": checkpoint.exists(),
            "latest_checkpoint": str(checkpoint.parent),
            "latest": last,
        }
    paired = {row["step"]: row for row in all_rows["normal"]}
    for row in all_rows["simplicial"]:
        if row["step"] in paired:
            for key in [
                "supervised_tokens",
                "input_tokens",
                "consumed_supervised_tokens",
                "window_cursor",
                "learning_rate",
            ]:
                if row[key] != paired[row["step"]][key]:
                    out["errors"].append(f"paired{key} differs at{row['step']}")
    out["healthy"] = all(x["healthy"] for x in out["variants"].values()) and not out["errors"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--version", default="v5")
    p.add_argument("--min-step", type=int, default=100)
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--interval", type=float, default=45)
    a = p.parse_args()
    if not 0 < a.interval <= 60:
        raise ValueError("interval must be in(0,60]")
    a.output.mkdir(parents=True, exist_ok=True)
    for number in range(a.samples):
        state = snapshot(a.root, a.version, a.min_step)
        tmp = a.output / "LATEST.json.tmp"
        tmp.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        tmp.replace(a.output / "LATEST.json")
        with (a.output / "samples.jsonl").open("a") as stream:
            stream.write(json.dumps(state, allow_nan=False) + "\n")
        brief = {
            "utc": state["utc"],
            "healthy": state["healthy"],
            "errors": state["errors"],
            "variants": {
                v: {
                    k: x[k]
                    for k in [
                        "step",
                        "tokens",
                        "loss",
                        "mean_router_cv20",
                        "expert_coverage20",
                        "median_update_seconds20",
                    ]
                    if k in x
                }
                for v, x in state["variants"].items()
            },
        }
        print(json.dumps(brief), flush=True)
        if state["errors"]:
            raise RuntimeError("scratch health checks found an error; seeLATEST.json")
        if state["healthy"]:
            (a.output / "HEALTH_VERIFIED.json").write_text(
                json.dumps(state, indent=2, allow_nan=False) + "\n"
            )
            return
        if number + 1 < a.samples:
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
