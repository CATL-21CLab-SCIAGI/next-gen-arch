"""Integrity checks and small utilities for the published metric table."""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_METRICS = REPOSITORY_ROOT / "results" / "key-metrics.csv"
_INSTALLED_METRICS = Path(__file__).parent / "data" / "key-metrics.csv"
DEFAULT_METRICS = _SOURCE_METRICS if _SOURCE_METRICS.exists() else _INSTALLED_METRICS


def load_metrics(path: Path = DEFAULT_METRICS) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_metrics(path: Path = DEFAULT_METRICS) -> dict[str, int]:
    rows = load_metrics(path)
    required = {
        "campaign",
        "scale",
        "budget_regime",
        "variant",
        "parameters",
        "mean_final_bpb",
        "paired_delta_bpb",
        "valid_seeds",
        "throughput_ratio",
        "status",
        "as_of",
    }
    if not rows or not required <= rows[0].keys():
        raise ValueError(f"Metric table is missing required columns: {sorted(required)}")
    seen: set[tuple[str, str, str]] = set()
    baselines = {
        (row["campaign"], row["scale"]): float(row["mean_final_bpb"])
        for row in rows
        if row["variant"] == "baseline" and row["mean_final_bpb"]
    }
    for row in rows:
        key = (row["campaign"], row["scale"], row["variant"])
        if key in seen:
            raise ValueError(f"Duplicate metric row: {key}")
        seen.add(key)
        bpb_text = row["mean_final_bpb"]
        status = row["status"]
        if status not in {"complete", "partial", "failed-nonfinite"}:
            raise ValueError(f"Unknown result status for {key}: {status}")
        if status == "complete" and not bpb_text:
            raise ValueError(f"Completed row has no BPB: {key}")
        if bpb_text:
            bpb = float(bpb_text)
            if not math.isfinite(bpb) or bpb <= 0:
                raise ValueError(f"Invalid BPB for {key}: {bpb}")
        delta = row["paired_delta_bpb"]
        if delta and not math.isfinite(float(delta)):
            raise ValueError(f"Invalid paired delta for {key}: {delta}")
        seeds = int(row["valid_seeds"])
        if seeds < 0 or seeds > 3:
            raise ValueError(f"Invalid seed count for {key}: {seeds}")
        if status == "complete" and seeds != 3:
            raise ValueError(f"Completed row must contain three seeds: {key}")
        if status == "failed-nonfinite" and (bpb_text or seeds):
            raise ValueError(f"Failed row must not publish a BPB: {key}")
        if row["parameters"] and int(row["parameters"]) <= 0:
            raise ValueError(f"Invalid parameter count for {key}")
        if row["throughput_ratio"]:
            throughput = float(row["throughput_ratio"])
            if not math.isfinite(throughput) or throughput <= 0:
                raise ValueError(f"Invalid throughput ratio for {key}: {throughput}")
        date.fromisoformat(row["as_of"])

        baseline = baselines.get((row["campaign"], row["scale"]))
        if baseline is not None and bpb_text and delta and status == "complete":
            expected_delta = float(bpb_text) - baseline
            if not math.isclose(float(delta), expected_delta, abs_tol=2e-9):
                raise ValueError(f"Paired delta disagrees with the published baseline for {key}")
    return {"rows": len(rows), "campaigns": len({row["campaign"] for row in rows})}
