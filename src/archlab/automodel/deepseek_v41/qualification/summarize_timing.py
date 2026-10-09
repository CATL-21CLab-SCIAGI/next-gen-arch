"""Summarize complete, bounded 16-rank runs without hiding latency outliers."""

import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(path):
    records = [json.loads((path / f"rank-{rank:02d}.json").read_text()) for rank in range(16)]
    primary = records[0]
    rows = [r for r in primary["steps"] if r["captured"]]
    for record in records:
        assert record["training_state_saved"] is False
        assert record["source_sha256"] == primary["source_sha256"]
        assert [r["supervised_tokens"] for r in record["steps"]] == [
            r["supervised_tokens"] for r in primary["steps"]
        ]
        assert all(
            math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm_before_clip"])
            for r in record["steps"]
        )
    seconds = [r["wall_seconds"] for r in rows]
    phases = {
        k: statistics.median(r["phase_seconds"][k] for r in rows)
        for k in rows[0].get("phase_seconds", {})
    }
    return {
        "path": str(path),
        "stage": primary["stage"],
        "variant": primary["variant"],
        "width": primary["width"],
        "microbatch": primary.get("microbatch", 4),
        "measured_updates": len(rows),
        "median_seconds": statistics.median(seconds),
        "mean_seconds": statistics.mean(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "valid_tokens_per_second": sum(r["supervised_tokens"] for r in rows) / sum(seconds),
        "valid_tokens": sum(r["supervised_tokens"] for r in rows),
        "physical_slots": sum(r["input_tokens"] for r in rows),
        "median_rank0_phase_seconds": phases,
        "peak_memory_gib": max(r["max_memory_allocated_gib"] for r in primary["steps"]),
        "loss_first": primary["steps"][0]["loss"],
        "loss_last": primary["steps"][-1]["loss"],
        "source": primary["source"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    values = [summarize(path) for path in args.runs]
    args.output.write_text(json.dumps(values, indent=2) + "\n")
    for row in values:
        print(
            f"{row['stage']:14s} {row['variant']:10s} median={row['median_seconds']:.3f}s "
            f"range={row['min_seconds']:.3f}–{row['max_seconds']:.3f}s "
            f"valid-tokens/s={row['valid_tokens_per_second']:.0f}"
        )


if __name__ == "__main__":
    main()
