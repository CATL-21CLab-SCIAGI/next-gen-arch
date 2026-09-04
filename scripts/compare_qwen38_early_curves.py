#!/usr/bin/env python3
"""Plot token- and wall-time-aligned early Qwen3.8 learning curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ("cross entropy", "mtp loss", "lm loss", "grad-norm", "learning-rate")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_scalars(run: Path) -> dict[str, list[Any]]:
    accumulator = EventAccumulator(
        str(run / "tensorboard"),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    missing = sorted(set(TAGS) - available)
    if missing:
        raise RuntimeError(f"{run} lacks TensorBoard tags: {missing}")
    return {tag: accumulator.Scalars(tag) for tag in TAGS}


def _by_step(events: list[Any]) -> dict[int, Any]:
    return {event.step: event for event in events}


def _trailing_mean(values: list[float], window: int) -> list[float]:
    output = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        output.append(total / min(index + 1, window))
    return output


def _persistent_crossover(
    steps: list[int],
    quarter: list[float],
    full: list[float],
    *,
    persistence: int,
) -> int | None:
    differences = [
        full_value - quarter_value
        for quarter_value, full_value in zip(quarter, full, strict=True)
    ]
    for index, difference in enumerate(differences):
        if difference >= 0:
            continue
        stop = min(len(differences), index + persistence)
        if stop - index == persistence and all(value < 0 for value in differences[index:stop]):
            return steps[index]
    return None


def _median_step_seconds(events: list[Any], *, start_step: int, stop_step: int) -> float:
    selected = [event for event in events if start_step <= event.step <= stop_step]
    deltas = [
        current.wall_time - previous.wall_time
        for previous, current in zip(selected, selected[1:], strict=False)
        if current.wall_time > previous.wall_time
    ]
    if not deltas:
        return math.nan
    return statistics.median(deltas)


def _plot(
    output: Path,
    *,
    common_steps: list[int],
    tokens_per_step: int,
    quarter_ce: list[float],
    full_ce: list[float],
    quarter_events: list[Any],
    full_events: list[Any],
    smoothing_window: int,
) -> None:
    import matplotlib.pyplot as plt

    quarter_smooth = _trailing_mean(quarter_ce, smoothing_window)
    full_smooth = _trailing_mean(full_ce, smoothing_window)
    tokens_millions = [step * tokens_per_step / 1e6 for step in common_steps]
    full_start = full_events[0].wall_time
    full_elapsed = full_events[-1].wall_time - full_start
    quarter_start = quarter_events[0].wall_time
    quarter_wall = [
        event for event in quarter_events if event.wall_time - quarter_start <= full_elapsed
    ]
    full_wall_minutes = [(event.wall_time - full_start) / 60.0 for event in full_events]
    quarter_wall_minutes = [(event.wall_time - quarter_start) / 60.0 for event in quarter_wall]
    quarter_wall_ce = [event.value for event in quarter_wall]
    full_wall_ce = [event.value for event in full_events]

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), constrained_layout=True)
    colors = {"quarter": "#3178c6", "full": "#d97706"}
    axes[0].plot(tokens_millions, quarter_ce, alpha=0.13, color=colors["quarter"])
    axes[0].plot(tokens_millions, full_ce, alpha=0.13, color=colors["full"])
    axes[0].plot(
        tokens_millions,
        quarter_smooth,
        linewidth=2.2,
        color=colors["quarter"],
        label="Quartered (0.95B)",
    )
    axes[0].plot(
        tokens_millions,
        full_smooth,
        linewidth=2.2,
        color=colors["full"],
        label="Full (27.32B)",
    )
    axes[0].set_title("Matched training tokens")
    axes[0].set_xlabel("Consumed tokens (millions)")
    axes[0].set_ylabel("Causal cross-entropy")

    axes[1].plot(
        quarter_wall_minutes,
        quarter_wall_ce,
        alpha=0.12,
        color=colors["quarter"],
    )
    axes[1].plot(full_wall_minutes, full_wall_ce, alpha=0.12, color=colors["full"])
    axes[1].plot(
        quarter_wall_minutes,
        _trailing_mean(quarter_wall_ce, smoothing_window),
        linewidth=2.2,
        color=colors["quarter"],
        label="Quartered (0.95B)",
    )
    axes[1].plot(
        full_wall_minutes,
        _trailing_mean(full_wall_ce, smoothing_window),
        linewidth=2.2,
        color=colors["full"],
        label="Full (27.32B)",
    )
    axes[1].set_title("Matched training wall time")
    axes[1].set_xlabel("Training time (minutes)")
    axes[1].set_ylabel("Causal cross-entropy")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    figure.suptitle(
        f"Quartered vs full Qwen3.8 early learning ({smoothing_window}-step trailing mean)"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarter-run", required=True, type=Path)
    parser.add_argument("--full-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokens-per-step", type=int, default=1_048_576)
    parser.add_argument("--smoothing-window", type=int, default=20)
    return parser


def main() -> None:
    args = _parser().parse_args()
    quarter = _read_scalars(args.quarter_run)
    full = _read_scalars(args.full_run)
    quarter_maps = {tag: _by_step(events) for tag, events in quarter.items()}
    full_maps = {tag: _by_step(events) for tag, events in full.items()}
    common_steps = sorted(
        set(quarter_maps["cross entropy"]) & set(full_maps["cross entropy"])
    )
    if not common_steps:
        raise RuntimeError("the two runs have no common training steps")
    quarter_ce = [quarter_maps["cross entropy"][step].value for step in common_steps]
    full_ce = [full_maps["cross entropy"][step].value for step in common_steps]
    quarter_smooth = _trailing_mean(quarter_ce, args.smoothing_window)
    full_smooth = _trailing_mean(full_ce, args.smoothing_window)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "early-learning-curve.csv"
    fieldnames = [
        "step",
        "training_tokens",
        "quarter_cross_entropy",
        "full_cross_entropy",
        "quarter_cross_entropy_smoothed",
        "full_cross_entropy_smoothed",
        "quarter_mtp_loss",
        "full_mtp_loss",
        "quarter_grad_norm",
        "full_grad_norm",
        "learning_rate",
    ]
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, step in enumerate(common_steps):
            writer.writerow(
                {
                    "step": step,
                    "training_tokens": step * args.tokens_per_step,
                    "quarter_cross_entropy": quarter_ce[index],
                    "full_cross_entropy": full_ce[index],
                    "quarter_cross_entropy_smoothed": quarter_smooth[index],
                    "full_cross_entropy_smoothed": full_smooth[index],
                    "quarter_mtp_loss": quarter_maps["mtp loss"][step].value,
                    "full_mtp_loss": full_maps["mtp loss"][step].value,
                    "quarter_grad_norm": quarter_maps["grad-norm"][step].value,
                    "full_grad_norm": full_maps["grad-norm"][step].value,
                    "learning_rate": full_maps["learning-rate"][step].value,
                }
            )
    os.replace(temporary, csv_path)

    start_step = min(20, common_steps[-1])
    quarter_step_seconds = _median_step_seconds(
        quarter["cross entropy"],
        start_step=start_step,
        stop_step=common_steps[-1],
    )
    full_step_seconds = _median_step_seconds(
        full["cross entropy"],
        start_step=start_step,
        stop_step=common_steps[-1],
    )
    full_start_wall = full["cross entropy"][0].wall_time
    full_elapsed = full_maps["cross entropy"][common_steps[-1]].wall_time - full_start_wall
    quarter_start_wall = quarter["cross entropy"][0].wall_time
    quarter_same_wall = [
        event
        for event in quarter["cross entropy"]
        if event.wall_time - quarter_start_wall <= full_elapsed
    ]
    quarter_contract = json.loads((args.quarter_run / "RUN_CONTRACT.json").read_text())
    full_contract = json.loads((args.full_run / "RUN_CONTRACT.json").read_text())
    endpoint_window = min(args.smoothing_window, len(common_steps))
    crossover = _persistent_crossover(
        common_steps,
        quarter_smooth,
        full_smooth,
        persistence=endpoint_window,
    )
    summary = {
        "snapshot_at_unix": time.time(),
        "comparison_metric": "causal cross entropy; MTP contribution excluded",
        "tokens_per_step": args.tokens_per_step,
        "common_window": {
            "steps": common_steps[-1],
            "tokens": common_steps[-1] * args.tokens_per_step,
            "quarter_endpoint_mean": statistics.mean(quarter_ce[-endpoint_window:]),
            "full_endpoint_mean": statistics.mean(full_ce[-endpoint_window:]),
            "full_minus_quarter_endpoint": statistics.mean(full_ce[-endpoint_window:])
            - statistics.mean(quarter_ce[-endpoint_window:]),
            "persistent_smoothed_crossover_step": crossover,
            "persistent_smoothed_crossover_tokens": (
                crossover * args.tokens_per_step if crossover is not None else None
            ),
        },
        "throughput": {
            "quarter_median_seconds_per_step": quarter_step_seconds,
            "full_median_seconds_per_step": full_step_seconds,
            "quarter_tokens_per_second": args.tokens_per_step / quarter_step_seconds,
            "full_tokens_per_second": args.tokens_per_step / full_step_seconds,
            "full_to_quarter_step_time_ratio": full_step_seconds / quarter_step_seconds,
        },
        "same_wall_time": {
            "seconds": full_elapsed,
            "full_step": common_steps[-1],
            "full_tokens": common_steps[-1] * args.tokens_per_step,
            "full_cross_entropy_smoothed": full_smooth[-1],
            "quarter_step": quarter_same_wall[-1].step,
            "quarter_tokens": quarter_same_wall[-1].step * args.tokens_per_step,
            "quarter_cross_entropy_smoothed": _trailing_mean(
                [event.value for event in quarter_same_wall],
                args.smoothing_window,
            )[-1],
        },
        "parameter_count": {
            "quarter": quarter_contract["parameter_count"],
            "full": full_contract["parameter_count"],
        },
        "batch": {
            "quarter": quarter_contract["batch"],
            "full": full_contract["batch"],
        },
        "source_commit": {
            "quarter": quarter_contract.get("source_commit"),
            "full": full_contract.get("source_commit"),
        },
        "caveat": (
            "This is not a pure parameter-scaling ablation: the historical quarter run uses "
            "the 0fdc753 quarter MTP and sigmoid attention gate, while the full run uses the "
            "344e678 source-faithful full MTP and SiLU gate."
        ),
    }
    _atomic_json(args.output_dir / "early-learning-summary.json", summary)
    _plot(
        args.output_dir / "early-learning-curve.png",
        common_steps=common_steps,
        tokens_per_step=args.tokens_per_step,
        quarter_ce=quarter_ce,
        full_ce=full_ce,
        quarter_events=quarter["cross entropy"],
        full_events=[full_maps["cross entropy"][step] for step in common_steps],
        smoothing_window=args.smoothing_window,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
