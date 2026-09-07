"""Read-only TensorBoard comparison aligned to equal consumed-token blocks."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def trailing_means(values, window):
    return [statistics.fmean(values[max(0, i + 1 - window):i + 1]) for i in range(len(values))]


def read_run(label, directory, loss_tag, tokens_per_step, block_tokens):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    directory = Path(directory)
    if block_tokens % tokens_per_step:
        raise ValueError("comparison blocks must contain an integer number of native steps")
    per_block = block_tokens // tokens_per_step
    accumulator = EventAccumulator(str(directory / "tensorboard"), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    events = {event.step: event for event in accumulator.Scalars(loss_tag)}
    points = []
    for block in range(1, max(events) // per_block + 1):
        steps = range((block - 1) * per_block + 1, block * per_block + 1)
        if not all(step in events for step in steps):
            continue
        selected = [events[step] for step in steps]
        points.append({"block": block, "step": selected[-1].step,
                       "tokens": block * block_tokens,
                       "ce": statistics.fmean(item.value for item in selected),
                       "wall_time": selected[-1].wall_time})
    if not points:
        raise ValueError("no complete token blocks")
    validation_tag = f"{loss_tag} validation"
    validation = [{"step": e.step, "tokens": e.step * tokens_per_step, "ce": e.value}
                  for e in accumulator.Scalars(validation_tag)] if validation_tag in tags else []
    contract_file = directory / "RUN_CONTRACT.json"
    contract = json.loads(contract_file.read_text()) if contract_file.exists() else {}
    rates = {e.step: e.value for e in accumulator.Scalars("learning-rate")} if "learning-rate" in tags else {}
    return {"label": label, "directory": str(directory), "loss_tag": loss_tag,
            "tokens_per_optimizer_step": tokens_per_step, "points": points,
            "validation": validation, "validation_loss_tag": validation_tag,
            "latest_learning_rate": rates.get(max(events)),
            "contract": contract}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=4, action="append", required=True,
                        metavar=("LABEL", "DIRECTORY", "LOSS_TAG", "TOKENS_PER_STEP"))
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--block-tokens", type=int, default=8388608)
    parser.add_argument("--smoothing-blocks", type=int, default=20)
    args = parser.parse_args()
    outputs = {kind: args.output_prefix.with_suffix("." + kind) for kind in ("json", "csv", "png")}
    if any(path.exists() for path in outputs.values()):
        raise ValueError("refusing to overwrite comparison evidence")
    if min(args.block_tokens, args.smoothing_blocks) < 1:
        raise ValueError("comparison windows must be positive")
    runs = [read_run(label, path, tag, int(tokens), args.block_tokens)
            for label, path, tag, tokens in args.run]
    milestones = []
    for block in (64, 128, 256, 512, 640, 671, 686, 1192):
        record = {"end_block": block, "tokens": block * args.block_tokens,
                  "window_tokens": args.smoothing_blocks * args.block_tokens}
        for run in runs:
            selected = [p for p in run["points"] if block - args.smoothing_blocks < p["block"] <= block]
            record[run["label"]] = statistics.fmean(p["ce"] for p in selected) if len(selected) == args.smoothing_blocks else None
        milestones.append(record)
    for run in runs:
        points = run["points"]
        run["latest"] = points[-1]
        run["last20_mean"] = statistics.fmean(p["ce"] for p in points[-20:])
        recent = points[-51:]
        deltas = [(b["wall_time"] - a["wall_time"]) / (b["block"] - a["block"])
                  for a, b in zip(recent, recent[1:], strict=False) if b["wall_time"] > a["wall_time"]]
        run["median_seconds_per_block_last50"] = statistics.median(deltas) if deltas else None
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps({"block_tokens": args.block_tokens,
        "smoothing_blocks": args.smoothing_blocks, "milestones": milestones, "runs": runs,
        "notes": ["Training main CE only; auxiliary/MTP losses excluded.",
                  "Non-overlapping equal-token blocks; no extrapolation or interpolation.",
                  "Different architectures, active parameter counts, batches, and optimizers are not a controlled ablation.",
                  "Wall time is elapsed since each run's first scalar, including pauses/checkpoints; not GPU hours."]}, indent=2) + "\n")
    with outputs["csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run", "block", "step", "tokens", "ce", "wall_time"))
        writer.writeheader()
        for run in runs:
            for point in run["points"]:
                writer.writerow({"run": run["label"], **point})
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    limit = runs[0]["points"][-1]["tokens"]
    for run in runs:
        points = [p for p in run["points"] if p["tokens"] <= limit]
        smooth = trailing_means([p["ce"] for p in points], args.smoothing_blocks)
        hours = [(p["wall_time"] - points[0]["wall_time"]) / 3600 for p in points]
        axes[0].plot([p["tokens"] / 1e9 for p in points], smooth, label=run["label"], linewidth=1.8)
        axes[1].plot(hours, smooth, label=run["label"], linewidth=1.8)
    axes[0].set(xlabel="Consumed training tokens (billions)", title="Equal-token comparison")
    axes[1].set(xlabel="Elapsed training wall time (hours)", title="Historical wall-time comparison (not controlled)")
    for axis in axes:
        axis.set_ylabel("Training cross-entropy (nats/token)")
        axis.set_ylim(2.5, 7)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Width-320 DLC baseline vs earlier runs · {args.smoothing_blocks}-block mean (~168M tokens)\n"
                 "CE axis zoomed to 2.5–7; near-uniform initialization is above this range", fontsize=12)
    fig.savefig(outputs["png"], dpi=170)
    plt.close(fig)
    print(json.dumps({"milestones": milestones, "latest": [
        {k: r[k] for k in ("label", "latest", "last20_mean", "validation", "median_seconds_per_block_last50")}
        for r in runs]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
