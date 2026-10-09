"""Plot and summarize two explicitly selected, token-matched training runs."""

import argparse
import csv
import json
from pathlib import Path

from archlab.artifacts import atomic_write_json
from archlab.reporting.runs import matched_rows, read_training_run, token_smooth, token_window_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", nargs=2, action="append", required=True, metavar=("LABEL", "DIRECTORY")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoothing-tokens", type=int, default=5_000_000)
    parser.add_argument("--summary-tokens", type=int, default=10_000_000)
    parser.add_argument("--current-phase-only", action="store_true")
    parser.add_argument("--title", default="Paired training at matched token counts")
    args = parser.parse_args()
    if len(args.run) != 2 or len({label for label, _ in args.run}) != 2:
        parser.error("provide exactly two runs with distinct labels")
    if min(args.smoothing_tokens, args.summary_tokens) <= 0:
        parser.error("token windows must be positive")
    loaded = [
        read_training_run(path, include_history=not args.current_phase_only) for _, path in args.run
    ]
    paired = matched_rows(loaded[0][0], loaded[1][0])
    stop = paired[0][-1]["consumed_supervised_tokens"]
    start = paired[0][0]["consumed_supervised_tokens"] - paired[0][0]["supervised_tokens"]
    low = max(start, stop - args.summary_tokens)
    summary = {
        "matched_steps": [paired[0][0]["step"], paired[0][-1]["step"]],
        "matched_token_interval": [start, stop],
        "summary_token_interval": [low, stop],
        "smoothing_tokens": args.smoothing_tokens,
        "runs": {},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for index, (label, path) in enumerate(args.run):
        rows, sources = loaded[index]
        summary["runs"][label] = {
            "directory": str(Path(path).resolve()),
            "sources": sources,
            "latest_step": rows[-1]["step"],
            "latest_tokens": rows[-1]["consumed_supervised_tokens"],
            "matched_token_weighted_ce": token_window_mean(paired[index], low, stop),
        }
        (args.output / f"run-{index}-snapshot.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    labels = [label for label, _ in args.run]
    summary["first_minus_second_ce"] = (
        summary["runs"][labels[0]]["matched_token_weighted_ce"]
        - summary["runs"][labels[1]]["matched_token_weighted_ce"]
    )
    atomic_write_json(args.output / "SUMMARY.json", summary, allow_nan=False)
    with (args.output / "matched-training-curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "supervised_tokens", *labels])
        for left, right in zip(*paired, strict=True):
            writer.writerow(
                [left["step"], left["consumed_supervised_tokens"], left["loss"], right["loss"]]
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axis, delta) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    curves = []
    for (label, _), rows in zip(args.run, paired, strict=True):
        x, y = token_smooth(rows, args.smoothing_tokens)
        curves.append(y)
        axis.plot(x / 1e6, y, label=label)
    delta.plot(x / 1e6, curves[0] - curves[1])
    delta.axhline(0, color="gray", linewidth=1)
    axis.set(title=args.title, ylabel="Token-weighted training cross-entropy")
    delta.set(xlabel="Supervised training tokens (millions)", ylabel=f"{labels[0]} − {labels[1]}")
    axis.legend(frameon=False)
    for subplot in (axis, delta):
        subplot.grid(alpha=0.2)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"learning-curve-comparison.{extension}", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
