"""Plot token-matched scratch learning curves and held-out validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def smooth(data, window):
    weights = np.array([r["supervised_tokens"] for r in data], dtype=float)
    weighted = weights * np.array([r["loss"] for r in data], dtype=float)
    w = np.concatenate(([0.0], np.cumsum(weights)))
    y = np.concatenate(([0.0], np.cumsum(weighted)))
    ends = np.arange(1, len(data) + 1)
    starts = np.maximum(0, ends - window)
    return (y[ends] - y[starts]) / (w[ends] - w[starts])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((args.root / "eval-matched-240m-v2/COMPARISON.json").read_text())
    target = comparison["cursor"]["supervised_tokens"]
    histories = {}
    validation = {}
    summary = {"matched_tokens": target, "training_smoothing_updates": 50, "variants": {}}
    for v in ("normal", "simplicial"):
        run = args.root / f"production-{v}-v5"
        data = rows(run / "prior-phase-metrics.jsonl") + rows(run / "train-metrics.jsonl")
        data = [r for r in data if r["consumed_supervised_tokens"] <= target]
        assert [r["step"] for r in data] == list(range(1, comparison["cursor"]["step"] + 1))
        histories[v] = data
        by_step = {r["step"]: r for r in data}
        validation[v] = [
            {
                "tokens": by_step[r["step"]]["consumed_supervised_tokens"],
                "loss": r["loss"],
                "source": "training_validation",
            }
            for r in rows(run / "validation.jsonl")
        ]
        validation[v].append(
            {
                "tokens": target,
                "loss": comparison["heldout_fineweb"][v]["cross_entropy"],
                "source": "matched_checkpoint_evaluation",
            }
        )
        tail = data[-100:]
        mean = sum(r["loss"] * r["supervised_tokens"] for r in tail) / sum(
            r["supervised_tokens"] for r in tail
        )
        summary["variants"][v] = {
            "steps": len(data),
            "last_100_updates_token_weighted_train_ce": mean,
            "validation": validation[v],
            "final_validation": comparison["heldout_fineweb"][v],
        }
    for a, b in zip(histories["normal"], histories["simplicial"], strict=True):
        for key in (
            "step",
            "consumed_supervised_tokens",
            "supervised_tokens",
            "window_cursor",
            "learning_rate",
        ):
            assert a[key] == b[key], (key, a["step"])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, (train, val) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
        layout="constrained",
    )
    colors = {"normal": "#2563eb", "simplicial": "#db4d35"}
    names = {"normal": "Normal attention", "simplicial": "2-simplicial"}
    for v in histories:
        data = histories[v]
        train.plot(
            [r["consumed_supervised_tokens"] / 1e6 for r in data],
            smooth(data, 50),
            color=colors[v],
            linewidth=2,
            label=names[v],
        )
        points = validation[v]
        x = [r["tokens"] / 1e6 for r in points]
        y = [r["loss"] for r in points]
        val.plot(x, y, "o-", color=colors[v], linewidth=2, markersize=6, label=names[v])
        val.scatter(x[-1], y[-1], s=90, color=colors[v], edgecolor="white", linewidth=1.2, zorder=4)
        offset = -16 if v == "normal" else 13
        val.annotate(
            f"{y[-1]:.4f}",
            (x[-1], y[-1]),
            xytext=(-7, offset),
            textcoords="offset points",
            ha="right",
            color=colors[v],
            weight="bold",
        )
    train.set_title(
        "Scratch learning curves · 640-wide, 20 layers · matched data and token budget",
        loc="left",
        weight="bold",
        pad=12,
    )
    train.set_ylabel("Training cross-entropy\n50-update token-weighted mean")
    train.legend(loc="upper right", frameon=False)
    val.set_ylabel("Held-out cross-entropy\nSame 1M FineWeb-Edu targets")
    val.set_xlabel("Supervised training tokens (millions)")
    val.set_ylim(4.0, 4.86)
    train.set_xlim(0, 250)
    for ax in (train, val):
        ax.grid(alpha=0.2)
        ax.axvline(target / 1e6, color="#94a3b8", linestyle=":", linewidth=1)
    train.text(
        0.985,
        0.68,
        f"Both stopped at\n{target / 1e6:.3f}M tokens\n(step {comparison['cursor']['step']:,})",
        transform=train.transAxes,
        ha="right",
        va="top",
        color="#475569",
    )
    val.set_title("Held-out validation · lower is better", loc="left", fontsize=11, pad=8)
    fig.savefig(args.output / "learning-curve-comparison.png", dpi=170)
    fig.savefig(args.output / "learning-curve-comparison.pdf")
    plt.close(fig)
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "training-curves.csv").open("w") as f:
        f.write("step,supervised_tokens,normal_train_ce,simplicial_train_ce\n")
        for a, b in zip(histories["normal"], histories["simplicial"], strict=True):
            f.write(f"{a['step']},{a['consumed_supervised_tokens']},{a['loss']},{b['loss']}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
