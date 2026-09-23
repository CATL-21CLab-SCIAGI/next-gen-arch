"""Token-matched full-finetuning learning curves with a paired loss difference."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from archlab.reporting.v41_full_curves import smooth, snapshot, window_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = {}
    metadata = {}
    starts = {}
    for v in ("normal", "simplicial"):
        data[v], metadata[v] = snapshot(
            args.root / f"production-{v}-v2/train-metrics.jsonl",
            args.output / f"{v}-snapshot.jsonl",
        )
        starts[v] = data[v][0]["consumed_supervised_tokens"] - data[v][0]["supervised_tokens"]
    assert starts["normal"] == starts["simplicial"]
    start = starts["normal"]
    count = min(len(x) for x in data.values())
    paired = {v: r[:count] for v, r in data.items()}
    for n, s in zip(paired["normal"], paired["simplicial"], strict=True):
        for key in (
            "step",
            "phase_step",
            "consumed_supervised_tokens",
            "supervised_tokens",
            "input_tokens",
            "learning_rate",
        ):
            assert n[key] == s[key], (n["step"], key)
    stop = paired["normal"][-1]["consumed_supervised_tokens"]
    width = 5000000
    summary = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "full_finetune_start_tokens": start,
        "matched_tokens": stop,
        "smoothing_tokens": width,
        "variants": {},
        "sources": metadata,
    }
    for v in data:
        summary["variants"][v] = {
            "latest_tokens": data[v][-1]["consumed_supervised_tokens"],
            "latest_step": data[v][-1]["step"],
            "matched_last_10m_training_ce": window_mean(paired[v], start, stop - 10000000, stop),
        }
    summary["normal_minus_simplicial_last_10m"] = (
        summary["variants"]["normal"]["matched_last_10m_training_ce"]
        - summary["variants"]["simplicial"]["matched_last_10m_training_ce"]
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, delta) = plt.subplots(
        2,
        1,
        figsize=(11, 7.5),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [2, 1]},
    )
    colors = {"normal": "#2563eb", "simplicial": "#db4d35"}
    names = {"normal": "Normal attention", "simplicial": "2-simplicial"}
    for v, rows in data.items():
        x, y = smooth(rows, start, width)
        ax.plot(
            x,
            y,
            color=colors[v],
            lw=2,
            label=f"{names[v]} · {rows[-1]['consumed_supervised_tokens'] / 1e6:.2f}M tokens",
        )
    x, n = smooth(paired["normal"], start, width)
    _, s = smooth(paired["simplicial"], start, width)
    diff = n - s
    delta.plot(x, diff, color="#475569", lw=1.5)
    delta.axhline(0, color="#64748b", lw=1)
    delta.fill_between(
        x,
        0,
        diff,
        where=diff >= 0,
        color=colors["simplicial"],
        alpha=0.22,
        label="2-simplicial lower loss",
    )
    delta.fill_between(
        x, 0, diff, where=diff < 0, color=colors["normal"], alpha=0.22, label="Normal lower loss"
    )
    end = max(r[-1]["consumed_supervised_tokens"] for r in data.values()) / 1e6
    for a in (ax, delta):
        a.grid(alpha=0.2)
        a.axvline(stop / 1e6, color="#94a3b8", ls=":", lw=1)
        if end > stop / 1e6:
            a.axvspan(stop / 1e6, end, color="#94a3b8", alpha=0.12)
    ax.set_title(
        "Four-node full fine-tuning · 16 B300 GPUs per variant", loc="left", weight="bold", pad=12
    )
    ax.set_ylabel("Training cross-entropy\n5M-token trailing weighted mean")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.015,
        0.06,
        f"Full fine-tuning starts at {start / 1e6:.2f}M total tokens.\nPaired comparison ends at {stop / 1e6:.2f}M; shaded tail is unmatched progress.",
        transform=ax.transAxes,
        color="#475569",
        fontsize=9,
    )
    delta.set_ylabel("Normal − 2-simplicial\ntraining cross-entropy")
    delta.set_xlabel("Total supervised training tokens (millions)")
    delta.legend(frameon=False, loc="upper right", fontsize=9)
    delta.set_xlim(start / 1e6, end + 1)
    fig.savefig(args.output / "learning-curve-comparison.png", dpi=170)
    fig.savefig(args.output / "learning-curve-comparison.pdf")
    plt.close(fig)
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "matched-training-curves.csv").open("w") as f:
        f.write("step,total_supervised_tokens,normal_ce,simplicial_ce\n")
        for n, s in zip(paired["normal"], paired["simplicial"], strict=True):
            f.write(f"{n['step']},{n['consumed_supervised_tokens']},{n['loss']},{s['loss']}\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
