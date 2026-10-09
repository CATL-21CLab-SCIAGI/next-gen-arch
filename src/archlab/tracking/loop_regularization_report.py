"""Report the preregistered V4.1 Figure 5 transfer from new measurements only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from archlab.artifacts import atomic_write_json
from archlab.automodel.loop_regularization import WEIGHT_DECAYS, cell_name, cells, validate_cell


def load_measurements(plan):
    """Only exact-budget, completed cells enter the final-loss comparison."""
    if sorted(cell_name(r["cell"]) for r in plan["runs"]) != sorted(cell_name(c) for c in cells()):
        raise ValueError("report plan differs from the preregistered 222-cell grid")
    completed, trajectories = [], []
    for run in plan["runs"]:
        cell, root = run["cell"], Path(run["output"])
        validate_cell(cell)
        validation = root / "validation.jsonl"
        rows = (
            [json.loads(line) for line in validation.read_text().split("\n")[:-1] if line]
            if validation.exists()
            else []
        )
        for row in rows:
            if row.get("consumed_supervised_tokens", 0) > 0:
                if not math.isfinite(row["loss"]) or row["training_gpu_seconds"] <= 0:
                    raise ValueError(f"invalid validation measurement: {root}")
                trajectories.append(
                    dict(
                        cell=cell_name(cell),
                        **cell,
                        loss=row["loss"],
                        compute=row["training_gpu_seconds"],
                        tokens=row["consumed_supervised_tokens"],
                        step=row["step"],
                    )
                )
        marker = root / "COMPLETE.json"
        if not marker.exists():
            continue
        done = json.loads(marker.read_text())
        contract = json.loads((root / "RUN_CONTRACT.json").read_text())
        if (
            done.get("passed") is not True
            or done["supervised_tokens"] != 1_000_000_000
            or contract["repeated_data_sweep"] != cell
            or contract["project_commit"] != run["source_commit"]
        ):
            raise ValueError(f"completed run differs from its registered contract: {root}")
        final = next((row for row in rows if row["step"] == done["step"]), None)
        if (
            final is None
            or final["consumed_supervised_tokens"] != done["supervised_tokens"]
            or final["training_gpu_seconds"] != done["training_gpu_seconds"]
        ):
            raise ValueError(f"completed run lacks its final validation measurement: {root}")
        completed.append(
            dict(
                cell=cell_name(cell),
                **cell,
                loss=final["loss"],
                compute=final["training_gpu_seconds"],
                source=str(root),
                source_commit=run["source_commit"],
            )
        )
    return completed, trajectories


def interpolate(points, budget):
    """Linear interpolation in measured compute; never extrapolate."""
    ordered = sorted(points, key=lambda p: p["compute"])
    for point in ordered:
        if point["compute"] == budget:
            return point["loss"]
    for left, right in zip(ordered, ordered[1:], strict=False):
        if left["compute"] < budget < right["compute"]:
            fraction = (budget - left["compute"]) / (right["compute"] - left["compute"])
            return left["loss"] + fraction * (right["loss"] - left["loss"])
    return None


def envelope(points):
    """Lower convex envelope of the Pareto frontier in log GPU-seconds."""
    result, best = [], math.inf
    for point in sorted(points, key=lambda p: (p["compute"], p["loss"])):
        if point["loss"] >= best:
            continue
        best = point["loss"]
        while len(result) >= 2:
            a, b = result[-2:]
            cross = (math.log(b["compute"]) - math.log(a["compute"])) * (
                point["loss"] - a["loss"]
            ) - (b["loss"] - a["loss"]) * (math.log(point["compute"]) - math.log(a["compute"]))
            if cross > 0:
                break
            result.pop()
        result.append(point)
    return result


def complete_wd_groups(points):
    groups = defaultdict(list)
    for point in points:
        groups[(point["reference_depth"], point["recursions"])].append(point)
    return {
        key: min(group, key=lambda p: p["loss"])
        for key, group in groups.items()
        if {p["weight_decay"] for p in group} == set(WEIGHT_DECAYS)
    }


def comparison(points):
    fixed = [p for p in points if p["weight_decay"] == 0.8]
    ladders = defaultdict(list)
    for point in fixed:
        ladders[point["recursions"]].append(point)
    cuts = []
    if points:
        low, high = min(p["compute"] for p in points), max(p["compute"] for p in points)
        budgets = [math.exp(math.log(low) + i / 8 * math.log(high / low)) for i in range(9)]
        for budget in budgets:
            losses = {k: interpolate(group, budget) for k, group in sorted(ladders.items())}
            losses = {k: v for k, v in losses.items() if v is not None}
            if len(losses) >= 2:
                cuts.append(dict(compute=budget, losses=losses, best_k=min(losses, key=losses.get)))
    tuned = complete_wd_groups(points)
    k1_tuned = [p for (d, k), p in sorted(tuned.items()) if k == 1]
    tuned_pool = [
        p
        for best in k1_tuned
        for p in points
        if p["reference_depth"] == best["reference_depth"]
        and p["weight_decay"] == best["weight_decay"]
    ]
    k1_fixed = sorted((p for p in fixed if p["recursions"] == 1), key=lambda p: p["compute"])
    # Match the published recipe convention; retain this cell in CSV/cuts/gains.
    eligible = [p for p in fixed if (p["reference_depth"], p["recursions"]) != (4, 6)]
    frontiers = [envelope(eligible), envelope(tuned_pool)]
    if k1_fixed:
        cap = k1_fixed[-1]["compute"]
        for i, frontier in enumerate(frontiers):
            end = next(
                (j + 1 for j, p in enumerate(frontier) if p["compute"] >= cap), len(frontier)
            )
            frontiers[i] = frontier[:end]
    return dict(
        cuts=cuts,
        recipes={
            "K1 · WD 0.8": k1_fixed,
            "K1 · tuned WD": sorted(k1_tuned, key=lambda p: p["compute"]),
            "Loop frontier · WD 0.8": frontiers[0],
            "Loop frontier · K1-tuned WD": frontiers[1],
        },
        optimal_weight_decay=[
            dict(reference_depth=d, recursions=k, weight_decay=p["weight_decay"])
            for (d, k), p in sorted(tuned.items())
        ],
    )


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_report(output, points, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), layout="constrained")
    fixed = {(p["reference_depth"], p["recursions"]): p for p in points if p["weight_decay"] == 0.8}
    for cut in analysis["cuts"]:
        axes[0].plot(
            list(cut["losses"]),
            list(cut["losses"].values()),
            "o-",
            label=f"{cut['compute'] / 3600:.1f}",
        )
    axes[0].set(
        title="Matched-compute cuts", xlabel="Recursions K", ylabel="Interpolated validation loss"
    )
    for d in (6, 8, 10, 12, 14):
        if (d, 1) not in fixed:
            continue
        ks = sorted(k for depth, k in fixed if depth == d)
        axes[1].plot(
            ks,
            [fixed[d, k]["loss"] - fixed[d, 1]["loss"] for k in ks],
            "o-",
            label=f"L{fixed[d, 1]['stored_layers']}",
        )
    axes[1].axhline(0, color="gray", linewidth=0.6)
    axes[1].set(title="Marginal gain · WD 0.8", xlabel="Recursions K", ylabel="Loss(K) − Loss(K1)")
    for label, recipe in analysis["recipes"].items():
        if recipe:
            axes[2].plot(
                [p["compute"] / 3600 for p in recipe],
                [p["loss"] for p in recipe],
                "o-",
                label=label,
            )
    axes[2].set(
        title="Scaling recipes",
        xlabel="Training B300 GPU-hours",
        ylabel="Final validation loss",
        xscale="log",
    )
    depths, ks = (6, 8, 10, 12, 14), (1, 2, 3, 4, 6, 8, 12)
    heat = np.full((len(depths), len(ks)), np.nan)
    for point in analysis["optimal_weight_decay"]:
        d, k = point["reference_depth"], point["recursions"]
        if d in depths and k in ks:
            heat[depths.index(d), ks.index(k)] = point["weight_decay"]
    axes[3].imshow(
        np.ma.masked_invalid(heat),
        cmap="viridis",
        vmin=min(WEIGHT_DECAYS),
        vmax=max(WEIGHT_DECAYS),
        aspect="auto",
    )
    for i, j in zip(*np.where(np.isfinite(heat)), strict=True):
        axes[3].text(j, i, f"{heat[i, j]:g}", ha="center", va="center", color="white")
    axes[3].set(title="Optimal matrix weight decay", xlabel="Recursions K", ylabel="Stored layers")
    axes[3].set_xticks(range(len(ks)), ks)
    axes[3].set_yticks(range(len(depths)), [d * 5 // 2 for d in depths])
    for i, axis in enumerate(axes[:3]):
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=7, title="GPU-hours" if i == 0 else None)
        else:
            axis.text(
                0.5,
                0.5,
                "Awaiting completed cells",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.grid(alpha=0.15)
    state = "Complete grid" if len(points) == len(cells()) else "PARTIAL — no conclusion yet"
    fig.suptitle(f"V4.1 repeated-data transfer · {len(points)}/222 cells · {state}", fontsize=13)
    fig.savefig(output / "figure5.png", dpi=180)
    fig.savefig(output / "figure5.pdf")
    plt.close(fig)


def report(plan_path, output):
    plan = json.loads(plan_path.read_text())
    points, trajectories = load_measurements(plan)
    analysis = comparison(points)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        expected_cells=len(cells()),
        completed_cells=len(points),
        partial=len(points) != len(cells()),
        generated_at_epoch=time.time(),
        compute_axis="B300 training GPU-seconds; max rank update duration × 16; excludes validation/checkpoint IO",
        interpolation="linear in measured GPU-seconds; nine log-spaced observed budgets; no extrapolation",
        tuning="all six WD measurements required; recipe tuning selects K1's WD at each stored size",
        frontier="lower convex envelope in log compute; fixed frontier excludes reference depth4/K6 as published; retain first point at or beyond largest K1 budget",
        scope="V4.1 protocol transfer; cannot numerically replicate the paper's dense-model FLOP axis",
        reference_data_fallback=False,
        **analysis,
    )
    fields = [
        "cell",
        "reference_depth",
        "hidden_size",
        "stored_layers",
        "recursions",
        "weight_decay",
        "compute",
        "loss",
    ]
    write_csv(output / "completed-cells.csv", points, fields + ["source", "source_commit"])
    write_csv(output / "validation-trajectories.csv", trajectories, fields + ["tokens", "step"])
    write_csv(output / "preferred-recursion.csv", analysis["cuts"], ["compute", "best_k"])
    atomic_write_json(output / "REPORT.json", metadata)
    plot_report(output, points, analysis)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    previous = None
    while True:
        plan = json.loads(args.plan.read_text())
        signature = [
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for run in plan["runs"]
            for name in ("validation.jsonl", "COMPLETE.json")
            if (path := Path(run["output"]) / name).exists()
        ]
        if signature != previous:
            value = report(args.plan, args.output)
            print(
                json.dumps({k: value[k] for k in ("completed_cells", "expected_cells", "partial")}),
                flush=True,
            )
            previous = signature
        if not args.watch:
            break
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
