"""What agentic RL rollouts say about learning and waste, read from Uni-Agent session logs.

Sessions live in <agents>/step_N/<session>/session.json (task = dataset_index, a row of the training snapshot in use
at that step; reward_score; num_turns). One task at one step is a GRPO group. Reports:
- waste: share of groups (and session time) where every rollout failed; GRPO gives them no gradient;
- learning: per task, success on later visits minus the first visit within one data period (a training-pool change
  starts a new period). Unlike pooled training reward, this is not moved by changing the task mix;
- learnable rows: tasks worth sampling next (success on the last visit, never visited, or a deterministic re-check).
usage: python -m archlab.rl.rollouts {learning,waste} <agents dir> [--period first:last ...]
       python -m archlab.rl.rollouts learnable <agents dir> <source parquet> <label> [--write <dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def sessions(agents: Path, first: int = 0, last: int = 10**9, timing: bool = False) -> list[dict]:
    rows = []
    for step_dir in agents.glob("step_*"):
        step = int(step_dir.name.split("_")[1])
        if not first <= step <= last:
            continue
        for path in step_dir.glob("session-*/session.json"):
            try:
                info = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            row = {"step": step, "task": info["dataset_index"], "win": (info.get("reward_score") or 0) > 0,
                   "turns": info.get("num_turns") or 0}
            if timing:
                try:
                    with open(path.parent / "framework.log") as log:
                        start = datetime.strptime(log.read(19), "%Y-%m-%d %H:%M:%S").timestamp()
                    row["seconds"] = max(0.0, path.stat().st_mtime - start)
                except (OSError, ValueError):
                    row["seconds"] = 0.0
            rows.append(row)
    return rows


def bootstrap_mean(values: list[float], *, draws: int = 2000, seed: int = 0) -> tuple[float, list[float]]:
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws))
    return sum(values) / len(values), [boots[int(0.025 * draws)], boots[int(0.975 * draws) - 1]]


def learning(rows: list[dict]) -> dict:
    """Per task: later-visit success minus first-visit success (task-bootstrap CI); > 0 means the policy improved."""
    by_task: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in rows:
        cell = by_task[row["task"]][row["step"]]
        cell[0] += row["win"]
        cell[1] += 1
    diffs = []
    for steps in by_task.values():
        order = sorted(steps)
        if len(order) < 2:
            continue
        first = steps[order[0]][0] / steps[order[0]][1]
        later = sum(steps[s][0] for s in order[1:]) / sum(steps[s][1] for s in order[1:])
        diffs.append(later - first)
    if not diffs:
        return {"tasks_revisited": 0}
    mean, ci = bootstrap_mean(diffs)
    return {"tasks_revisited": len(diffs), "later_minus_first": mean, "ci95": ci}


def waste(rows: list[dict]) -> dict:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["step"])].append(row)
    fail = [g for g in groups.values() if not any(r["win"] for r in g)]
    seconds = sum(r.get("seconds", 0.0) for r in rows) or 1.0
    return {"groups": len(groups), "all_fail_group_share": len(fail) / max(1, len(groups)),
            "all_fail_session_seconds_share": sum(r.get("seconds", 0.0) for g in fail for r in g) / seconds,
            "success_rate": sum(r["win"] for r in rows) / max(1, len(rows))}


def learnable_rows(rows: list[dict], total: int, label: str, recheck: float = 0.15) -> list[int]:
    """Rows with a success on their last visit or never visited, plus a deterministic `recheck` share of the rest."""
    last: dict[int, tuple[int, int]] = {}
    for row in rows:
        step, wins = last.get(row["task"], (-1, 0))
        if row["step"] > step:
            last[row["task"]] = (row["step"], int(row["win"]))
        elif row["step"] == step:
            last[row["task"]] = (step, wins + int(row["win"]))

    def recheck_row(row: int) -> bool:
        return int(hashlib.sha256(f"{label}:{row}".encode()).hexdigest(), 16) % 1000 < recheck * 1000

    return [row for row in range(total) if row not in last or last[row][1] > 0 or recheck_row(row)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", choices=("learning", "waste", "learnable"))
    parser.add_argument("agents", type=Path)
    parser.add_argument("rest", nargs="*")
    parser.add_argument("--period", action="append", default=[], help="first:last step window (one per data period)")
    parser.add_argument("--write", type=Path, help="learnable: directory for the filtered parquet and its receipt")
    args = parser.parse_args(argv)
    periods = [tuple(int(x) for x in p.split(":")) for p in args.period] or [(0, 10**9)]
    if args.report in ("learning", "waste"):
        out = {}
        for first, last in periods:
            rows = sessions(args.agents, first, last, timing=args.report == "waste")
            out[f"{first}-{last}"] = learning(rows) if args.report == "learning" else waste(rows)
        print(json.dumps(out, indent=1))
        return
    import pandas as pd

    source, label = Path(args.rest[0]), args.rest[1]
    frame = pd.read_parquet(source)
    keep = learnable_rows(sessions(args.agents, *periods[-1]), len(frame), label)
    receipt = {"source": str(source), "label": label, "source_rows": len(frame), "kept": len(keep), "rows": keep}
    if args.write:
        out = args.write / f"learnable-{label}-{len(keep)}.parquet"
        frame.iloc[keep].reset_index(drop=True).to_parquet(out)
        out.with_suffix(".json").write_text(json.dumps(receipt, indent=1) + "\n")
        print(out)
    print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}))


if __name__ == "__main__":
    main()
