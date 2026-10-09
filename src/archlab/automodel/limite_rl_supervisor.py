"""Supervise only the Limite child; protect concurrent scaling throughput."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


def recent_seconds(path):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 256000))
        lines = stream.read().splitlines()[1:]
    values = []
    for line in lines:
        try:
            row = json.loads(line)
            values.append(float(row["wall_seconds"]))
        except (ValueError, KeyError):
            continue
    if len(values) < 8:
        raise ValueError("insufficient baseline metrics")
    return statistics.median(values[-8:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scaling-metrics", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    baseline = recent_seconds(args.scaling_metrics)
    command = args.command[1:] if args.command[0] == "--" else args.command
    with (args.run_root / "train.log").open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    (args.run_root / "PROCESS.json").write_text(
        json.dumps(dict(pid=child.pid, command=command, baseline_seconds=baseline)) + "\n"
    )
    slow_since = None
    while child.poll() is None:
        current = recent_seconds(args.scaling_metrics)
        ratio = current / baseline
        now = time.time()
        slow_since = (slow_since or now) if ratio > 1.3 else None
        report = dict(
            time=now,
            baseline_seconds=baseline,
            current_seconds=current,
            slowdown_ratio=ratio,
            pid=child.pid,
        )
        (args.run_root / "GUARD.json").write_text(json.dumps(report) + "\n")
        if slow_since is not None and now - slow_since > 180:
            (args.run_root / "STOP_REQUEST").write_text(
                "Scaling slowdown >30% sustained for three minutes\n"
            )
        time.sleep(15)
    (args.run_root / "EXIT.json").write_text(
        json.dumps(dict(returncode=child.returncode, time=time.time())) + "\n"
    )


if __name__ == "__main__":
    main()
