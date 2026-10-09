"""Bounded node-local evaluation supervisor; it can signal only its own children."""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import time
import traceback
from pathlib import Path

from archlab.artifacts import atomic_write_json


def tail_metrics(path, count=64):
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        length = stream.tell()
        stream.seek(max(0, length - 512 * 1024))
        payload = stream.read().splitlines()
    rows = []
    for line in payload:
        try:
            row = json.loads(line)
            if "seconds" in row and "step" in row:
                rows.append(row)
        except (ValueError, UnicodeError):
            continue
    if not rows:
        raise ValueError("training has no completed metrics")
    rows = rows[-count:]
    return dict(
        step=rows[-1]["step"],
        median_seconds=statistics.median(r["seconds"] for r in rows),
        tokens=sum(r["supervised_tokens"] for r in rows),
        update_seconds=sum(r["seconds"] for r in rows),
        samples=len(rows),
        mtime=Path(path).stat().st_mtime,
    )


def guard_reason(plan, snapshot, free_gpu_gib, available_host_gib, now):
    if free_gpu_gib < 55:
        return "GPU headroom below 55 GiB"
    if available_host_gib < 32:
        return "host headroom below 32 GiB"
    if any(now - row["mtime"] > 300 for row in snapshot.values()):
        return "trainer metrics stale for more than five minutes"
    for root in plan["training_roots"].values():
        if (Path(root) / "TRAINING_COMPLETE.json").exists():
            return "yield evaluation before the training queue advances"
    return None


def stop_child(process):
    """The Popen child is a new session; never look up or signal trainer PIDs."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def supervise(plan_path, node, tiny=False):
    import psutil
    import pynvml

    plan = json.loads(plan_path.read_text())
    output = Path(plan["output"])
    prefix = "tiny" if tiny else "queue"
    if (output / "STOP").exists() or (output / f"{prefix}-node-{node}-COMPLETE.json").exists():
        raise ValueError("queue output already stopped or complete")
    if not tiny and any(
        not (output / f"tiny-{p['name']}" / "COMPLETE.json").exists() for p in plan["phases"][:2]
    ):
        raise ValueError("both variants require completed tiny distributed qualification")
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(8)]
    metrics_paths = {
        arm: Path(root) / "train-metrics.jsonl" for arm, root in plan["training_roots"].items()
    }
    baseline = {arm: tail_metrics(path) for arm, path in metrics_paths.items()}
    atomic_write_json(output / f"{prefix}-node-{node}-BASELINE.json", baseline)
    began = time.monotonic()
    process = None
    try:
        for index, phase in enumerate(plan["phases"][:2] if tiny else plan["phases"]):
            if index:
                while not all(
                    (output / f"{prefix}-node-{peer}-phase-{index - 1}-DONE.json").exists()
                    for peer in (0, 1)
                ):
                    if (output / "STOP").exists() or time.monotonic() - began > plan[
                        "maximum_queue_hours"
                    ] * 3600:
                        raise RuntimeError("peer phase did not complete or queue stopped")
                    time.sleep(5)
            env = os.environ.copy()
            env.update(plan["environment"])
            env["PYTHONPATH"] = phase["model_source"] + "/src:" + plan["upstream_source"]
            command = [
                "/opt/venv/bin/python",
                "-m",
                "torch.distributed.run",
                "--nnodes=2",
                "--nproc-per-node=8",
                f"--node-rank={node}",
                f"--master-addr={plan['master_addr']}",
                f"--master-port={plan['master_port'] + index + (10 if tiny else 0)}",
                "--max-restarts=0",
                plan["driver"],
                "--plan",
                str(plan_path),
                "--phase",
                str(index),
            ]
            if tiny:
                command.append("--tiny")
            with (output / f"{prefix}-node-{node}-phase-{index}.log").open("x") as log:
                process = subprocess.Popen(
                    command,
                    env=env,
                    cwd=phase["model_source"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                atomic_write_json(
                    output / f"{prefix}-node-{node}-ACTIVE.json",
                    dict(pid=process.pid, phase=phase["name"], command=command, unix=time.time()),
                )
                phase_start = time.monotonic()
                severe = 0
                while process.poll() is None:
                    snapshot = {arm: tail_metrics(path) for arm, path in metrics_paths.items()}
                    free = min(pynvml.nvmlDeviceGetMemoryInfo(h).free / 2**30 for h in handles)
                    host_free = psutil.virtual_memory().available / 2**30
                    reason = guard_reason(plan, snapshot, free, host_free, time.time())
                    ratios = {
                        arm: row["median_seconds"] / baseline[arm]["median_seconds"]
                        for arm, row in snapshot.items()
                    }
                    # Rest only evaluation; trainers continue independently.
                    if node == 0:
                        atomic_write_json(
                            output / "PACING.json",
                            dict(
                                rest_factor=19
                                if max(ratios.values()) > plan["maximum_training_slowdown"]
                                else 9
                            ),
                        )
                    severe = severe + 1 if max(ratios.values()) > 1.75 else 0
                    if severe >= 6:
                        reason = "training slowdown exceeded 75% for three minutes"
                    if (
                        time.monotonic() - phase_start > plan["maximum_phase_hours"] * 3600
                        or time.monotonic() - began > plan["maximum_queue_hours"] * 3600
                    ):
                        reason = "bounded evaluation deadline"
                    if (output / "STOP").exists():
                        reason = "evaluation stop requested"
                    status = dict(
                        unix=time.time(),
                        phase=phase["name"],
                        training=snapshot,
                        slowdown_ratios=ratios,
                        minimum_gpu_free_gib=free,
                        host_available_gib=host_free,
                        reason=reason,
                    )
                    atomic_write_json(output / f"{prefix}-node-{node}-HEALTH.json", status)
                    if reason:
                        raise RuntimeError(reason)
                    time.sleep(30)
                if process.returncode:
                    raise RuntimeError(f"evaluation child failed with exit {process.returncode}")
            marker = output / ("tiny-" + phase["name"] if tiny else phase["name"]) / "COMPLETE.json"
            if not marker.exists():
                raise RuntimeError("evaluation exited without its completion receipt")
            atomic_write_json(
                output / f"{prefix}-node-{node}-phase-{index}-DONE.json",
                dict(unix=time.time(), marker=str(marker)),
            )
        atomic_write_json(output / f"{prefix}-node-{node}-COMPLETE.json", dict(unix=time.time()))
    except BaseException:
        (output / "STOP").touch()
        atomic_write_json(
            output / f"{prefix}-node-{node}-FAILED.json", dict(traceback=traceback.format_exc())
        )
        raise
    finally:
        if process is not None:
            stop_child(process)
        pynvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--node", type=int, choices=(0, 1), required=True)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    supervise(args.plan, args.node, args.tiny)


if __name__ == "__main__":
    main()
