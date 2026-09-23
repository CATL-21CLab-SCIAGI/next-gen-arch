"""Read-only sampling of the two GPU-exclusive full-finetuning runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from archlab.reporting.runs import jsonl_rows


def gpu_sample(host, key):
    command = [
        "ssh",
        "-i",
        str(key),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        f"root@{host}",
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if result.returncode:
        return {"host": host, "error": result.stderr.strip()}
    gpus = []
    for line in result.stdout.splitlines():
        index, used, total, utilization, power = [x.strip() for x in line.split(",")]
        gpus.append(
            {
                "index": int(index),
                "memory_used_gib": float(used) / 1024,
                "memory_total_gib": float(total) / 1024,
                "utilization_percent": float(utilization),
                "power_watts": float(power),
            }
        )
    return {"host": host, "gpu_type": "B300", "gpus": gpus}


def sample(root, key, version):
    comparison = json.loads((root / "COMPARISON.json").read_text())
    hosts = [host for v in comparison["variants"].values() for host in v["nodes"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        nodes = dict(zip(hosts, pool.map(lambda h: gpu_sample(h, key), hosts), strict=False))
    snapshot = {"utc": datetime.now(timezone.utc).isoformat(), "variants": {}}
    for variant, info in comparison["variants"].items():
        output = root / f"production-{variant}-{version}"
        path = output / "train-metrics.jsonl"
        metrics = jsonl_rows(path, missing_ok=True)
        failures = sorted(p.name for p in output.glob("*failure.json"))
        checkpoint_paths = sorted(output.glob("checkpoints/*/COMPLETE.json"))
        snapshot["variants"][variant] = {
            "nodes": [nodes[h] for h in info["nodes"]],
            "updates": len(metrics),
            "latest": metrics[-1] if metrics else None,
            "failures": failures,
            "complete_checkpoints": [str(p.parent) for p in checkpoint_paths],
        }
    with (root / "gpu-health-samples.jsonl").open("a") as stream:
        stream.write(json.dumps(snapshot, allow_nan=False) + "\n")
    summary = {"utc": snapshot["utc"]}
    for variant, info in snapshot["variants"].items():
        gpus = [g for node in info["nodes"] for g in node.get("gpus", [])]
        summary[variant] = {
            "updates": info["updates"],
            "loss": None if info["latest"] is None else info["latest"]["loss"],
            "max_used_gib": max((g["memory_used_gib"] for g in gpus), default=None),
            "mean_gpu_utilization": sum(g["utilization_percent"] for g in gpus) / len(gpus)
            if gpus
            else None,
            "failures": len(info["failures"]),
            "complete_checkpoints": len(info["complete_checkpoints"]),
        }
    print(json.dumps(summary), flush=True)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--version", default="v2")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    if not 0 < args.interval <= 60 or args.samples < 1:
        raise ValueError("positive sample count and interval of at most60 seconds are required")
    for i in range(args.samples):
        sample(args.root, args.ssh_key, args.version)
        if i + 1 < args.samples:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
