"""Observe a full-state resume without signalling or modifying its workers."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from archlab.reporting.runs import jsonl_rows as read_rows


def node_sample(host, pids, checkpoint, index, key):
    program = """from pathlib import Path
import json, subprocess
pids = PIDS
checkpoint = CHECKPOINT
workers = []
for pid in pids:
    p = Path(f'/proc/{pid}')
    files = []
    if p.exists():
        for fd in (p/'fd').iterdir():
            try: target = str(fd.readlink())
            except OSError: continue
            if target.startswith(checkpoint + '/'):
                files.append(Path(target).name)
    workers.append({'pid': pid, 'alive': p.exists(), 'checkpoint_files': files})
r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
gpus = []
if r.returncode == 0:
    for line in r.stdout.splitlines():
        n, used, total, util, power = line.split(',')
        gpus.append({'index': int(n), 'used_gib': float(used)/1024, 'total_gib': float(total)/1024, 'utilization_pct': float(util), 'power_watts': float(power)})
print(json.dumps({'workers': workers, 'gpu_type': 'B300', 'gpus': gpus, 'gpu_query_error': r.stderr.strip() if r.returncode else None}))
""".replace("PIDS", repr(pids)).replace("CHECKPOINT", repr(str(checkpoint)))
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
        "/opt/venv/bin/python -",
    ]
    try:
        result = subprocess.run(command, input=program, capture_output=True, text=True, timeout=30)
        if result.returncode:
            return {"host": host, "error": result.stderr.strip()}
        data = json.loads(result.stdout)
        data["host"] = host
        for worker in data["workers"]:
            positions = [
                index["files"][name]["preceding_bytes"]
                for name in worker["checkpoint_files"]
                if name in index["files"]
            ]
            worker["observed_weight_progress_pct"] = (
                100 * max(positions) / index["total_weight_bytes"] if positions else None
            )
        return data
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        return {"host": host, "error": str(error)}


def validate_resume(rows, receipt, expected, peer_rows):
    cursor = dict(receipt["cursor"])
    resumed = [row for row in rows if row["step"] > cursor["step"]]
    peer = {row["step"]: row for row in peer_rows}
    errors = []
    for row in resumed:
        for field, want in [
            ("step", cursor["step"] + 1),
            ("phase_step", cursor["phase_step"] + 1),
            ("consumed_supervised_tokens", cursor["supervised_tokens"] + row["supervised_tokens"]),
        ]:
            if row[field] != want:
                errors.append(f"step {row['step']} {field}: {row[field]} != {want}")
        for field in [
            "loss",
            "gradient_norm_before_clip",
            "seconds",
            "learning_rate",
            "max_memory_allocated_gib",
        ]:
            if not math.isfinite(row[field]) or row[field] <= 0:
                errors.append(f"step {row['step']} invalid {field}: {row[field]}")
        if (
            row["updated_parameter_tensors"] != rows[0]["updated_parameter_tensors"]
            or row["changed_local_elements"] <= 0
        ):
            errors.append(f"step {row['step']} incomplete parameter update")
        if not all(math.isfinite(x) and x >= 0 for x in row["indexer_kl_local"]):
            errors.append(f"step {row['step']} invalid indexer KL")
        if row["step"] in peer:
            for field in [
                "input_tokens",
                "supervised_tokens",
                "consumed_supervised_tokens",
                "phase_step",
                "learning_rate",
            ]:
                if row[field] != peer[row["step"]][field]:
                    errors.append(f"step {row['step']} differs from peer data/schedule: {field}")
        cursor = {
            "step": row["step"],
            "phase_step": row["phase_step"],
            "supervised_tokens": row["consumed_supervised_tokens"],
        }
    if resumed:
        for field, want in expected.items():
            if resumed[0][field] != want:
                errors.append(f"first resumed update {field}: {resumed[0][field]} != {want}")
    return resumed, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--peer-output", type=Path, required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=45)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--healthy-updates", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.interval <= 60 or args.samples < 1 or args.healthy_updates < 1:
        raise ValueError(
            "positive sampling arguments required; interval must be at most 60 seconds"
        )
    receipt = json.loads((args.resume / "RESUME.json").read_text())
    index = json.loads((args.resume / "WEIGHT_RESTORE_INDEX.json").read_text())
    expected = json.loads((args.resume / "EXPECTED_NEXT_UPDATE.json").read_text())
    output = Path(receipt["same_output_continuation"])
    for number in range(args.samples):
        with ThreadPoolExecutor(max_workers=len(receipt["worker_pids"])) as pool:
            nodes = list(
                pool.map(
                    lambda item: node_sample(
                        item[0], item[1], receipt["resume_checkpoint"], index, args.ssh_key
                    ),
                    receipt["worker_pids"].items(),
                )
            )
        rows = read_rows(output / "train-metrics.jsonl")
        peer_rows = read_rows(args.peer_output / "train-metrics.jsonl")
        resumed, errors = validate_resume(rows, receipt, expected, peer_rows)
        failures = sorted(str(p) for p in output.glob("*failure.json"))
        workers = [worker for node in nodes for worker in node.get("workers", [])]
        live_workers = sum(worker["alive"] for worker in workers)
        observations = [
            worker["observed_weight_progress_pct"]
            for worker in workers
            if worker["observed_weight_progress_pct"] is not None
        ]
        good = (
            len(resumed) >= args.healthy_updates
            and not errors
            and not failures
            and live_workers == receipt["gpus"]
            and all(node.get("gpus") and not node.get("error") for node in nodes)
        )
        recent = resumed[1:] if len(resumed) > 1 else resumed
        seconds = sum(row["seconds"] for row in recent)
        snapshot = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "state": "healthy" if good else "checking updates" if resumed else "restoring",
            "resumed_updates": len(resumed),
            "latest": rows[-1],
            "peer_latest": peer_rows[-1],
            "validation_errors": errors,
            "failure_files": failures,
            "live_workers": live_workers,
            "nodes": nodes,
            "observed_weight_progress_pct": [min(observations), max(observations)]
            if observations
            else None,
            "steady_update_seconds_median": statistics.median(row["seconds"] for row in recent)
            if recent
            else None,
            "steady_supervised_tokens_per_second": sum(row["supervised_tokens"] for row in recent)
            / seconds
            if seconds
            else None,
            "steady_input_tokens_per_second": sum(row["input_tokens"] for row in recent) / seconds
            if seconds
            else None,
        }
        (args.resume / "LATEST_HEALTH.json").write_text(
            json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
        )
        with (args.resume / "health-samples.jsonl").open("a") as stream:
            stream.write(json.dumps(snapshot, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    k: snapshot[k]
                    for k in [
                        "utc",
                        "state",
                        "resumed_updates",
                        "live_workers",
                        "observed_weight_progress_pct",
                        "validation_errors",
                        "failure_files",
                        "steady_update_seconds_median",
                    ]
                },
                allow_nan=False,
            ),
            flush=True,
        )
        if good:
            (args.resume / "HEALTH_VERIFIED.json").write_text(
                json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
            )
            return
        if errors or failures or any(not worker["alive"] for worker in workers):
            raise RuntimeError(
                "resume health check found an actionable problem; see LATEST_HEALTH.json"
            )
        if number + 1 < args.samples:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
