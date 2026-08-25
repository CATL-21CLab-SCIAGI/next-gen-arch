"""Run one node's safe GPU queues for the frozen 10M Megatron campaign."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from next_gen_arch.training.campaigns import TEN_M_SEEDS, TEN_M_VARIANTS


@dataclass(frozen=True)
class Task:
    variant: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.variant}-seed{self.seed}"


def _parse_gpus(value: str) -> tuple[int, ...]:
    try:
        gpus = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPUs must be comma-separated integers") from error
    if not gpus or len(gpus) != len(set(gpus)) or min(gpus) < 0:
        raise argparse.ArgumentTypeError("GPU allowlist must be unique and non-negative")
    return gpus


def _gpu_uuid_map() -> dict[int, str]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    rows = csv.reader(output.splitlines(), skipinitialspace=True)
    return {int(index): uuid.strip() for index, uuid in rows}


def _busy_gpu_uuids() -> set[str]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {row[0].strip() for row in csv.reader(output.splitlines(), skipinitialspace=True) if row}


def _require_idle(gpu: int, uuid_by_index: dict[int, str]) -> None:
    if gpu not in uuid_by_index:
        raise RuntimeError(f"GPU {gpu} does not exist on this node")
    if uuid_by_index[gpu] in _busy_gpu_uuids():
        raise RuntimeError(f"refusing to launch because allowlisted GPU {gpu} is busy")


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _campaign_tasks(mode: str) -> list[Task]:
    seeds = (TEN_M_SEEDS[0],) if mode == "probe" else TEN_M_SEEDS
    return [Task(variant.name, seed) for variant in TEN_M_VARIANTS for seed in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-index", required=True, type=int)
    parser.add_argument("--num-nodes", required=True, type=int)
    parser.add_argument("--gpus", required=True, type=_parse_gpus)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("probe", "full"), required=True)
    parser.add_argument("--probe-steps", type=int, default=1)
    args = parser.parse_args()
    if args.num_nodes < 1 or not 0 <= args.node_index < args.num_nodes:
        parser.error("node index must be inside [0, num-nodes)")
    if args.probe_steps < 1:
        parser.error("--probe-steps must be positive")
    if not os.environ.get("NANOCHAT_BASE_DIR"):
        parser.error("NANOCHAT_BASE_DIR must point at the frozen campaign data root")

    output_root = args.output_root.expanduser().resolve()
    runs_root = output_root / "runs"
    logs_root = output_root / "logs"
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    uuid_by_index = _gpu_uuid_map()
    for gpu in args.gpus:
        _require_idle(gpu, uuid_by_index)

    all_tasks = _campaign_tasks(args.mode)
    node_tasks = [
        task for index, task in enumerate(all_tasks) if index % args.num_nodes == args.node_index
    ]
    queues = [node_tasks[index :: len(args.gpus)] for index in range(len(args.gpus))]
    state_path = output_root / f"node-{args.node_index}-{args.mode}.json"
    lock = threading.Lock()
    state = {
        "mode": args.mode,
        "node_index": args.node_index,
        "num_nodes": args.num_nodes,
        "gpu_allowlist": list(args.gpus),
        "started_at_unix": time.time(),
        "tasks": {task.name: {"status": "queued"} for task in node_tasks},
    }
    _atomic_write_json(state_path, state)

    def update(task: Task, **values) -> None:
        with lock:
            state["tasks"][task.name].update(values)
            _atomic_write_json(state_path, state)

    torchrun = Path(sys.executable).parent / "torchrun"

    def worker(gpu: int, tasks: list[Task]) -> None:
        for task in tasks:
            try:
                _require_idle(gpu, uuid_by_index)
            except BaseException as error:
                update(task, status="blocked", error=str(error), finished_at_unix=time.time())
                continue
            run_dir = runs_root / task.name
            log_path = logs_root / f"{task.name}.log"
            command = [
                str(torchrun),
                "--standalone",
                "--nproc-per-node=1",
                "-m",
                "next_gen_arch.training.megatron_train",
                "--variant",
                task.variant,
                "--seed",
                str(task.seed),
                "--run-dir",
                str(run_dir),
            ]
            if args.mode == "probe":
                command.extend(("--probe-steps", str(args.probe_steps)))
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment.setdefault("NANOCHAT_ATTENTION_BACKEND", "sdpa")
            cache_root = output_root / "cache" / task.name
            for variable, directory in (
                ("CUDA_CACHE_PATH", "cuda"),
                ("TRITON_CACHE_DIR", "triton"),
                ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
            ):
                cache_path = cache_root / directory
                cache_path.mkdir(parents=True, exist_ok=True)
                environment[variable] = str(cache_path)
            ptxas_path = environment.get("NGA_PTXAS_PATH") or environment.get("TRITON_PTXAS_PATH")
            if ptxas_path:
                environment["TRITON_PTXAS_PATH"] = ptxas_path
                environment["PTXAS_CUDA_PATH"] = ptxas_path
            update(
                task,
                status="running",
                gpu=gpu,
                command=command,
                triton_ptxas_path=environment.get("TRITON_PTXAS_PATH"),
                started_at_unix=time.time(),
            )
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    env=environment,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            update(
                task,
                status="complete" if completed.returncode == 0 else "failed",
                returncode=completed.returncode,
                log=str(log_path),
                finished_at_unix=time.time(),
            )

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(worker, gpu, queue)
            for gpu, queue in zip(args.gpus, queues, strict=True)
        ]
        for future in futures:
            future.result()
    state["finished_at_unix"] = time.time()
    state["status"] = (
        "complete"
        if all(item["status"] == "complete" for item in state["tasks"].values())
        else "failed"
    )
    _atomic_write_json(state_path, state)
    if state["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
