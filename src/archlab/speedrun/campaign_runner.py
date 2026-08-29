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

from archlab.optimizers.recipes import OPTIMIZATION_RECIPES
from archlab.speedrun.campaigns import TEN_M_SEEDS, TEN_M_VARIANTS


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


def _task_queues(
    mode: str,
    node_index: int,
    num_nodes: int,
    queue_count: int,
    partition_strategy: str,
) -> list[list[Task]]:
    """Partition tasks without ever assigning work outside explicit GPU queues."""
    if partition_strategy == "seed":
        node_tasks = [
            task
            for index, task in enumerate(_campaign_tasks(mode))
            if index % num_nodes == node_index
        ]
        return [node_tasks[index::queue_count] for index in range(queue_count)]

    seeds = (TEN_M_SEEDS[0],) if mode == "probe" else TEN_M_SEEDS
    node_variants = [
        variant for index, variant in enumerate(TEN_M_VARIANTS) if index % num_nodes == node_index
    ]
    queues: list[list[Task]] = [[] for _index in range(queue_count)]
    for index, variant in enumerate(node_variants):
        queues[index % queue_count].extend(Task(variant.name, seed) for seed in seeds)
    return queues


def _is_matching_complete_run(
    run_dir: Path,
    task: Task,
    mode: str,
    backend_profile: str,
    optimization_recipe: str,
) -> bool:
    """Reuse only a fully materialized result with the exact requested controls."""
    marker = run_dir / "COMPLETE.json"
    result_path = run_dir / "result.json"
    if not marker.is_file() and not result_path.is_file():
        return False
    if not marker.is_file() or not result_path.is_file():
        raise RuntimeError(f"incomplete existing run directory: {run_dir}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    variant = payload.get("variant", "")
    variant_name = variant.get("name", "") if isinstance(variant, dict) else variant
    profile = payload.get("backend_profile", "")
    profile_name = profile.get("name", "") if isinstance(profile, dict) else profile
    recipe = payload.get("optimization_recipe", "")
    recipe_name = recipe.get("name", "") if isinstance(recipe, dict) else recipe
    actual = (
        payload.get("status"),
        variant_name,
        payload.get("seed"),
        payload.get("mode"),
        profile_name,
        recipe_name,
    )
    expected = (
        "complete",
        task.variant,
        task.seed,
        mode,
        backend_profile,
        optimization_recipe,
    )
    if actual != expected:
        raise RuntimeError(
            f"existing result contract mismatch in {run_dir}: {actual} != {expected}"
        )
    return True


def _cache_directory(
    output_root: Path,
    external_cache_root: Path | None,
    node_index: int,
    task: Task,
    partition_strategy: str,
) -> Path:
    if external_cache_root is not None:
        # The original campaign used one task-name cache shared through NAS.
        # Keeping this explicit avoids silently borrowing artifacts from an
        # unrelated campaign while enabling a deliberate warm-cache pass.
        return external_cache_root / task.name
    cache_key = task.variant if partition_strategy == "variant" else task.name
    return output_root / "cache" / f"node-{node_index}" / cache_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-index", required=True, type=int)
    parser.add_argument("--num-nodes", required=True, type=int)
    parser.add_argument("--gpus", required=True, type=_parse_gpus)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="explicit existing task-name cache root for a provenance-labeled warm pass",
    )
    parser.add_argument("--mode", choices=("probe", "full"), required=True)
    parser.add_argument(
        "--partition-strategy",
        choices=("seed", "variant"),
        default="seed",
        help="seed preserves simultaneous pairing; variant reuses compiler caches across seeds",
    )
    parser.add_argument("--probe-steps", type=int, default=1)
    parser.add_argument("--backend-profile", default="legacy")
    parser.add_argument(
        "--optimization-recipe", choices=tuple(OPTIMIZATION_RECIPES), default="baseline"
    )
    args = parser.parse_args()
    if args.num_nodes < 1 or not 0 <= args.node_index < args.num_nodes:
        parser.error("node index must be inside [0, num-nodes)")
    if args.probe_steps < 1:
        parser.error("--probe-steps must be positive")
    if not os.environ.get("NANOCHAT_BASE_DIR"):
        parser.error("NANOCHAT_BASE_DIR must point at the frozen campaign data root")

    output_root = args.output_root.expanduser().resolve()
    external_cache_root = (
        args.cache_root.expanduser().resolve() if args.cache_root is not None else None
    )
    runs_root = output_root / "runs"
    logs_root = output_root / "logs"
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    uuid_by_index = _gpu_uuid_map()
    for gpu in args.gpus:
        _require_idle(gpu, uuid_by_index)

    queues = _task_queues(
        args.mode,
        args.node_index,
        args.num_nodes,
        len(args.gpus),
        args.partition_strategy,
    )
    node_tasks = [task for queue in queues for task in queue]
    state_path = output_root / f"node-{args.node_index}-{args.mode}.json"
    lock = threading.Lock()
    state = {
        "mode": args.mode,
        "node_index": args.node_index,
        "num_nodes": args.num_nodes,
        "gpu_allowlist": list(args.gpus),
        "partition_strategy": args.partition_strategy,
        "backend_profile": args.backend_profile,
        "optimization_recipe": args.optimization_recipe,
        "external_cache_root": str(external_cache_root) if external_cache_root else None,
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
            run_dir = runs_root / task.name
            try:
                if _is_matching_complete_run(
                    run_dir,
                    task,
                    args.mode,
                    args.backend_profile,
                    args.optimization_recipe,
                ):
                    update(
                        task,
                        status="complete",
                        reused_complete_result=True,
                        finished_at_unix=time.time(),
                    )
                    continue
            except BaseException as error:
                update(task, status="blocked", error=str(error), finished_at_unix=time.time())
                continue
            try:
                _require_idle(gpu, uuid_by_index)
            except BaseException as error:
                update(task, status="blocked", error=str(error), finished_at_unix=time.time())
                continue
            log_path = logs_root / f"{task.name}.log"
            command = [
                str(torchrun),
                "--standalone",
                "--nproc-per-node=1",
                "-m",
                "archlab.megatron.train",
                "--variant",
                task.variant,
                "--seed",
                str(task.seed),
                "--run-dir",
                str(run_dir),
                "--backend-profile",
                args.backend_profile,
                "--optimization-recipe",
                args.optimization_recipe,
            ]
            if args.mode == "probe":
                command.extend(("--probe-steps", str(args.probe_steps)))
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment.setdefault("NANOCHAT_ATTENTION_BACKEND", "sdpa")
            cache_root = _cache_directory(
                output_root,
                external_cache_root,
                args.node_index,
                task,
                args.partition_strategy,
            )
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
            failure = None
            failed_path = run_dir / "FAILED.json"
            if completed.returncode != 0 and failed_path.is_file():
                failed_payload = json.loads(failed_path.read_text(encoding="utf-8"))
                failure = failed_payload.get("failure")
            update(
                task,
                status="complete" if completed.returncode == 0 else "failed",
                returncode=completed.returncode,
                log=str(log_path),
                failure=failure,
                retry_recommended=bool(isinstance(failure, dict) and failure.get("retriable")),
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
