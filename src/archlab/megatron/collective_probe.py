"""Fail-fast NCCL and topology probe for a multi-node Megatron launch."""

from __future__ import annotations

import json
import os
import socket
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path


def validate_topology(
    hostnames: list[str], *, world_size: int, expected_nodes: int, gpus_per_node: int
) -> dict[str, int]:
    counts = dict(sorted(Counter(hostnames).items()))
    if len(hostnames) != world_size:
        raise RuntimeError(f"gathered {len(hostnames)} ranks, expected {world_size}")
    if len(counts) != expected_nodes:
        raise RuntimeError(f"observed {len(counts)} hosts, expected {expected_nodes}: {counts}")
    invalid = {host: count for host, count in counts.items() if count != gpus_per_node}
    if invalid:
        raise RuntimeError(f"unexpected ranks per host: {invalid}")
    return counts


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    expected_nodes = int(os.environ["NGA_EXPECTED_NODES"])
    gpus_per_node = int(os.environ["NGA_GPUS_PER_NODE"])
    visible_gpus = torch.cuda.device_count()
    if visible_gpus != gpus_per_node:
        raise RuntimeError(f"visible GPUs={visible_gpus}, expected {gpus_per_node}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        expected_world = expected_nodes * gpus_per_node
        if world_size != expected_world:
            raise RuntimeError(f"world size={world_size}, expected {expected_world}")

        value = torch.tensor(float(rank + 1), dtype=torch.float64, device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size + 1) / 2
        if value.item() != expected_sum:
            raise RuntimeError(f"NCCL all-reduce={value.item()}, expected {expected_sum}")

        hostnames: list[str | None] = [None] * world_size
        gpu_names: list[str | None] = [None] * world_size
        dist.all_gather_object(hostnames, socket.gethostname())
        dist.all_gather_object(gpu_names, torch.cuda.get_device_name(local_rank))
        topology = validate_topology(
            [str(host) for host in hostnames],
            world_size=world_size,
            expected_nodes=expected_nodes,
            gpus_per_node=gpus_per_node,
        )
        if rank == 0:
            _write_json(
                Path(os.environ["NGA_OUTPUT_ROOT"]) / "COLLECTIVE_VALIDATED.json",
                {
                    "backend": "nccl",
                    "world_size": world_size,
                    "nodes": topology,
                    "gpus_per_node": gpus_per_node,
                    "gpu_names": sorted(set(str(name) for name in gpu_names)),
                    "all_reduce_sum": value.item(),
                    "validated_at_unix": time.time(),
                },
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
