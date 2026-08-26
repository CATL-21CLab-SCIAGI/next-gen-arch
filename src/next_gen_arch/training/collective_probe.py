"""Fail-fast NCCL topology probe for a frozen multi-node launch."""

from __future__ import annotations

import argparse
import json
import os
import socket
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist


def _encode_text(value: str, *, width: int, device: torch.device) -> torch.Tensor:
    payload = value.encode("utf-8")[:width]
    result = torch.zeros(width, dtype=torch.uint8, device=device)
    if payload:
        result[: len(payload)] = torch.tensor(tuple(payload), dtype=torch.uint8, device=device)
    return result


def _decode_text(value: torch.Tensor) -> str:
    payload = bytes(value.cpu().tolist())
    return payload.split(b"\0", 1)[0].decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-local-world-size", type=int, required=True)
    parser.add_argument("--expected-hosts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size != args.expected_world_size:
        raise RuntimeError(f"world-size mismatch: {world_size} != {args.expected_world_size}")
    if local_world_size != args.expected_local_world_size:
        raise RuntimeError(
            f"local-world-size mismatch: {local_world_size} != {args.expected_local_world_size}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    try:
        rank_sum = torch.tensor(float(rank), dtype=torch.float64, device=device)
        dist.all_reduce(rank_sum, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size - 1) / 2
        if rank_sum.item() != expected_sum:
            raise RuntimeError(f"all-reduce mismatch: {rank_sum.item()} != {expected_sum}")

        hostname = socket.gethostname()
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        host_tensor = _encode_text(hostname, width=128, device=device)
        device_tensor = _encode_text(visible_devices, width=128, device=device)
        all_hosts = [torch.empty_like(host_tensor) for _ in range(world_size)]
        all_devices = [torch.empty_like(device_tensor) for _ in range(world_size)]
        dist.all_gather(all_hosts, host_tensor)
        dist.all_gather(all_devices, device_tensor)
        hosts = [_decode_text(value) for value in all_hosts]
        device_allowlists = [_decode_text(value) for value in all_devices]
        host_counts = Counter(hosts)
        if len(host_counts) != args.expected_hosts:
            raise RuntimeError(
                f"host-count mismatch: {dict(host_counts)}; expected {args.expected_hosts} hosts"
            )
        if set(host_counts.values()) != {args.expected_local_world_size}:
            raise RuntimeError(
                "rank distribution mismatch: "
                f"{dict(host_counts)}; expected {args.expected_local_world_size} ranks per host"
            )

        if rank == 0:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "backend": dist.get_backend(),
                        "status": "pass",
                        "world_size": world_size,
                        "local_world_size": local_world_size,
                        "hosts": dict(sorted(host_counts.items())),
                        "cuda_visible_devices": sorted(set(device_allowlists)),
                        "all_reduce_rank_sum": rank_sum.item(),
                        "expected_rank_sum": expected_sum,
                        "gpu_name": torch.cuda.get_device_name(local_rank),
                        "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
                        "nccl_ib_hca": os.environ.get("NCCL_IB_HCA"),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        print(
            f"rank={rank} local_rank={local_rank} host={hostname} "
            f"gpu={torch.cuda.get_device_name(local_rank)} all_reduce=pass",
            flush=True,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
