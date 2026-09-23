# ruff: noqa: I001  # Preserve the checkpoint-qualified import order.
"""Same-mesh full-state checkpoints with bounded CPU serialization buffers.

Weights and optimizer state stay on GPU during training. Only serialization
stages CPU chunks. COMPLETE is published after every rank has written its
manifest; interrupted writes remain visibly incomplete. Sharding is explicit.
"""

from pathlib import Path
import hashlib
import json
import os  # noqa: F401 — preserve checkpoint-qualified executable AST
import torch
import torch.distributed as dist
from archlab.artifacts import atomic_write_json
from archlab.optimizers.sharded_adafactor import local_tensor


def _checksum(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def save_full_checkpoint(path, model, optimizer, cursor, contract):
    path = Path(path)
    rank, world = dist.get_rank(), dist.get_world_size()
    rank_path = path / f"rank-{rank:02d}"
    rank_path.mkdir(parents=True, exist_ok=False)
    manifest = {
        "rank": rank,
        "world_size": world,
        "tensors": [],
        "cursor": cursor,
        "contract": contract,
    }
    named = list(model.named_parameters()) + list(model.named_buffers())
    for number, (name, tensor) in enumerate(named):
        local = local_tensor(tensor.detach())
        flat = local.reshape(-1)
        entry = {
            "name": name,
            "shape": list(local.shape),
            "global_shape": list(tensor.shape),
            "dtype": str(local.dtype),
            "chunks": [],
        }
        # At most 256 MiB staged for any payload, including a huge Engram table.
        count = max(1, 256 * 1024 * 1024 // local.element_size())
        for part, first in enumerate(range(0, flat.numel(), count)):
            host = flat[first : first + count].cpu().clone()
            filename = f"tensor-{number:04d}-{part:03d}.pt"
            torch.save(host, rank_path / filename)
            entry["chunks"].append(
                {"file": filename, "elements": host.numel(), "sha256": _checksum(host)}
            )
            del host
        manifest["tensors"].append(entry)
    states = []
    for number, p in enumerate(optimizer.param_groups[0]["params"]):
        state = optimizer.state[p]
        payload = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()
        }
        filename = f"optimizer-{number:04d}.pt"
        torch.save(payload, rank_path / filename)
        states.append(filename)
    manifest["optimizer_states"] = states
    manifest["optimizer_options"] = [
        {k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups
    ]
    torch.save(
        {"cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()},
        rank_path / "rng.pt",
    )
    atomic_write_json(rank_path / "MANIFEST.json", manifest)
    dist.barrier()
    if rank == 0:
        atomic_write_json(
            path / "COMPLETE.json",
            {
                "format": "archlab-v41-full-sharded-v1",
                "world_size": world,
                "cursor": cursor,
                "contract": contract,
                "manifests": [f"rank-{r:02d}/MANIFEST.json" for r in range(world)],
            },
        )
    dist.barrier()


@torch.no_grad()
def restore_full_checkpoint(path, model, optimizer, contract):
    path = Path(path)
    marker = json.loads((path / "COMPLETE.json").read_text())
    if (
        marker["format"] != "archlab-v41-full-sharded-v1"
        or marker["world_size"] != dist.get_world_size()
        or marker["contract"] != contract
    ):
        raise ValueError("full checkpoint requires the same immutable contract and mesh")
    rank_path = path / f"rank-{dist.get_rank():02d}"
    manifest = json.loads((rank_path / "MANIFEST.json").read_text())
    named = list(model.named_parameters()) + list(model.named_buffers())
    for (name, tensor), entry in zip(named, manifest["tensors"], strict=True):
        local = local_tensor(tensor)
        if (
            name != entry["name"]
            or list(local.shape) != entry["shape"]
            or str(local.dtype) != entry["dtype"]
        ):
            raise ValueError(f"full checkpoint tensor contract differs: {name}")
        flat, first = local.reshape(-1), 0
        for chunk in entry["chunks"]:
            host = torch.load(rank_path / chunk["file"], map_location="cpu", weights_only=True)
            if _checksum(host) != chunk["sha256"] or host.numel() != chunk["elements"]:
                raise ValueError(f"checkpoint checksum failed: {name}")
            flat[first : first + host.numel()].copy_(host)
            first += host.numel()
        if first != flat.numel():
            raise ValueError(f"incomplete tensor: {name}")
    for p, filename in zip(
        optimizer.param_groups[0]["params"], manifest["optimizer_states"], strict=True
    ):
        saved = torch.load(rank_path / filename, map_location="cpu", weights_only=True)
        optimizer.state[p] = {
            k: v.to(p.device) if isinstance(v, torch.Tensor) else v for k, v in saved.items()
        }
    for group, options in zip(optimizer.param_groups, manifest["optimizer_options"], strict=True):
        group.update(options)
    rng = torch.load(rank_path / "rng.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(rng["cpu_rng"])
    torch.cuda.set_rng_state(rng["cuda_rng"])
    dist.barrier()
    return marker["cursor"]
