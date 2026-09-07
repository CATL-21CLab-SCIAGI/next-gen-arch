"""Adapter-only distributed state using the pinned AutoModel PEFT optimizer path."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp


def checkpoint_payload(adapters, optimizer):
    from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState

    # Reuse the upstream PEFT+EP path, including lazy Adam-state materialization.
    # A fresh optimizer otherwise has an empty DCP load skeleton and can silently
    # omit saved moments. These additions are PEFT, although they are not LoRA.
    optimizer_state = OptimizerState(torch.nn.ModuleDict(adapters), optimizer,
                                     is_peft=True, has_expert_parallelism=True).state_dict()["optim"]
    return {"adapters": {name: module.state_dict() for name, module in adapters.items()},
            "optimizer": optimizer_state,
            f"rng_rank_{dist.get_rank()}": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}}


def assert_state_equal(actual, expected):
    """Exact optimizer/RNG restoration, independent of kernel nondeterminism."""
    if isinstance(expected, torch.Tensor):
        actual = actual.to_local() if hasattr(actual, "to_local") else actual
        expected = expected.to_local() if hasattr(expected, "to_local") else expected
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_state_equal(a, b)
    else:
        assert actual == expected


def state_digest(value) -> str:
    """Hash nested state exactly; DTensors contribute their local shard bytes.

    Tensor leaves may have arbitrary shape/dtype and are never modified.
    At most one tensor's local bytes are staged on CPU at a time.
    """
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            local = item.to_local() if hasattr(item, "to_local") else item
            digest.update(str((tuple(local.shape), str(local.dtype))).encode())
            digest.update(local.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Atomically publish task metadata after flushing its temporary file."""
    with tempfile.NamedTemporaryFile(mode="w", prefix=path.name + ".", suffix=".tmp",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_training_checkpoint(path: Path, *, adapters: dict, optimizer, scheduler,
                             contract: dict, cursor: int) -> None:
    """Save only added modules, Adam, scheduler and rank RNG; publish completion last."""
    if dist.get_rank() == 0:
        path.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    payload = checkpoint_payload(adapters, optimizer)
    payload["scheduler"] = scheduler.state_dict()
    fingerprints = [None] * dist.get_world_size()
    dist.all_gather_object(fingerprints, state_digest(payload))
    dcp.save(payload, checkpoint_id=path / "state")
    if dist.get_rank() == 0:
        write_json(path / "COMPLETE.json", {"format": "archlab-simplicial-training-v1",
                   "contract": contract, "cursor": cursor, "completed_steps": cursor,
                   "rank_state_sha256": fingerprints})
    dist.barrier()


def read_training_checkpoint(path: Path, contract: dict) -> dict:
    """Reject incomplete, diagnostic or incompatible checkpoints before loading tensors."""
    metadata = json.loads((path / "COMPLETE.json").read_text())
    if metadata.get("format") != "archlab-simplicial-training-v1" or metadata.get("contract") != contract:
        raise ValueError("checkpoint format or experiment/runtime/data contract mismatch")
    cursor = metadata.get("cursor")
    if type(cursor) is not int or cursor < 0 or metadata.get("completed_steps") != cursor:
        raise ValueError("invalid checkpoint data cursor/step")
    if cursor > contract["total_steps"]:
        raise ValueError("checkpoint cursor exceeds the one-pass budget")
    if len(metadata.get("rank_state_sha256", [])) != contract["training"]["world_size"]:
        raise ValueError("checkpoint rank-state coverage mismatch")
    return metadata


def load_training_checkpoint(path: Path, *, adapters: dict, optimizer, scheduler, contract: dict) -> int:
    """Restore a fresh process and verify exact local tensor/optimizer/scheduler/RNG state."""
    metadata = read_training_checkpoint(path, contract)
    payload = checkpoint_payload(adapters, optimizer)
    payload["scheduler"] = scheduler.state_dict()
    dcp.load(payload, checkpoint_id=path / "state")
    expected = metadata["rank_state_sha256"][dist.get_rank()]
    if state_digest(payload) != expected:
        raise RuntimeError("loaded checkpoint state digest mismatch")
    for name, adapter in adapters.items():
        adapter.load_state_dict(payload["adapters"][name], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    rng = payload[f"rng_rank_{dist.get_rank()}"]
    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state(rng["cuda"])
    actual = checkpoint_payload(adapters, optimizer)
    actual["scheduler"] = scheduler.state_dict()
    if state_digest(actual) != expected or scheduler.num_steps != metadata["cursor"]:
        raise RuntimeError("restored live state/cursor differs from checkpoint")
    return metadata["cursor"]
