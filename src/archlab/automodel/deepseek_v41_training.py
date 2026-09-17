"""Shared training operations for the native V4.1 adapter pilot.

Data ranks are unique; EP is an overlay, not extra copies in token accounting.
Adapter parameters are complete replicas, with global gradient reduction before
headwise Muon. Frozen base weights never enter the optimizers or checkpoints.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.automodel.deepseek_v41_autograd import training_hidden
from archlab.automodel.deepseek_v41_loss import frozen_head_loss


def emit(event, **fields):
    record = {"event": event, "rank": dist.get_rank(), "unix_time": time.time(), **fields}
    print(json.dumps(record, allow_nan=False), flush=True)
    return record


def global_numbers(*values):
    numbers = torch.tensor(values, device="cuda", dtype=torch.float64)
    dist.all_reduce(numbers)
    return numbers.tolist()


def globally_reduce_gradients(parameters, targets):
    if targets <= 0:
        raise ValueError("an optimizer step needs supervised targets")
    missing = torch.tensor(int(any(p.grad is None for p in parameters)), device="cuda", dtype=torch.int32)
    dist.all_reduce(missing, op=dist.ReduceOp.MAX)
    if missing.item():
        raise RuntimeError("a trainable adapter parameter has no gradient")
    flat = torch.cat([p.grad.flatten() for p in parameters])
    dist.all_reduce(flat)
    flat.div_(targets)
    if not bool(flat.isfinite().all()):
        raise FloatingPointError("nonfinite globally reduced adapter gradient")
    norm = flat.norm()
    flat.mul_(torch.clamp(1 / norm.clamp_min(1e-12), max=1.0))
    offset = 0
    for p in parameters:
        p.grad.copy_(flat[offset:offset + p.numel()].view_as(p))
        offset += p.numel()
    return float(norm)


def optimizer_step(reference, model, optimizers, inputs, labels, *, learning_rate):
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    start = time.perf_counter()
    hidden = training_hidden(reference, model, inputs)
    loss = frozen_head_loss(hidden, labels, model.head)
    local_count = int((labels != -100).sum())
    loss_sum, target_count, input_count = global_numbers(float(loss.detach()), local_count, inputs.numel())
    if not math.isfinite(loss_sum) or target_count < 1:
        raise FloatingPointError("nonfinite loss or empty global training batch")
    loss.backward()
    parameters = [p for optimizer in optimizers for group in optimizer.param_groups for p in group["params"]]
    grad_norm = globally_reduce_gradients(parameters, target_count)
    for optimizer in optimizers:
        optimizer.step()
    torch.cuda.synchronize()
    seconds = torch.tensor(time.perf_counter() - start, device="cuda", dtype=torch.float64)
    dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    return {"loss": loss_sum / target_count, "supervised_tokens": int(target_count),
            "input_tokens": int(input_count), "seconds": seconds.item(), "learning_rate": learning_rate,
            "gradient_norm_before_clip": grad_norm,
            "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}


@torch.no_grad()
def evaluate(reference, model, data, *, step):
    local_sum, local_count = 0., 0
    by_mode = {mode: [0., 0.] for mode in ("low", "medium", "high")}
    for cursor in range(math.ceil(len(data) / dist.get_world_size())):
        index = cursor * dist.get_world_size() + dist.get_rank()
        inputs, labels, count = data.batch(index, device="cuda")
        hidden = training_hidden(reference, model, inputs)
        value = float(frozen_head_loss(hidden, labels, model.head))
        local_sum += value
        local_count += count
        if index < len(data):
            mode = data.windows[index]["mode"]
            by_mode[mode][0] += value
            by_mode[mode][1] += count
    loss_sum, count = global_numbers(local_sum, local_count)
    report = {"step": step, "loss": loss_sum / count, "supervised_tokens": int(count), "by_mode": {}}
    for mode, values in by_mode.items():
        total, tokens = global_numbers(*values)
        report["by_mode"][mode] = {"loss": total / tokens if tokens else None, "targets": int(tokens)}
    return report


def save_adapter_checkpoint(path: Path, adapters, optimizers, cursor, contract):
    """One shared DP state plus every rank's RNG; COMPLETE is published last."""
    rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state().cpu()}
    all_rng = [None] * dist.get_world_size()
    dist.all_gather_object(all_rng, rng)
    if dist.get_rank() == 0:
        path.mkdir(parents=True, exist_ok=False)
        payload = {"format": "archlab-native-v41-adapters-v2", "cursor": cursor, "contract": contract,
                   "adapters": {str(i): a.state_dict() for i, a in adapters.items()},
                   "optimizers": [o.state_dict() for o in optimizers], "rng": all_rng}
        target = path / "adapter-state.pt"
        with target.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_write_json(path / "COMPLETE.json",
                          {"format": payload["format"], "cursor": cursor, "contract": contract,
                           "state_bytes": target.stat().st_size, "state_sha256": sha256_file(target)},
                          allow_nan=False, create_parents=False)
    dist.barrier()


def restore_adapter_checkpoint(path: Path, adapters, optimizers, *, contract):
    marker = json.loads((path / "COMPLETE.json").read_text())
    if marker.get("format") != "archlab-native-v41-adapters-v2":
        raise ValueError("checkpoint is not the verified adapter-state format")
    if marker["contract"] != contract or (path / "adapter-state.pt").stat().st_size != marker["state_bytes"]:
        raise ValueError("incompatible or incomplete adapter checkpoint")
    if sha256_file(path / "adapter-state.pt") != marker["state_sha256"]:
        raise ValueError("adapter checkpoint checksum mismatch")
    payload = torch.load(path / "adapter-state.pt", map_location="cpu", weights_only=True)
    if payload["contract"] != contract or payload["cursor"] != marker["cursor"]:
        raise ValueError("checkpoint payload/manifest disagreement")
    if set(payload["adapters"]) != {str(i) for i in adapters} or len(payload["optimizers"]) != len(optimizers):
        raise ValueError("checkpoint module/optimizer partition mismatch")
    for i, adapter in adapters.items():
        adapter.load_state_dict(payload["adapters"][str(i)], strict=True)
    for optimizer, state in zip(optimizers, payload["optimizers"], strict=True):
        optimizer.load_state_dict(state)
    rng = payload["rng"][dist.get_rank()]
    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state(rng["cuda"])
    return payload["cursor"]


def learning_rate(step, tokens, warmup_tokens, *, budget=1_000_000_000, warmup_steps=100):
    if step < warmup_steps:
        return 1e-5 * (step + 1) / warmup_steps
    progress = min(1., max(0., (tokens - warmup_tokens) / max(1, budget - warmup_tokens)))
    return 1e-6 + .5 * (1e-5 - 1e-6) * (1 + math.cos(math.pi * progress))


def append_metric(path: Path, record):
    if dist.get_rank() == 0:
        with path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
