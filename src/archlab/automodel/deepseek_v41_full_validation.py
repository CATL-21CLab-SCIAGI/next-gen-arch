# ruff: noqa: I001  # Preserve the checkpoint-qualified import order.
"""Fixed held-out, resident-weight validation with no training RNG side effects."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import random
import time


LEGACY_TRAINER_SHA256 = "aa9639749370e83f2190ca9101305c1f1188ae12950f7654084d5e667cbec034"
LEGACY_UPDATE_SOURCE_SHA256 = "69b5c11e0a20714a4eeb3d6411228058bf000c8a233f9b4143389aa2ba722b65"


def admit_evaluation_upgrade(saved, current, update_function):
    """Admit only the recorded v2 -> periodic-eval change, never model/runtime drift."""
    if saved == current:
        return
    mutable = {"project_commit", "implementation_sha256", "periodic_evaluation"}
    if {k: v for k, v in saved.items() if k not in mutable} != {
        k: v for k, v in current.items() if k not in mutable
    }:
        raise ValueError("resume changed model, optimizer, data cursor contract, or runtime")
    old, new = saved["implementation_sha256"], current["implementation_sha256"]
    trainer = "automodel/deepseek_v41_full_training.py"
    validation = "automodel/deepseek_v41_full_validation.py"
    if old.get(trainer) != LEGACY_TRAINER_SHA256 or validation in old:
        raise ValueError("only the recorded v2 trainer is eligible for the eval-only upgrade")
    if set(new) != set(old) | {validation} or any(
        new[k] != v for k, v in old.items() if k != trainer
    ):
        raise ValueError(
            "resume changed a protected model, optimizer, checkpoint, or data implementation"
        )
    digest = hashlib.sha256(inspect.getsource(update_function).encode()).hexdigest()
    if digest != LEGACY_UPDATE_SOURCE_SHA256:
        raise ValueError("training update changed during the eval-only upgrade")


def make_plan(pilot, target_budget):
    if target_budget <= 0:
        raise ValueError("validation budget must be positive")
    remaining, selected = target_budget, []
    for index, window in enumerate(pilot.windows):
        count = min(remaining, window["targets"])
        if count:
            selected.append({"index": index, "targets": count})
            remaining -= count
        if not remaining:
            break
    if remaining:
        raise ValueError("held-out pilot has fewer targets than the evaluation budget")
    payload = {
        "format": "archlab-full-periodic-eval-v1",
        "targets": target_budget,
        "pilot_windows_sha256": pilot.manifest["windows_sha256"],
        "source_ready_sha256": pilot.manifest["source_ready_sha256"],
        "order_seed": pilot.order_seed,
        "windows": selected,
    }
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def capped_batch(pilot, item, device):
    import torch

    if item is None:
        return pilot.batch(len(pilot), device=device)
    inputs, labels, count = pilot.batch(item["index"], device=device)
    if not 0 < item["targets"] <= count:
        raise ValueError("invalid held-out target cap")
    labels = labels.clone()
    positions = torch.where(labels.reshape(-1) != -100)[0]
    labels.reshape(-1)[positions[item["targets"] :]] = -100
    return inputs, labels, item["targets"]


@contextmanager
def evaluation_state(model):
    import numpy as np
    import torch

    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def head_totals(hidden, labels, weight, chunk_size=128):
    """Count, summed NLL, top-1/top-5 hits, and entropy; exact assistant mask."""
    import torch

    x, targets = hidden.flatten(0, -2), labels.flatten()
    positions = torch.where(targets != -100)[0]
    total = torch.zeros(5, device=hidden.device, dtype=torch.float64)
    total[0] = positions.numel()
    for ids in positions.split(chunk_size):
        if not ids.numel():
            continue
        logits = torch.nn.functional.linear(x[ids].float(), weight)
        logp = logits.log_softmax(-1)
        selected = targets[ids]
        total[1] -= logp.gather(1, selected[:, None]).sum(dtype=torch.float64)
        total[2] += (logits.argmax(-1) == selected).sum()
        total[3] += (
            (logits.topk(min(5, logits.shape[-1]), -1).indices == selected[:, None]).any(-1).sum()
        )
        total[4] -= (logp.exp() * logp).sum(dtype=torch.float64)
    return total


def evaluate_batches(model, batches):
    """Every rank must execute the same number of forward calls, including padding."""
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    device = next(model.parameters()).device
    total = torch.zeros(5, device=device, dtype=torch.float64)
    started = time.monotonic()
    with evaluation_state(model):
        for inputs, labels, targets in batches:
            hidden = model(input_ids=inputs, return_hidden_states=True).hidden_states
            head = model.lm_head
            sharded = isinstance(head, FSDPModule)
            if sharded:
                head.unshard()
            try:
                if isinstance(head.weight, DTensor) or head.weight.dtype != torch.float32:
                    raise ValueError("validation requires the unsharded native FP32 head")
                values = head_totals(hidden, labels, head.weight)
                if int(values[0]) != targets:
                    raise ValueError("held-out mask and target count disagree")
                total += values
            finally:
                if sharded:
                    head.reshard()
            del hidden
    dist.all_reduce(total)
    torch.cuda.synchronize()
    seconds = torch.tensor(time.monotonic() - started, device=device, dtype=torch.float64)
    dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    if not bool(total.isfinite().all()) or not total[0] > 0:
        raise FloatingPointError("nonfinite or empty held-out evaluation")
    count, nll, top1, top5, entropy = total.tolist()
    ce = nll / count
    return {
        "loss": ce,
        "cross_entropy": ce,
        "perplexity": math.exp(ce),
        "top1_token_accuracy": top1 / count,
        "top5_token_accuracy": top5 / count,
        "mean_predictive_entropy": entropy / count,
        "targets": int(count),
        "seconds": float(seconds),
    }


def evaluate_pilot(model, pilot, plan):
    import torch.distributed as dist

    rank, world = dist.get_rank(), dist.get_world_size()
    entries = plan["windows"]
    batches = (
        capped_batch(pilot, entries[first + rank] if first + rank < len(entries) else None, "cuda")
        for first in range(0, len(entries), world)
    )
    result = evaluate_batches(model, batches)
    if result["targets"] != plan["targets"]:
        raise ValueError("distributed validation target count differs from fixed plan")
    return {**result, "plan_sha256": plan["sha256"], "windows": len(entries)}
