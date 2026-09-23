"""Bound RL allocator use and offload only checkpoint input activations.

Weights and gradients remain on their existing GPU owners. Checkpoint wrappers
and parameter names are preserved; saved input activations make an exact CPU
round trip between the original forward and its checkpoint recomputation.
"""

from __future__ import annotations

import math
from types import MethodType

import torch


def configure_gpu_budget(gib):
    if gib is None:
        return {"enabled": False}
    if (
        isinstance(gib, bool)
        or not isinstance(gib, (int, float))
        or not math.isfinite(gib)
        or gib <= 0
    ):
        raise ValueError("GPU allocator budget must be a positive finite GiB value")
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    budget = int(gib * 2**30)
    if budget >= total:
        raise ValueError("GPU allocator budget must leave room outside this process")
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    return {
        "enabled": True,
        "allocator_budget_bytes": budget,
        "device_total_bytes": total,
        "nominal_remaining_bytes": total - budget,
        "scope": "PyTorch allocator; CUDA contexts and external library allocations are additional",
    }


def _storage_key(tensor):
    return tensor.device, tensor.untyped_storage().data_ptr()


def _forward_with_input_offload(self, *args, **kwargs):
    original = self._archlab_original_checkpoint_forward
    if not torch.is_grad_enabled():
        return original(*args, **kwargs)
    # Only explicit checkpoint inputs are eligible. Saved weights, even if an
    # inner operation saves them, must never be moved by this hook.
    inputs = (*args, *kwargs.values())
    eligible = {
        _storage_key(value)
        for value in inputs
        if isinstance(value, torch.Tensor) and value.is_cuda and value.numel()
    }
    if not eligible:
        return original(*args, **kwargs)

    def pack(tensor):
        if not tensor.is_cuda or not tensor.numel() or _storage_key(tensor) not in eligible:
            return tensor
        host = torch.empty_like(tensor, device="cpu", pin_memory=True)
        host.copy_(tensor.detach(), non_blocking=False)
        self._archlab_input_offload_stats["tensor_copies"] += 1
        self._archlab_input_offload_stats["copied_bytes"] += tensor.numel() * tensor.element_size()
        return tensor.device, host

    def unpack(saved):
        if isinstance(saved, torch.Tensor):
            return saved
        device, host = saved
        return host.to(device=device, non_blocking=True)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        return original(*args, **kwargs)


def install_checkpoint_input_offload(model):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        CheckpointWrapper,
    )

    selected = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, CheckpointWrapper)
    ]
    if not selected:
        raise ValueError("activation offload requires existing checkpoint wrappers")
    for name, module in selected:
        if module.checkpoint_impl != CheckpointImpl.NO_REENTRANT:
            raise ValueError(
                f"only the existing non-reentrant checkpoint policy is admitted: {name}"
            )
        if hasattr(module, "_archlab_original_checkpoint_forward"):
            raise ValueError("checkpoint input offload is already installed")
    before = {name: id(value) for name, value in model.named_parameters()}
    for _, module in selected:
        module._archlab_original_checkpoint_forward = module.forward
        module._archlab_input_offload_stats = {"tensor_copies": 0, "copied_bytes": 0}
        module.forward = MethodType(_forward_with_input_offload, module)
    if before != {name: id(value) for name, value in model.named_parameters()}:
        raise RuntimeError("activation offload changed parameter names or identities")
    return {
        "enabled": True,
        "kind": "checkpoint-input-activations-to-pinned-CPU-v1",
        "modules": [name for name, _ in selected],
        "weights_offloaded": False,
        "gradients_offloaded": False,
        "parameter_identity_preserved": True,
    }


def input_offload_statistics(model):
    rows = [
        module._archlab_input_offload_stats
        for module in model.modules()
        if hasattr(module, "_archlab_input_offload_stats")
    ]
    return {key: sum(row[key] for row in rows) for key in ("tensor_copies", "copied_bytes")}


def qualify_replay_memory(
    model, optimizer, indexers, prompts, *, config, policy_version, stops, pad
):
    """Stress a maximum-context replay without updating weights or optimizer state."""
    import threading

    import torch.distributed as dist

    from archlab.automodel.deepseek_v41_rl_update import policy_gradient_step
    from archlab.rl.rollout import sample_rollouts

    target = config["context_limit"] - 1
    extended = [(row * ((target + len(row) - 1) // len(row)))[:target] for row in prompts]
    device = model.lm_head.weight.device
    minimum_free = [0]
    done = threading.Event()

    def observe():
        while not done.wait(0.05):
            minimum_free[0] = min(minimum_free[0], torch.cuda.mem_get_info(device)[0])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    minimum_free[0] = torch.cuda.mem_get_info(device)[0]
    watcher = threading.Thread(target=observe, daemon=True)
    watcher.start()
    try:
        rollout = sample_rollouts(
            model,
            extended,
            policy_version=policy_version,
            max_new_tokens=1,
            context_limit=config["context_limit"],
            eos_token_ids=stops,
            pad_token_id=pad,
            seed=config["seed"] + 3000000,
            retain_weights=False,
        )
        rewards = torch.zeros(
            len(prompts) // config["group_size"], config["group_size"], device=device
        )
        rewards[:, 0] = 1
        audit = policy_gradient_step(
            model,
            optimizer,
            indexers,
            rollout,
            rewards,
            lr=config["learning_rate"],
            group_size=config["group_size"],
            replay_mode="sampled-prefix",
            replay_prefixes=1,
            replay_seed=config["seed"],
            replay_tolerance=config["replay_tolerance"],
            audit_only=True,
            loss_normalization=config["loss_normalization"],
        )
        torch.cuda.synchronize(device)
    finally:
        done.set()
        watcher.join()
        optimizer.zero_grad(set_to_none=True)
    local = {
        "rank": dist.get_rank(),
        "minimum_driver_free_bytes": minimum_free[0],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "replay_verified": audit.get("replay_verified") is True,
        "replay_max_abs_error": audit.get("replay_max_abs_error"),
        "optimizer_state_empty": not any(optimizer.state.values()),
        "input_offload": input_offload_statistics(model),
    }
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, local)
    reserve = int(config["evaluation_reserve_gib"] * 2**30)
    return {
        "passed": all(
            row["replay_verified"]
            and row["optimizer_state_empty"]
            and row["minimum_driver_free_bytes"] >= reserve
            for row in rows
        ),
        "kind": "maximum-context-replay-memory-v1",
        "context_limit": config["context_limit"],
        "required_evaluation_reserve_gib": config["evaluation_reserve_gib"],
        "minimum_driver_free_gib": min(row["minimum_driver_free_bytes"] for row in rows) / 2**30,
        "peak_allocated_gib": max(row["peak_allocated_bytes"] for row in rows) / 2**30,
        "optimizer_updates": 0,
        "synthetic_tokens_for_memory_admission_only": True,
        "ranks": rows,
    }
