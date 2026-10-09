"""Small resident-policy rollout throughput measurements, not MFU estimates."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist

from archlab.rl.rollout import sample_rollouts


def _admit(errors, signature=None):
    packets = [(errors, signature)]
    if dist.is_initialized():
        packets = [None] * dist.get_world_size()
        dist.all_gather_object(packets, (errors, signature))
    if any(packet[0] for packet in packets):
        raise ValueError(f"rollout profile admission failed: {[packet[0] for packet in packets]}")
    if signature is not None and any(packet[1] != signature for packet in packets):
        raise ValueError("rollout profile configuration and prompt must match across ranks")


@contextmanager
def _preserve_execution_state(model):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    # Never initialize CUDA just to preserve RNG for a CPU-only test/process.
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.modules()]
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        for module, training in modes:
            module.training = training


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _actual_work(rollout, batch_size, max_new_tokens):
    if len(rollout.generated_ids) != batch_size:
        raise ValueError("sampler returned the wrong batch size")
    lengths = [len(row) for row in rollout.generated_ids]
    if any(not 1 <= length <= max_new_tokens for length in lengths):
        raise ValueError("generated length violates the profiling budget")
    generated = sum(lengths)
    if generated != rollout.receipt.get("generated_tokens") or generated != int(
        rollout.response_mask.sum()
    ):
        raise ValueError("actual generated tokens disagree with the sampling receipt")
    shapes = rollout.receipt.get("forward_shapes")
    if (
        not isinstance(shapes, list)
        or not shapes
        or len(shapes) != rollout.receipt.get("forward_count")
    ):
        raise ValueError("missing model-forward receipt")
    if any(
        len(shape) != 2 or shape[0] != batch_size or type(shape[1]) is not int or shape[1] < 1
        for shape in shapes
    ):
        raise ValueError("invalid model-forward shape receipt")
    return generated, len(shapes), sum(shape[0] * shape[1] for shape in shapes)


def profile_rollout_batches(
    model,
    prompt_ids: list[int],
    *,
    policy_version,
    max_new_tokens=4,
    context_limit,
    eos_token_ids,
    pad_token_id,
    seed,
    batch_sizes=(1, 4),
    warmup=1,
    repeats=1,
):
    """Warm then measure actual on-policy generation at each local batch size.

    All ranks use the same supplied prompt, copying it for every local sequence.
    The sampler retains independent rank seeds and temperature=top_p=1. Timings
    include the complete sampler call, with CUDA synchronization and a maximum
    across ranks. Throughput divides globally summed *actually generated* tokens
    by that maximum duration; early EOS does not earn hypothetical token counts.

    Input work is the sum of each forward receipt's batch*sequence dimensions,
    including right padding and repeated uncached prefixes. A distributed forward
    is counted once per participating rank in ``rank_forward_calls_sum``.
    Warmup is excluded. Python/NumPy/Torch CPU and initialized CUDA RNG states,
    and each module's original train/eval mode, are restored even on failure.
    No optimizer is accepted or run; no FLOP estimate or MFU percentage is inferred.
    The sampler inherits the actor's optional retained-weight setting. Each warmup
    and measurement is a separate residency context, so its timing includes one
    initial gather and complete cleanup rather than hiding setup in the warmup.
    """
    errors = []
    signature = None
    try:
        if (
            not isinstance(prompt_ids, list)
            or not prompt_ids
            or any(type(token) is not int or token < 0 for token in prompt_ids)
        ):
            raise ValueError("prompt_ids must be a nonempty list of token IDs")
        if not isinstance(policy_version, str) or not policy_version:
            raise ValueError("policy_version must be nonempty")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
            raise ValueError("profiling max_new_tokens must lie in [1,128]")
        if type(context_limit) is not int or len(prompt_ids) + max_new_tokens > context_limit:
            raise ValueError("prompt plus generation budget exceeds context_limit")
        if type(seed) is not int or type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError("seed and pad_token_id must be integers")
        sizes = tuple(batch_sizes)
        if (
            not sizes
            or len(set(sizes)) != len(sizes)
            or any(type(size) is not int or not 1 <= size <= 32 for size in sizes)
        ):
            raise ValueError("batch_sizes must be unique integers in [1,32]")
        if (
            type(warmup) is not int
            or not 0 <= warmup <= 10
            or type(repeats) is not int
            or not 1 <= repeats <= 10
        ):
            raise ValueError("warmup/repeats must be bounded nonnegative/positive integers")
        stops = tuple(sorted(set(eos_token_ids)))
        if not stops or any(type(token) is not int or token < 0 for token in stops):
            raise ValueError("native EOS token IDs must be supplied")
        device = model.lm_head.weight.device
        vocabulary = model.lm_head.weight.shape[0]
        if max(prompt_ids + list(stops) + [pad_token_id]) >= vocabulary:
            raise ValueError("profile token outside the model vocabulary")
        prompt_digest = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()
        signature = (
            policy_version,
            max_new_tokens,
            context_limit,
            stops,
            pad_token_id,
            seed,
            sizes,
            warmup,
            repeats,
            prompt_digest,
        )
    except (AttributeError, TypeError, ValueError) as error:
        errors.append(str(error))
    _admit(errors, signature)
    world = dist.get_world_size() if dist.is_initialized() else 1
    results = []
    with _preserve_execution_state(model):
        for batch_index, batch_size in enumerate(sizes):
            prompts = [list(prompt_ids) for _ in range(batch_size)]
            samples = []
            for iteration in range(warmup + repeats):
                _synchronize(device)
                if dist.is_initialized():
                    dist.barrier()
                measured = iteration >= warmup
                if measured:
                    started = time.perf_counter()
                rollout = sample_rollouts(
                    model,
                    prompts,
                    policy_version=policy_version,
                    max_new_tokens=max_new_tokens,
                    context_limit=context_limit,
                    eos_token_ids=stops,
                    pad_token_id=pad_token_id,
                    seed=seed + batch_index * 1_000_000 + iteration,
                    temperature=1.0,
                    top_p=1.0,
                    device=device,
                )
                _synchronize(device)
                if not measured:
                    del rollout
                    continue
                elapsed = time.perf_counter() - started
                errors = []
                try:
                    work = _actual_work(rollout, batch_size, max_new_tokens)
                    if not math.isfinite(elapsed) or elapsed <= 0:
                        raise ValueError("profile duration must be finite and positive")
                except (AttributeError, TypeError, ValueError) as error:
                    errors.append(str(error))
                _admit(errors)
                counts = torch.tensor(work, device=device, dtype=torch.int64)
                duration = torch.tensor(elapsed, device=device, dtype=torch.float64)
                maximum_forwards = torch.tensor(work[1], device=device, dtype=torch.int64)
                if dist.is_initialized():
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                    dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                    dist.all_reduce(maximum_forwards, op=dist.ReduceOp.MAX)
                samples.append(
                    {
                        "repeat": iteration - warmup,
                        "max_rank_seconds": float(duration),
                        "actual_generated_tokens_global": int(counts[0]),
                        "rank_forward_calls_sum": int(counts[1]),
                        "model_forward_calls_per_rank": int(maximum_forwards),
                        "input_tokens_processed_global": int(counts[2]),
                        "generated_tokens_per_second": int(counts[0]) / float(duration),
                        "rollout_backend": rollout.receipt.get(
                            "backend", "unspecified-test-backend"
                        ),
                        "retained_weights": rollout.receipt.get("retained_weights", False),
                    }
                )
                del rollout
            tokens = sum(sample["actual_generated_tokens_global"] for sample in samples)
            seconds = sum(sample["max_rank_seconds"] for sample in samples)
            results.append(
                {
                    "batch_size_per_rank": batch_size,
                    "global_batch_size": batch_size * world,
                    "warmup_calls_per_rank": warmup,
                    "measured_calls_per_rank": repeats,
                    "actual_generated_tokens_global": tokens,
                    "max_rank_seconds_sum": seconds,
                    "rank_forward_calls_sum": sum(
                        sample["rank_forward_calls_sum"] for sample in samples
                    ),
                    "input_tokens_processed_global": sum(
                        sample["input_tokens_processed_global"] for sample in samples
                    ),
                    "generated_tokens_per_second": tokens / seconds,
                    "samples": samples,
                }
            )
    throughput = {
        item["batch_size_per_rank"]: item["generated_tokens_per_second"] for item in results
    }
    return {
        "format": "archlab-rl-rollout-throughput-v1",
        "scope": "uncached resident-policy generation; no MFU percentage claimed",
        "policy_version": policy_version,
        "world_size": world,
        "measurement_device": str(device),
        "prompt_sha256": prompt_digest,
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": max_new_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "cached": False,
        "retained_weights": getattr(model, "_archlab_rl_retain_weights", False),
        "residency_setup_and_cleanup_included": True,
        "input_token_count_includes_padding": True,
        "warmup_included_in_measurements": False,
        "rng_and_module_modes_restored": True,
        "batches": results,
        "batch4_over_batch1_throughput_ratio": throughput[4] / throughput[1]
        if 1 in throughput and 4 in throughput
        else None,
    }
