"""Architecture costs and phase accounting routed through verl's own logger."""
from __future__ import annotations

import math
from collections import defaultdict

from archlab.architectures.limite_cost import baseline_flops, converted_flops

B300_DENSE_BF16_FLOPS = 2.25e15


def estimate_limite_flops(config, tokens_sum, batch_seqlens, delta_time):
    conversion = getattr(config, "conversion", {})
    if not conversion and config.adapter.get("variant") != "normal":
        raise ValueError("Limite FLOP ledger is qualified only for native normal preludes")
    if (not math.isfinite(delta_time) or delta_time <= 0 or tokens_sum != sum(batch_seqlens)
            or any(n < 0 or int(n) != n for n in batch_seqlens)):
        raise ValueError("invalid FLOP measurement")
    def ledger(n):
        return (converted_flops(config, n) if conversion
                else baseline_flops(config, n, config.trainable_mode))
    total = sum(ledger(int(n)).per_token * n
                for n in batch_seqlens if n > 0)
    return total / delta_time / 1e12


def register_flops():
    from verl.utils.flops_counter import ESTIMATE_FUNC

    ESTIMATE_FUNC["archlab_limite"] = estimate_limite_flops


def step_metrics(*, config, batch, tags, timing, step):
    """GPU-seconds mean allocated phase time, never measured device idleness.

    Colocated GPUs generate while the learner waits. Do not describe gen*time
    as idle GPU time, or sum nested timers into allocation time twice.
    """
    if config.trainer.v1.trainer_mode not in {"sync", "colocate_async"}:
        raise ValueError("phase accounting needs a resource map for separate rollout pools")
    gpu_count = config.trainer.nnodes * config.trainer.n_gpus_per_node
    seconds = float(timing["step"])
    if seconds <= 0 or any(not math.isfinite(v) or v < 0 for v in timing.values()):
        raise ValueError("invalid step timing")
    phases = ("gen", "reward", "old_log_prob", "ref", "values", "adv", "update_critic",
              "update_actor", "save_checkpoint", "update_weights")
    measured = sum(timing.get(k, 0) for k in phases)
    if measured > seconds * 1.01:
        raise ValueError("phase timers overlap; cannot report additive GPU-seconds")
    result = {f"perf/allocated_gpu_seconds/{k}": gpu_count * timing.get(k, 0) for k in phases}
    result.update({"perf/allocated_gpu_seconds/step": gpu_count * seconds,
                   "perf/allocated_gpu_seconds/unattributed": gpu_count * max(0, seconds - measured),
                   "perf/measured_idle_available": 0,
                   "training/trainer_steps": step, "training/policy_version": step})
    if len(tags) != len(batch.batch["response_mask"]):
        raise ValueError("rollout tags and unpadded metrics batch differ")
    groups = defaultdict(list)
    sessions = defaultdict(set)
    rewards = batch.batch["rm_scores"].sum(-1).tolist()
    for tag, reward in zip(tags, rewards, strict=True):
        groups[tag["group_id"]].append(reward)
        sessions[tag["group_id"]].add(tag["session_id"])
    # One session may produce multiple context segments. Do not mistake those
    # segments for independent GRPO samples or crash logging on a failed group.
    incomplete = sum(len(values) != config.actor_rollout_ref.rollout.n for values in sessions.values())
    useful = sum(len(set(values)) > 1 for values in groups.values())
    generated = sum(tag["generated_tokens"] for tag in tags)
    result.update({"rollout/consumed_generated_tokens": generated,
                   "rollout/consumed_observation_tokens": sum(tag["observation_tokens"] for tag in tags),
                   "rollout/native_eos_tokens": sum(tag["native_eos_tokens"] for tag in tags),
                   "rollout/context_limit_count": sum(tag["context_limit_reached"] for tag in tags),
                   "rollout/truncated_count": sum(tag["termination_reason"] == "truncated" for tag in tags),
                   "rollout/consumed_groups": len(groups), "rollout/useful_groups": useful,
                   "rollout/incomplete_groups": incomplete,
                   "perf/consumed_generated_tokens_per_second": generated / seconds,
                   "perf/bf16_reference_tflops_per_gpu": B300_DENSE_BF16_FLOPS / 1e12})
    return result
