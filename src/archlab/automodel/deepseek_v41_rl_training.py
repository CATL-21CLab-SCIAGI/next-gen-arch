"""Bounded online RLOO continuation of the matched pretrained V4.1 pair.

Run with torchrun on the existing sixteen-rank ownership mesh. Numerical
qualification is performed before admission and never applies synthetic-reward
updates. Each real update uses fresh current-policy samples and outcome rewards.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from archlab.artifacts import atomic_write_json, sha256_file

SOURCE_FILES = (
    "architectures/deepseek_v41_swiglu.py",
    "architectures/ordered_scatter.py",
    "source_compatibility.py",
    "data/source-formatting-20260923.json",
    "automodel/deepseek_v41_rl_training.py",
    "automodel/deepseek_v41_rl_model.py",
    "automodel/deepseek_v41_rl_head.py",
    "automodel/deepseek_v41_rl_update.py",
    "automodel/deepseek_v41_rl_fsdp_probe.py",
    "automodel/deepseek_v41_full_checkpoint.py",
    "optimizers/sharded_adafactor.py",
    "rl/rollout.py",
    "rl/evaluation.py",
    "rl/rewards.py",
    "rl/objectives.py",
    "rl/nemotron_data.py",
    "rl/profiling.py",
    "rl/weight_residency.py",
    "automodel/deepseek_v41_rl_cache.py",
    "automodel/deepseek_v41_rl_memory.py",
    "rl/regularization.py",
    "automodel/deepseek_v41_rl_memory_policy.py",
    "architectures/simplicial_decode.py",
    "preprocessing/deepseek_v41.py",
)

RL_NUMERICAL_PRECISION = {
    "cuda_matmul_allow_tf32": False,
    "cuda_matmul_allow_bf16_reduced_precision_reduction": False,
}


def configure_numerical_precision():
    """Apply the explicit RL numerical contract without changing parent weights."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    observed = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
    }
    if observed != RL_NUMERICAL_PRECISION:
        raise RuntimeError("Cannot establish the declared RL matrix precision")
    return observed


def replay_options(config, *, rollout_step, world, qualification=False):
    """Both actors select the same deterministic global replay-time RNG stream."""
    options = {
        "replay_mode": config["replay_mode"],
        "replay_prefixes": config["replay_prefixes"],
        "replay_seed": config["seed"] + (1000000 if qualification else rollout_step * world),
    }
    if config.get("loss_normalization", "sequence_sum") != "sequence_sum":
        options["loss_normalization"] = config["loss_normalization"]
    return options


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def resolve_value(value):
    if isinstance(value, dict):
        return {key: resolve_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_value(item) for item in value]
    if isinstance(value, str) and value.startswith("env:"):
        return os.environ[value[4:]]
    if isinstance(value, str) and value.startswith("package:"):
        return str(Path(__file__).resolve().parents[1] / value[8:])
    return value


def read_recipe(path):
    import yaml

    path = Path(path)
    text = path.read_text()
    config = resolve_value(json.loads(text) if path.suffix == ".json" else yaml.safe_load(text))
    validate_recipe(config)
    return config


def validate_recipe(config):
    required = (
        "assets",
        "weights",
        "container_kernel_packages",
        "data_manifest",
        "data_manifest_sha256",
        "prompt_template",
        "parents",
        "seed",
        "learning_rate",
        "max_new_tokens",
        "context_limit",
        "component_qualification",
        "component_qualification_sha256",
    )
    if any(key not in config for key in required):
        raise ValueError("RL recipe is missing required provenance or sampling fields")
    for key in ("learning_rate", "replay_tolerance"):
        if key in config:
            config[key] = float(config[key])
    config.setdefault("prompt_order", "length-bucketed-global-batches-shuffled-v1")
    if config["prompt_order"] != "length-bucketed-global-batches-shuffled-v1":
        raise ValueError("Unsupported paired prompt ordering")
    if (
        config.get("schema_version") != 1
        or config.get("family") != "full"
        or config.get("world_size") != 16
        or config.get("group_size") != 4
        or config.get("prompts_per_rank") != 1
    ):
        raise ValueError("This admitted pair uses full16, one prompt/rank, four samples/group")
    if config.get("temperature", 1.0) != 1 or config.get("top_p", 1.0) != 1:
        raise ValueError("On-policy RLOO requires temperature=top_p=1")
    integers = {
        "max_new_tokens": 128,
        "context_limit": 2048,
        "max_rollout_updates": 128,
        "pilot_updates": 2,
        "eval_count": 64,
        "eval_interval_updates": 8,
        "qualification_eval_count": 8,
        "eval_local_batch_size": 1,
        "checkpoint_interval_updates": 16,
        "head_chunk_size": 128,
        "qualification_max_new_tokens": 16,
        "replay_prefixes": 4,
    }
    for key, default in integers.items():
        config.setdefault(key, default)
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (
        config["max_rollout_updates"] > 128
        or config["pilot_updates"] > config["max_rollout_updates"]
    ):
        raise ValueError("The bounded RL contract allows at most 128 fresh rollout batches")
    if not 0 < config["learning_rate"] <= 1e-4 or not math.isfinite(config["learning_rate"]):
        raise ValueError("Invalid conservative RL learning rate")
    if type(config["seed"]) is not int:
        raise ValueError("A fixed integer RL seed is required")
    config.setdefault("replay_tolerance", 0.02)
    config.setdefault("retain_weights", True)
    config.setdefault("cache_policy", False)
    config.setdefault("weight_reserve_gib", 16)
    config.setdefault("freeze_router", False)
    config.setdefault("checkpoint_input_offload", False)
    config.setdefault("inplace_moe_accumulation", False)
    config.setdefault("checkpoint_expert_activations", False)
    config.setdefault("serialize_backward_gathers", False)
    config.setdefault("unshard_on_compute_stream", False)
    config.setdefault("hc_activation_offload", False)
    config.setdefault("gpu_memory_budget_gib", None)
    config.setdefault("evaluation_reserve_gib", 0.0)
    config.setdefault("initial_evaluation", True)
    config.setdefault("qualification_evaluation", True)
    config.setdefault("loss_normalization", "sequence_sum")
    from archlab.rl.regularization import DEFAULT_LENGTH_PENALTY, validate_length_penalty

    config.setdefault("length_penalty", dict(DEFAULT_LENGTH_PENALTY))
    validate_length_penalty(config["length_penalty"])
    if any(
        type(config[key]) is not bool
        for key in (
            "freeze_router",
            "checkpoint_input_offload",
            "inplace_moe_accumulation",
            "checkpoint_expert_activations",
            "serialize_backward_gathers",
            "unshard_on_compute_stream",
            "hc_activation_offload",
            "initial_evaluation",
            "qualification_evaluation",
        )
    ):
        raise ValueError("router freeze and checkpoint input offload must be boolean")
    if config["loss_normalization"] not in ("sequence_sum", "prompt_token_mean"):
        raise ValueError("unsupported declared RL loss normalization")
    if config["checkpoint_expert_activations"] and not config["inplace_moe_accumulation"]:
        raise ValueError("expert activation checkpoints require the qualified memory policy")
    budget = config["gpu_memory_budget_gib"]
    if budget is not None and (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or budget <= 0
    ):
        raise ValueError("GPU allocator budget must be finite positive GiB")
    if budget is not None and config["retain_weights"]:
        raise ValueError(
            "shared-GPU memory-budget runs must reshard gathered weights after forward"
        )
    reserve = config["evaluation_reserve_gib"]
    if (
        isinstance(reserve, bool)
        or not isinstance(reserve, (int, float))
        or not math.isfinite(reserve)
        or reserve < 0
    ):
        raise ValueError("evaluation reserve must be a finite nonnegative GiB value")
    if reserve and budget is None:
        raise ValueError("evaluation reserve requires a hard training allocator budget")
    if type(config["retain_weights"]) is not bool or config["weight_reserve_gib"] < 16:
        raise ValueError("Weight residency requires an explicit flag and at least 16 GiB reserve")
    if (
        type(config["cache_policy"]) is not bool
        or config["cache_policy"]
        and not config["retain_weights"]
    ):
        raise ValueError("resident cache requires an explicit boolean flag and retained weights")
    if config["cache_policy"]:
        if config["eval_local_batch_size"] != 1:
            raise ValueError(
                "cached evaluation requires one prompt per rank for uniform local lengths"
            )
        if config["qualification_max_new_tokens"] < min(512, config["max_new_tokens"]):
            raise ValueError(
                "cached policy qualification must replay a rollout of at least 512 tokens or the full shorter budget"
            )
    if not 0 < config["replay_tolerance"] <= 0.02:
        raise ValueError("Replay tolerance may not exceed the reviewed 0.02 nats")
    config.setdefault("replay_mode", "sampled-prefix")
    if config["replay_mode"] != "sampled-prefix":
        raise ValueError(
            "Production RL requires sampled-prefix replay of the actual behavior policy"
        )
    config.setdefault("numerical_precision", dict(RL_NUMERICAL_PRECISION))
    if config["numerical_precision"] != RL_NUMERICAL_PRECISION:
        raise ValueError("RL must disable TF32 and BF16 reduced-precision matmul reduction")
    for variant in ("normal", "simplicial"):
        parent = config["parents"][variant]
        if not parent.get("checkpoint") or not parent.get("marker_sha256"):
            raise ValueError("Both exact parent checkpoint identities are required")


def load_data(config):
    """Verify the approved manifest, every row, and prior-exposure exclusions."""
    from archlab.rl.nemotron_data import problem_key
    from archlab.rl.rewards import canonical_math_answer

    path = Path(config["data_manifest"])
    if sha256_file(path) != config["data_manifest_sha256"]:
        raise ValueError("RL data manifest fingerprint changed")
    manifest = json.loads(path.read_text())
    if manifest.get("training_authorized") is not True:
        raise ValueError("RL requires explicitly approved problem-disjoint data")
    if manifest.get("solution_columns_read") != []:
        raise ValueError("RL preparation must exclude reference solution trajectories")
    exclusion = manifest["exclusion_index"]
    if sha256_file(exclusion["path"]) != exclusion["sha256"]:
        raise ValueError("Historical problem exclusion index changed")
    previous = json.loads(Path(exclusion["path"]).read_text())
    exposed, uuids = set(previous["problem_sha256"]), set(previous["uuid"])
    splits = {}
    for name in ("train", "heldout"):
        spec = manifest["files"][name]
        if sha256_file(spec["path"]) != spec["sha256"]:
            raise ValueError(f"Changed RL {name} JSONL")
        with Path(spec["path"]).open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        identities = set()
        for row in rows:
            prompt = row["prompt"]
            if (
                len(prompt) != 1
                or set(prompt[0]) != {"role", "content"}
                or prompt[0]["role"] != "user"
            ):
                raise ValueError(
                    "RL policy input must contain one user problem, with no answer trace"
                )
            key = problem_key(prompt[0]["content"])
            if (
                row["id"] != key
                or row["problem_sha256"] != key
                or key in identities
                or key in exposed
            ):
                raise ValueError("Repeated, changed, or previously exposed RL problem")
            if row.get("uuid") and row["uuid"] in uuids:
                raise ValueError("Previously exposed RL problem UUID")
            if (
                canonical_math_answer(row["expected_answer"]) != row["canonical_answer"]
                or row["canonical_answer"] is None
            ):
                raise ValueError("Unsupported or changed reward reference")
            identities.add(key)
        if len(rows) != spec["rows"]:
            raise ValueError("RL split row count differs from its seal")
        splits[name] = rows
    if {row["id"] for row in splits["train"]} & {row["id"] for row in splits["heldout"]}:
        raise ValueError("RL training and heldout problems overlap")
    if (
        len(splits["train"])
        < config["max_rollout_updates"] * config["world_size"] * config["prompts_per_rank"]
    ):
        raise ValueError("RL data cannot cover the complete budget without repeating problems")
    if len(splits["heldout"]) < max(config["eval_count"], config["qualification_eval_count"]):
        raise ValueError("Heldout split cannot cover the exact evaluation budget")
    return splits, manifest


def encode_examples(rows, *, tokenizer, renderer, user_template, context_limit, max_new_tokens):
    result = []
    for row in rows:
        content = user_template.format(problem=row["prompt"][0]["content"])
        encoded = renderer.encoder.encode_messages(
            [{"role": "user", "content": content}], thinking_mode="chat"
        )
        tokens = tokenizer.encode(encoded, add_special_tokens=False)
        if not tokens or len(tokens) + max_new_tokens > context_limit:
            raise ValueError(f"Frozen RL problem exceeds the prompt budget: {row['id']}")
        result.append(
            {
                "problem_id": row["id"],
                "prompt_ids": tokens,
                "expected_answer": row["expected_answer"],
            }
        )
    return result


def length_bucket_order(examples, *, global_batch_size, seed):
    """Shuffle full length-sorted global batches; preserve every example exactly once."""
    if type(global_batch_size) is not int or global_batch_size < 1:
        raise ValueError("global_batch_size must be positive")
    ordered = sorted(examples, key=lambda row: (len(row["prompt_ids"]), row["problem_id"]))
    full_end = len(ordered) // global_batch_size * global_batch_size
    batches = [
        ordered[start : start + global_batch_size]
        for start in range(0, full_end, global_batch_size)
    ]
    random.Random(seed).shuffle(batches)
    result = [row for batch in batches for row in batch] + ordered[full_end:]
    return result, {
        "algorithm": "length-bucketed-global-batches-shuffled-v1",
        "seed": seed,
        "global_prompt_batch": global_batch_size,
        "partial_tail_retained_last": len(ordered) - full_end,
        "ordered_problem_ids_sha256": digest([row["problem_id"] for row in result]),
    }


class Activity:
    """Rank-zero progress heartbeat; contains no distributed or CUDA operations."""

    def __init__(self, output):
        self.path = Path(output) / "ACTIVITY.json"
        self.state = {"phase": "initializing"}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            atomic_write_json(
                self.path, {**self.state, "heartbeat_unix": time.time()}, allow_nan=False
            )
            self.stop.wait(10)

    def set(self, phase, **fields):
        self.state = {"phase": phase, **fields}
        print(json.dumps({"event": "rl_activity", **self.state}), flush=True)
        atomic_write_json(self.path, {**self.state, "heartbeat_unix": time.time()}, allow_nan=False)

    def close(self):
        self.stop.set()
        self.thread.join()
        atomic_write_json(self.path, {**self.state, "heartbeat_unix": time.time()}, allow_nan=False)


def rank_groups(examples, prompt_cursor, *, rank, world, prompts_per_rank, group_size):
    end = prompt_cursor + world * prompts_per_rank
    if prompt_cursor < 0 or end > len(examples):
        raise ValueError("RL prompt cursor would repeat or exhaust the sealed training split")
    first = prompt_cursor + rank * prompts_per_rank
    groups = examples[first : first + prompts_per_rank]
    return groups, [list(row["prompt_ids"]) for row in groups for _ in range(group_size)]


def append_jsonl(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def gather(value):
    import torch.distributed as dist

    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def source_identity():
    root = Path(__file__).resolve().parents[1]
    return {name: sha256_file(root / name) for name in SOURCE_FILES}


def make_contract(config, variant, loading, marker, tokenizer_provenance):
    from archlab.rl.rewards import REWARD_BACKEND

    runtime = {
        key: value
        for key, value in loading.items()
        if key not in ("local_parameter_gib", "rl_transition")
    }
    commit = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"], text=True
    ).strip()
    return json.loads(
        json.dumps(
            {
                "format": "archlab-v41-online-rl-v1",
                "variant": variant,
                "world_size": 16,
                "all_text_parameters_unfrozen": not config.get("freeze_router", False),
                "cpu_offload": False,
                "objective": (
                    "on-policy-REINFORCE-leave-one-out-sequence-sum-world-mean"
                    if config.get("loss_normalization", "sequence_sum") == "sequence_sum"
                    else "on-policy-REINFORCE-leave-one-out-group-token-mean-prompt-mean"
                ),
                "gradient_estimator": {
                    "replay_mode": config["replay_mode"],
                    "replay_prefixes": config["replay_prefixes"],
                    "prefix_selection": "uniform-without-replacement-over-global-generation-times",
                    "scale": "global_forward_count/selected_prefix_count",
                    "replay_seed": "recipe.seed + rollout_step * world_size; qualification seed + 1000000",
                    "original_sampling_inputs_masks_and_head_batch": not config.get(
                        "cache_policy", False
                    ),
                    "cached_policy_full_prefix_equivalence": config.get("cache_policy", False),
                },
                "numerical_precision": config["numerical_precision"],
                "rollout_weight_residency": {
                    "enabled": config["retain_weights"],
                    "minimum_free_gib": config["weight_reserve_gib"],
                },
                "reference_kl": False,
                "auxiliary_training_objectives": False,
                "optimizer": "fresh-FP32-factored-Adafactor-stochastic-BF16",
                "project_commit": commit,
                "source_commit": commit,
                "algorithm": "RLOO",
                "group_size": config["group_size"],
                "reward_backend": REWARD_BACKEND,
                "parent_checkpoint": {
                    "path": config["parents"][variant]["checkpoint"],
                    "sha256": config["parents"][variant]["marker_sha256"],
                },
                "data_manifest": {
                    "path": config["data_manifest"],
                    "sha256": config["data_manifest_sha256"],
                },
                "recipe": config,
                "runtime": runtime,
                "tokenizer": tokenizer_provenance,
                "implementation_sha256": source_identity(),
                "parent_cursor": marker["cursor"],
                "parent_marker_sha256": config["parents"][variant]["marker_sha256"],
            },
            allow_nan=False,
        )
    )


def component_gate(config, loading):
    path = Path(config["component_qualification"])
    if sha256_file(path) != config["component_qualification_sha256"]:
        raise ValueError("Component qualification fingerprint changed")
    receipt = json.loads(path.read_text())
    root = Path(__file__).resolve().parents[1]
    required_sources = (
        "automodel/deepseek_v41_rl_head.py",
        "rl/rollout.py",
        "automodel/deepseek_v41_rl_fsdp_probe.py",
        "rl/weight_residency.py",
        "automodel/deepseek_v41_rl_update.py",
    )
    sealed_sources = receipt.get("implementation_sha256", {})
    if set(sealed_sources) != set(required_sources) or any(
        sealed_sources[name] != sha256_file(root / name) for name in required_sources
    ):
        raise ValueError("Component qualification implementation hashes differ from current code")
    ranks = receipt.get("ranks", [])
    if (
        receipt.get("weight_residency_qualified") is not True
        or receipt.get("sampled_prefix_policy_gradient_qualified") is not True
    ):
        raise ValueError("Prefix replay and retained-weight cleanup require GPU qualification")
    if (
        receipt.get("passed") is not True
        or receipt.get("world_size") != 16
        or sorted(row.get("rank", -1) for row in ranks) != list(range(16))
        or any(
            row.get("passed") is not True
            or not row.get("signed_coefficients")
            or not row.get("masked_targets")
            or row.get("rollout_replay_max_abs_error", math.inf) > 2e-6
            for row in ranks
        )
    ):
        raise ValueError("Missing successful sixteen-rank signed-gradient component qualification")
    for key in ("container_image", "cuda", "nccl"):
        if receipt.get(key) != loading.get(key):
            raise ValueError(f"Component qualification runtime differs: {key}")
    normalize = lambda version: version.replace(".nv26.04.", ".nv26.4.")  # noqa: E731
    if normalize(receipt["torch"]) != normalize(loading["packages"]["torch"]):
        raise ValueError("Component qualification Torch version differs")
    return {
        "path": str(path),
        "sha256": config["component_qualification_sha256"],
        "passed": True,
        "world_size": 16,
        "synthetic_policy_updates": 0,
    }


def admit_qualification(receipt, contract):
    if (
        receipt.get("passed") is not True
        or receipt.get("contract_digest") != digest(contract)
        or receipt.get("kind") != "online-policy-numerical-v1"
        or receipt.get("synthetic_optimizer_updates") != 0
    ):
        raise ValueError("RL numerical qualification is missing, failed, or stale")
    reserve = contract.get("recipe", {}).get("evaluation_reserve_gib", 0)
    if reserve:
        memory = receipt.get("memory_admission", {})
        if (
            memory.get("passed") is not True
            or memory.get("kind") != "maximum-context-accumulated-replay-memory-v2"
            or memory.get("replay_prefixes", 0) < 2
            or memory.get("optimizer_state_reserve_bytes", 0) <= 0
            or memory.get("required_evaluation_reserve_gib") != reserve
            or memory.get("context_limit") != contract["recipe"]["context_limit"]
            or memory.get("minimum_driver_free_gib", -1) < reserve
        ):
            raise ValueError(
                "shared-GPU training requires its maximum-context replay memory admission"
            )
    if contract.get("recipe", {}).get("cache_policy", False):
        ranks = receipt.get("ranks", [])
        if (
            len(ranks) != 16
            or sorted(row.get("rank", -1) for row in ranks) != list(range(16))
            or any(row.get("cache_equivalence", {}).get("passed") is not True for row in ranks)
        ):
            raise ValueError(
                "cached RL requires all sixteen actual actors to pass prefix equivalence"
            )


def _snapshot_rng():
    import numpy as np
    import torch

    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
    )


def _restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state(state[3])


def qualify_leaf_head(device, width):
    """Independent dense-vocabulary signed derivative oracle on the actor GPU."""
    import torch
    from torch.nn import functional as F

    from archlab.automodel.deepseek_v41_rl_head import selected_log_probs

    generator = torch.Generator(device=device).manual_seed(711)
    hidden = (
        torch.randn(2, 5, width, device=device, generator=generator) / math.sqrt(width)
    ).requires_grad_()
    weight = torch.randn(31, width, device=device, generator=generator, requires_grad=True)
    labels = torch.randint(0, 31, (2, 5), device=device, generator=generator)
    labels[:, :2] = -100
    upstream = torch.randn(2, 5, device=device, generator=generator)
    actual = selected_log_probs(hidden, labels, weight, 3)
    expected = (
        F.linear(hidden, weight)
        .log_softmax(-1)
        .gather(-1, labels.clamp_min(0)[..., None])
        .squeeze(-1)
    )
    expected = expected.masked_fill(labels == -100, 0)
    first = torch.autograd.grad(actual, (hidden, weight), upstream)
    second = torch.autograd.grad(expected, (hidden, weight), upstream)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for observed, reference in zip(first, second, strict=True):
        torch.testing.assert_close(observed, reference, rtol=2e-5, atol=2e-6)
    return {
        "passed": True,
        "width": width,
        "vocabulary": 31,
        "signed_hidden_and_head_gradient_oracle": True,
        "output_max_abs": float((actual - expected).abs().max()),
        "gradient_max_abs": [
            float((a - b).abs().max()) for a, b in zip(first, second, strict=True)
        ],
    }


def save_qualification_rollout(rollout, output, *, rank, replay):
    """Save only small rollout tensors before replay, so a failure is diagnosable.

    No model weights, gradients, or optimizer state are copied or offloaded.
    """
    import torch

    path = Path(output) / f"rank-{rank:02d}-qualification-rollout.pt"
    if path.exists():
        raise FileExistsError("Refusing to replace a previous qualification rollout")
    tensors = {
        name: getattr(rollout, name).detach().cpu().clone()
        for name in (
            "input_ids",
            "attention_mask",
            "labels",
            "response_mask",
            "policy_log_probs",
            "behavior_log_probs",
        )
    }
    payload = {
        "format": "archlab-rl-qualification-rollout-v1",
        "rank": rank,
        "replay": replay,
        "receipt": rollout.receipt,
        "generated_ids": rollout.generated_ids,
        "prompt_lengths": rollout.prompt_lengths,
        "finish_reasons": rollout.finish_reasons,
        "audit_only": True,
        "optimizer_updates_requested": False,
        **tensors,
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    result = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rank": rank,
        "replay": replay,
        "policy_version": rollout.receipt["policy_version"],
        "contains_model_weights": False,
        "optimizer_updates_requested": False,
    }
    atomic_write_json(path.with_suffix(".json"), result, allow_nan=False)
    return result


def run_qualification(
    model, optimizer, indexers, encoded, tokenizer, config, contract, output, stops, pad
):
    import torch
    import torch.distributed as dist

    from archlab.automodel.deepseek_v41_rl_update import policy_gradient_step
    from archlab.rl.evaluation import evaluate_policy
    from archlab.rl.rollout import sample_rollouts

    rank, world = dist.get_rank(), dist.get_world_size()
    state = _snapshot_rng()
    try:
        leaf = qualify_leaf_head(model.lm_head.weight.device, model.lm_head.weight.shape[1])
        groups, prompts = rank_groups(
            encoded["train"],
            0,
            rank=rank,
            world=world,
            prompts_per_rank=config["prompts_per_rank"],
            group_size=config["group_size"],
        )
        version = digest(contract) + ":numerical-qualification"
        model._archlab_rl_policy_version = version
        memory_admission = None
        if config.get("evaluation_reserve_gib", 0):
            from archlab.automodel.deepseek_v41_rl_memory import qualify_replay_memory

            memory_admission = qualify_replay_memory(
                model,
                optimizer,
                indexers,
                prompts,
                config=config,
                policy_version=version,
                pad=pad,
            )
            if rank == 0:
                atomic_write_json(
                    output / "MEMORY_QUALIFICATION.json", memory_admission, allow_nan=False
                )
            if not memory_admission["passed"]:
                raise RuntimeError(
                    "maximum-context replay did not preserve the declared evaluation headroom"
                )
        cache_equivalence = None
        if config.get("cache_policy", False):
            from archlab.automodel.deepseek_v41_rl_cache import qualify_resident_cache

            cache_equivalence = qualify_resident_cache(
                model,
                prompts,
                pad_token_id=pad,
                context_limit=config["context_limit"],
                tolerance=config["replay_tolerance"],
                reserve_gib=config["weight_reserve_gib"],
            )
            atomic_write_json(
                output / f"rank-{rank:02d}-cache-equivalence.json",
                cache_equivalence,
                allow_nan=False,
            )
            if not cache_equivalence["passed"]:
                raise RuntimeError(
                    "resident cache disagrees with the actual actor's padded full-prefix policy"
                )
        if config.get("profile_rollouts", False):
            from archlab.rl.profiling import profile_rollout_batches

            if rank == 0:
                print(
                    json.dumps({"event": "rl_rollout_profile_start", "batch_sizes": [1, 4]}),
                    flush=True,
                )
            profile_prompt = gather(groups[0]["prompt_ids"])[0]
            profile = profile_rollout_batches(
                model,
                profile_prompt,
                policy_version=version,
                max_new_tokens=config.get("profile_new_tokens", 4),
                context_limit=config["context_limit"],
                eos_token_ids=stops,
                pad_token_id=pad,
                seed=config["seed"] + 2000000,
                batch_sizes=(1, 4),
                warmup=1,
                repeats=1,
            )
            if rank == 0:
                atomic_write_json(output / "PROFILE.json", profile, allow_nan=False)
        rollout = sample_rollouts(
            model,
            prompts,
            policy_version=version,
            max_new_tokens=min(config["qualification_max_new_tokens"], config["max_new_tokens"]),
            context_limit=config["context_limit"],
            eos_token_ids=stops,
            pad_token_id=pad,
            seed=config["seed"] + 1000000,
            **({"cache_policy": True} if config.get("cache_policy", False) else {}),
            prompt_group_ids=[
                row["problem_id"] for row in groups for _ in range(config["group_size"])
            ],
        )
        replay = replay_options(config, rollout_step=0, world=world, qualification=True)
        rollout_artifact = save_qualification_rollout(rollout, output, rank=rank, replay=replay)
        synthetic = torch.zeros(
            (len(groups), config["group_size"]), device=model.lm_head.weight.device
        )
        synthetic[:, 0] = 1
        try:
            audit = policy_gradient_step(
                model,
                optimizer,
                indexers,
                rollout,
                synthetic,
                lr=config["learning_rate"],
                group_size=config["group_size"],
                head_chunk_size=config["head_chunk_size"],
                replay_tolerance=config["replay_tolerance"],
                audit=True,
                audit_only=True,
                **replay,
            )
        except BaseException as error:
            atomic_write_json(
                output / f"rank-{rank:02d}-qualification-replay-failure.json",
                {
                    "passed": False,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "rollout_artifact": rollout_artifact,
                    "replay": replay,
                    "optimizer_updates_requested": False,
                    "optimizer_state_nonempty": any(optimizer.state.values()),
                },
                allow_nan=False,
            )
            raise
        if (
            audit.get("updated")
            or any(optimizer.state.values())
            or audit.get("numerical_qualification_passed") is not True
            or audit.get("replay_verified") is not True
            or audit.get("gradient_norm_before_clip", 0) <= 0
        ):
            raise RuntimeError(
                "Numerical qualification must never apply synthetic optimizer updates"
            )
        if config.get("qualification_evaluation", True):
            evaluation, records = evaluate_policy(
                model,
                tokenizer,
                encoded["heldout"],
                policy_version=version,
                max_new_tokens=config["max_new_tokens"],
                context_limit=config["context_limit"],
                eos_token_ids=stops,
                pad_token_id=pad,
                eval_count=config["qualification_eval_count"],
                local_batch_size=min(
                    config["eval_local_batch_size"],
                    max(1, math.ceil(config["qualification_eval_count"] / world)),
                ),
                seed=config["seed"],
                **({"cache_policy": True} if config.get("cache_policy", False) else {}),
            )
        else:
            evaluation, records = (
                {"skipped": True, "reason": "evaluation-deferred-until-after-training-pilot"},
                [],
            )
        local = {
            "rank": rank,
            "head_oracle": leaf,
            "policy_gradient_audit": audit,
            "rollout_artifact": rollout_artifact,
        }
        if cache_equivalence is not None:
            local["cache_equivalence"] = cache_equivalence
        atomic_write_json(output / f"rank-{rank:02d}-qualification.json", local, allow_nan=False)
        receipts = gather(local)
        receipt = {
            "passed": True,
            "kind": "online-policy-numerical-v1",
            "contract_digest": digest(contract),
            "contract": contract,
            "synthetic_optimizer_updates": 0,
            "real_reward_update_proven": False,
            "component_qualification": component_gate(config, contract["runtime"]),
            "full_checkpoint_roundtrip_executed": False,
            "checkpoint_evidence": "existing full-state writer/reader; bounded critical-state readback at first pilot checkpoint",
            "ranks": receipts,
            "heldout_evaluation": evaluation,
        }
        if memory_admission is not None:
            receipt["memory_admission"] = memory_admission
        if rank == 0:
            atomic_write_json(output / "QUALIFIED.json", receipt, allow_nan=False)
            atomic_write_json(
                output / "qualification-evaluation-records.json", records, allow_nan=False
            )
        return receipt
    finally:
        optimizer.zero_grad(set_to_none=True)
        _restore_rng(state)


def _python_rng_state():
    import numpy as np

    name, keys, pos, gaussian, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": [name, keys.tolist(), pos, gaussian, cached]}


def _restore_python_rng(payload):
    import numpy as np

    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, list) else value

    random.setstate(tuples(payload["python"]))
    name, keys, pos, gaussian, cached = payload["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))


def save_rl_checkpoint(path, model, optimizer, cursor, contract):
    import torch.distributed as dist

    from archlab.automodel.deepseek_v41_full_checkpoint import save_full_checkpoint

    rank = dist.get_rank()
    path = Path(path)
    if (path / "COMPLETE.json").exists():
        saved = json.loads((path / "COMPLETE.json").read_text())
        if saved["contract"] != contract or any(
            saved["cursor"].get(k) != v for k, v in cursor.items()
        ):
            raise ValueError("Refusing a checkpoint path containing different state")
        return
    sidecar = path / f"rl-rank-{rank:02d}-rng.json"
    atomic_write_json(sidecar, _python_rng_state(), allow_nan=False)
    checksums = gather({"file": sidecar.name, "sha256": sha256_file(sidecar)})
    saved_cursor = {**cursor, "rl_rng_sidecars": checksums}
    save_full_checkpoint(path, model, optimizer, saved_cursor, contract)
    if cursor["rollout_step"] == contract["recipe"]["pilot_updates"]:
        verify_critical_checkpoint(path, model, optimizer, saved_cursor, contract)


def verify_critical_checkpoint(path, model, optimizer, cursor, contract):
    """Read back the head, one adapter matrix, their optimizer states, and RNG.

    This is bounded pilot checkpoint validation, not a full-model restore claim.
    The writer has already checksummed every weight payload in its manifests.
    """
    import torch
    import torch.distributed as dist

    from archlab.optimizers.sharded_adafactor import local_tensor

    rank = dist.get_rank()
    marker = json.loads((Path(path) / "COMPLETE.json").read_text())
    if marker["contract"] != contract or marker["cursor"] != cursor:
        raise ValueError("Published pilot checkpoint cursor or contract differs")
    folder = Path(path) / f"rank-{rank:02d}"
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    named = dict(model.named_parameters())
    entries = {row["name"]: row for row in manifest["tensors"]}
    selected = [name for name in named if name.endswith("lm_head.weight")]
    adapter = next(
        (name for name in named if "simplicial_adapter" in name and name.endswith("output.weight")),
        None,
    )
    if not selected or adapter is None:
        raise ValueError("Pilot checkpoint lacks the trained head or adapter output")
    selected.append(adapter)
    optimizer_indices = {id(p): i for i, p in enumerate(optimizer.param_groups[0]["params"])}
    for name in selected:
        live = local_tensor(named[name]).detach().reshape(-1)
        offset = 0
        for chunk in entries[name]["chunks"]:
            saved = torch.load(folder / chunk["file"], map_location="cpu", weights_only=True)
            actual = hashlib.sha256(
                saved.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            if actual != chunk["sha256"] or not torch.equal(
                saved, live[offset : offset + saved.numel()].cpu()
            ):
                raise ValueError(f"Pilot checkpoint head/adapter readback differs: {name}")
            offset += saved.numel()
        if offset != live.numel():
            raise ValueError("Pilot critical tensor coverage differs")
        index = optimizer_indices[id(named[name])]
        saved_state = torch.load(
            folder / manifest["optimizer_states"][index], map_location="cpu", weights_only=True
        )
        live_state = optimizer.state[named[name]]
        if saved_state.keys() != live_state.keys():
            raise ValueError("Pilot critical optimizer fields differ")
        for key, value in saved_state.items():
            other = live_state[key]
            equal = (
                torch.equal(value, other.detach().cpu())
                if isinstance(value, torch.Tensor)
                else value == other
            )
            if not equal:
                raise ValueError("Pilot critical optimizer state differs")
    rng = torch.load(folder / "rng.pt", map_location="cpu", weights_only=True)
    if not torch.equal(rng["cpu_rng"], torch.get_rng_state()) or not torch.equal(
        rng["cuda_rng"], torch.cuda.get_rng_state()
    ):
        raise ValueError("Pilot checkpoint Torch RNG readback differs")
    local = {
        "passed": True,
        "rank": rank,
        "tensor_names": selected,
        "critical_weight_and_optimizer_readback": True,
        "torch_rng_exact": True,
        "full_model_roundtrip_executed": False,
    }
    receipts = gather(local)
    if rank == 0:
        atomic_write_json(
            Path(path) / "RL_CRITICAL_STATE_VERIFIED.json",
            {
                "passed": True,
                "ranks": receipts,
                "cursor": cursor,
                "scope": "bounded critical-state readback, not full restore",
            },
            allow_nan=False,
        )


def preflight_rl_checkpoint(path, contract, *, rank, world):
    """Reject mixed marker/rank metadata before the legacy reader mutates weights."""
    path = Path(path)
    marker = json.loads((path / "COMPLETE.json").read_text())
    if (
        marker.get("format") != "archlab-v41-full-sharded-v1"
        or marker.get("contract") != contract
        or marker.get("world_size") != world
        or marker.get("manifests") != [f"rank-{r:02d}/MANIFEST.json" for r in range(world)]
    ):
        raise ValueError("RL resume marker contract or mesh differs")
    manifest = json.loads((path / f"rank-{rank:02d}" / "MANIFEST.json").read_text())
    if manifest.get("rank") != rank or any(
        manifest.get(field) != marker[field] for field in ("world_size", "cursor", "contract")
    ):
        raise ValueError("RL resume rank manifest does not match its complete marker")
    sidecars = marker["cursor"].get("rl_rng_sidecars", [])
    if len(sidecars) != world or [entry["file"] for entry in sidecars] != [
        f"rl-rank-{r:02d}-rng.json" for r in range(world)
    ]:
        raise ValueError("RL resume RNG sidecar inventory differs")
    spec = sidecars[rank]
    if sha256_file(path / spec["file"]) != spec["sha256"]:
        raise ValueError("RL resume RNG sidecar changed")
    return marker


def restore_rl_checkpoint(path, model, optimizer, contract):
    import torch.distributed as dist

    from archlab.automodel.deepseek_v41_full_checkpoint import restore_full_checkpoint

    error = None
    try:
        preflight_rl_checkpoint(path, contract, rank=dist.get_rank(), world=dist.get_world_size())
    except (ValueError, KeyError, OSError) as caught:
        error = str(caught)
    errors = gather(error)
    if any(errors):
        raise ValueError(f"RL checkpoint preflight rejected: {errors}")
    cursor = restore_full_checkpoint(path, model, optimizer, contract)
    spec = cursor.pop("rl_rng_sidecars")[dist.get_rank()]
    sidecar = Path(path) / spec["file"]
    if (
        sidecar.name != f"rl-rank-{dist.get_rank():02d}-rng.json"
        or sha256_file(sidecar) != spec["sha256"]
    ):
        raise ValueError("RL checkpoint Python/NumPy RNG sidecar changed")
    _restore_python_rng(json.loads(sidecar.read_text()))
    return cursor


def initial_cursor():
    return {
        "step": 0,
        "phase_step": 0,
        "rollout_step": 0,
        "optimizer_step": 0,
        "prompt_cursor": 0,
        "generated_tokens": 0,
    }


def run_training_loop(
    model,
    optimizer,
    indexers,
    encoded,
    tokenizer,
    config,
    contract,
    output,
    *,
    mode,
    step_limit,
    cursor,
    stops,
    pad,
    stop_requested=lambda: False,
    sample_fn=None,
    update_fn=None,
    evaluate_fn=None,
    checkpoint_fn=None,
    activity_fn=lambda *_args, **_kwargs: None,
):
    """One rollout per update; injectable boundaries allow CPU control-flow tests."""
    import torch
    import torch.distributed as dist

    from archlab.rl.rewards import verify_math_answer

    if sample_fn is None:
        from archlab.rl.rollout import sample_rollouts as sample_fn
    if update_fn is None:
        from archlab.automodel.deepseek_v41_rl_update import policy_gradient_step as update_fn
    if evaluate_fn is None:
        from archlab.rl.evaluation import evaluate_policy as evaluate_fn
    if checkpoint_fn is None:
        checkpoint_fn = save_rl_checkpoint
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    device = model.lm_head.weight.device
    output = Path(output)
    contract_id = digest(contract)
    last_evaluated = None
    reason = "step_limit"

    def version():
        return f"{contract_id}:optimizer-update-{cursor['optimizer_step']}"

    def evaluate():
        nonlocal last_evaluated
        model._archlab_rl_policy_version = version()
        activity_fn("evaluating", **cursor, policy_version=version())
        summary, records = evaluate_fn(
            model,
            tokenizer,
            encoded["heldout"],
            policy_version=version(),
            max_new_tokens=config["max_new_tokens"],
            context_limit=config["context_limit"],
            eos_token_ids=stops,
            pad_token_id=pad,
            eval_count=config["eval_count"],
            local_batch_size=config["eval_local_batch_size"],
            seed=config["seed"],
            **({"cache_policy": True} if config.get("cache_policy", False) else {}),
        )
        summary.update(
            rollout_step=cursor["rollout_step"],
            optimizer_step=cursor["optimizer_step"],
            update_step=cursor["rollout_step"],
            step=cursor["rollout_step"],
            examples=summary["count"],
        )
        if rank == 0:
            append_jsonl(output / "evaluation.jsonl", summary)
            atomic_write_json(
                output / f"evaluation-step-{cursor['rollout_step']:06d}.json",
                records,
                allow_nan=False,
            )
        last_evaluated = cursor["rollout_step"]

    def checkpoint():
        activity_fn("checkpointing", **cursor)
        checkpoint_fn(
            output / "checkpoints" / f"step-{cursor['rollout_step']:06d}",
            model,
            optimizer,
            cursor,
            contract,
        )

    if config.get("initial_evaluation", True):
        evaluate()
    while cursor["rollout_step"] < step_limit:
        should_stop = any(gather(bool(stop_requested() or (output / "STOP_REQUEST").exists())))
        if should_stop:
            reason = "stop_request"
            break
        began = time.perf_counter()
        if config.get("gpu_memory_budget_gib") is not None and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        groups, prompts = rank_groups(
            encoded["train"],
            cursor["prompt_cursor"],
            rank=rank,
            world=world,
            prompts_per_rank=config["prompts_per_rank"],
            group_size=config["group_size"],
        )
        model._archlab_rl_policy_version = version()
        activity_fn("sampling", **cursor, policy_version=version())
        rollout = sample_fn(
            model,
            prompts,
            policy_version=version(),
            max_new_tokens=config["max_new_tokens"],
            context_limit=config["context_limit"],
            eos_token_ids=stops,
            pad_token_id=pad,
            seed=config["seed"] + cursor["rollout_step"] * world,
            temperature=1.0,
            top_p=1.0,
            prompt_group_ids=[
                row["problem_id"] for row in groups for _ in range(config["group_size"])
            ],
            **({"cache_policy": True} if config.get("cache_policy", False) else {}),
        )
        sampling_seconds = time.perf_counter() - began
        local_records, values = [], []
        for index, generated in enumerate(rollout.generated_ids):
            row = groups[index // config["group_size"]]
            ids = generated[:-1] if generated and generated[-1] in stops else generated
            completion = tokenizer.decode(ids, skip_special_tokens=False)
            reward = verify_math_answer(completion, row["expected_answer"])
            values.append(reward.reward)
            local_records.append(
                {
                    "problem_id": row["problem_id"],
                    "generated_ids": generated,
                    "completion": completion,
                    "verification": asdict(reward),
                    "finish_reason": rollout.finish_reasons[index],
                }
            )
        rewards = torch.tensor(values, device=device).reshape(len(groups), config["group_size"])
        length_regularization = {
            "eligible_groups": 0,
            "penalized_successes": 0,
            "total_deduction": 0.0,
        }
        if config.get("length_penalty", {}).get("enabled", False):
            from archlab.rl.regularization import successful_length_rewards

            lengths = torch.tensor(
                [len(row) for row in rollout.generated_ids], device=device
            ).reshape_as(rewards)
            rewards, length_regularization = successful_length_rewards(
                rewards, lengths, config["length_penalty"]
            )
        update_start = time.perf_counter()
        activity_fn("policy_gradient", **cursor, policy_version=version())
        metrics = update_fn(
            model,
            optimizer,
            indexers,
            rollout,
            rewards,
            lr=config["learning_rate"],
            group_size=config["group_size"],
            head_chunk_size=config["head_chunk_size"],
            replay_tolerance=config["replay_tolerance"],
            audit=cursor["optimizer_step"] < 2,
            **replay_options(config, rollout_step=cursor["rollout_step"], world=world),
        )
        update_seconds = time.perf_counter() - update_start
        count = sum(map(len, rollout.generated_ids))
        packet = {
            "sampling_seconds": sampling_seconds,
            "update_seconds": update_seconds,
            "iteration_seconds": time.perf_counter() - began,
            "generated_tokens": count,
            "correct": sum(values),
            "completions": len(values),
            "valid_answers": sum(
                r["verification"]["canonical_answer"] is not None for r in local_records
            ),
            "truncated": sum(r["finish_reason"] == "length" for r in local_records),
            "updated": bool(metrics.get("updated")),
        }
        packet.update(training_reward_sum=float(rewards.sum()), **length_regularization)
        if "mean_policy_entropy_nats" in rollout.receipt:
            packet.update(
                policy_entropy_sum=rollout.receipt["mean_policy_entropy_nats"] * count,
                eos_probability_sum=rollout.receipt["mean_eos_probability"] * count,
            )
        if config.get("gpu_memory_budget_gib") is not None and device.type == "cuda":
            packet.update(
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                gpu_driver_free_bytes=torch.cuda.mem_get_info(device)[0],
            )
        totals = gather(packet)
        if len({row["updated"] for row in totals}) != 1:
            raise RuntimeError("Ranks disagree on whether the policy updated")
        used_version = version()
        cursor["rollout_step"] += 1
        cursor["step"] = cursor["rollout_step"]
        cursor["optimizer_step"] += int(packet["updated"])
        cursor["phase_step"] = cursor["optimizer_step"]
        cursor["prompt_cursor"] += world * config["prompts_per_rank"]
        generated_count = sum(row["generated_tokens"] for row in totals)
        cursor["generated_tokens"] += generated_count
        model._archlab_rl_policy_version = version()
        timing = {
            key: max(row[key] for row in totals)
            for key in ("sampling_seconds", "update_seconds", "iteration_seconds")
        }
        completions = sum(row["completions"] for row in totals)
        metrics.update(
            **cursor,
            policy_version=used_version,
            next_policy_version=version(),
            update_step=cursor["rollout_step"],
            rollout_seconds=timing["sampling_seconds"],
            rollout_tokens=generated_count,
            rollout_tokens_per_second=generated_count / max(timing["sampling_seconds"], 1e-12),
            **timing,
            rollout_generated_tokens=generated_count,
            generated_tokens_per_second=generated_count / max(timing["iteration_seconds"], 1e-12),
            sampling_tokens_per_second=generated_count / max(timing["sampling_seconds"], 1e-12),
            reward_mean=sum(row["correct"] for row in totals) / completions,
            valid_answer_rate=sum(row["valid_answers"] for row in totals) / completions,
            truncation_rate=sum(row["truncated"] for row in totals) / completions,
            completions=completions,
            measured_mfu=None,
        )
        metrics.update(
            training_reward_mean=sum(row["training_reward_sum"] for row in totals) / completions,
            length_penalty_eligible_groups=sum(row["eligible_groups"] for row in totals),
            length_penalized_successes=sum(row["penalized_successes"] for row in totals),
            length_penalty_total_deduction=sum(row["total_deduction"] for row in totals),
        )
        if "gpu_peak_allocated_bytes" in packet:
            metrics.update(
                gpu_peak_allocated_gib=max(row["gpu_peak_allocated_bytes"] for row in totals)
                / 2**30,
                gpu_peak_reserved_gib=max(row["gpu_peak_reserved_bytes"] for row in totals) / 2**30,
                gpu_min_driver_free_gib=min(row["gpu_driver_free_bytes"] for row in totals) / 2**30,
            )
        if "policy_entropy_sum" in packet:
            metrics.update(
                mean_policy_entropy_nats=sum(row["policy_entropy_sum"] for row in totals)
                / generated_count,
                mean_eos_probability=sum(row["eos_probability_sum"] for row in totals)
                / generated_count,
            )
        append_jsonl(
            output / f"rank-{rank:02d}-rollouts.jsonl",
            {
                "rollout_step": cursor["rollout_step"],
                "receipt": rollout.receipt,
                "records": local_records,
            },
        )
        if rank == 0:
            append_jsonl(output / "rl-metrics.jsonl", metrics)
            if packet["updated"] and cursor["optimizer_step"] == 1:
                atomic_write_json(
                    output / "REAL_UPDATE_VERIFIED.json",
                    {"passed": True, **cursor, "real_outcome_rewards": True, "metrics": metrics},
                    allow_nan=False,
                )
            atomic_write_json(
                output / "PROGRESS.json",
                {**cursor, "policy_version": version(), "phase": "online_rl", "latest": metrics},
                allow_nan=False,
            )
        pilot_end = cursor["rollout_step"] == config["pilot_updates"]
        if pilot_end or cursor["rollout_step"] % config["checkpoint_interval_updates"] == 0:
            checkpoint()
        if pilot_end or cursor["rollout_step"] % config["eval_interval_updates"] == 0:
            evaluate()
        if pilot_end:
            if rank == 0:
                atomic_write_json(
                    output / "PILOT_RESULT.json",
                    {
                        **cursor,
                        "real_update_passed": cursor["optimizer_step"] > 0,
                        "numerical_admission_is_not_learning_evidence": True,
                    },
                    allow_nan=False,
                )
            if cursor["optimizer_step"] == 0:
                reason = "pilot_no_policy_update"
                break
    else:
        reason = (
            "budget_reached"
            if cursor["rollout_step"] == config["max_rollout_updates"]
            else "step_limit"
        )
    if last_evaluated != cursor["rollout_step"]:
        evaluate()
    checkpoint()
    report = {
        **cursor,
        "stop_reason": reason,
        "mode": mode,
        "policy_version": version(),
        "real_policy_updates": cursor["optimizer_step"],
        "budget": config["max_rollout_updates"],
    }
    if rank == 0:
        atomic_write_json(output / "STOPPED.json", report, allow_nan=False)
        if reason == "budget_reached" and cursor["optimizer_step"] > 0:
            atomic_write_json(output / "COMPLETE.json", {"passed": True, **report}, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--variant", choices=("normal", "simplicial"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("qualify", "pilot", "train"), default="pilot")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume-rl", type=Path)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    config = read_recipe(args.recipe)
    if args.steps is not None and not 0 < args.steps <= config["max_rollout_updates"]:
        parser.error("--steps is a positive cumulative cap within the sealed rollout budget")
    if args.resume_rl and not args.qualification:
        parser.error("RL resume requires its existing numerical --qualification receipt")
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    packages = select_container_kernel_packages(Path(config["container_kernel_packages"]))
    import numpy as np
    import torch
    import torch.distributed as dist
    import yaml
    from transformers import PreTrainedTokenizerFast

    from archlab.automodel.deepseek_v41_rl_head import install_rl_head
    from archlab.automodel.deepseek_v41_rl_model import EXECUTION_CHANGES, construct_rl_actor
    from archlab.optimizers.sharded_adafactor import ShardedAdafactor
    from archlab.preprocessing.deepseek_v41 import DeepSeekV41Renderer

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from archlab.automodel.deepseek_v41_rl_memory import (
        configure_gpu_budget,
        install_checkpoint_input_offload,
    )

    memory_budget = configure_gpu_budget(config["gpu_memory_budget_gib"])
    torch.set_num_threads(4)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=90),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    rank = dist.get_rank()
    requested_stop = [False]
    activity = None
    owns_output = False
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: requested_stop.__setitem__(0, True))
    try:
        if dist.get_world_size() != 16:
            raise ValueError("Pretrained RL requires exactly sixteen ranks per actor")
        if any(gather(args.output.exists())):
            raise FileExistsError("Use a fresh output directory; resume into a new directory")
        if rank == 0:
            args.output.mkdir(parents=True)
        dist.barrier()
        owns_output = True
        if rank == 0:
            activity = Activity(args.output)
            activity.set("verifying_data")
        splits, manifest = load_data(config)
        assets = Path(config["assets"])
        tokenizer = PreTrainedTokenizerFast.from_pretrained(assets, local_files_only=True)
        renderer = DeepSeekV41Renderer(assets)
        template = yaml.safe_load(Path(config["prompt_template"]).read_text())["templates"]["user"]
        encoded = {
            name: encode_examples(
                rows,
                tokenizer=tokenizer,
                renderer=renderer,
                user_template=template,
                context_limit=config["context_limit"],
                max_new_tokens=config["max_new_tokens"],
            )
            for name, rows in splits.items()
        }
        encoded["train"], ordering = length_bucket_order(
            encoded["train"], global_batch_size=16 * config["prompts_per_rank"], seed=config["seed"]
        )
        eos = tokenizer.encode(renderer.encoder.eos_token, add_special_tokens=False)
        if len(eos) != 1:
            raise ValueError("Native end-of-turn marker must be one token")
        stops = set(eos + config.get("extra_eos_token_ids", []))
        pad = 2 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        provenance = {
            "native_encoder_sha256": sha256_file(assets / "encoding/encoding.py"),
            "prompt_sha256": sha256_file(config["prompt_template"]),
            "tokenizer_files": {
                p.name: sha256_file(p) for p in assets.glob("tokenizer*") if p.is_file()
            },
            "thinking_mode": "chat",
            "eos_ids": sorted(stops),
            "pad_token_id": pad,
            "encoded_split_sha256": {name: digest(rows) for name, rows in encoded.items()},
            "training_prompt_order": ordering,
            "data_manifest_sha256": config["data_manifest_sha256"],
            "approved_data_format": manifest["format"],
        }
        random.seed(config["seed"] + rank)
        np.random.seed(config["seed"] + rank)
        torch.manual_seed(config["seed"] + rank)
        precision_before = configure_numerical_precision()
        parent = config["parents"][args.variant]
        if activity:
            activity.set("restoring_parent", checkpoint=parent["checkpoint"], variant=args.variant)
        model, indexers, loading, marker = construct_rl_actor(
            checkpoint=parent["checkpoint"],
            family="full",
            variant=args.variant,
            assets=assets,
            weights=Path(config["weights"]),
            declared_source_changes=parent.get("declared_source_changes", {}),
            declared_execution_changes=EXECUTION_CHANGES,
            resolved_kernel_packages=packages,
            expected_parent_marker_sha256=parent["marker_sha256"],
        )
        precision_after = configure_numerical_precision()
        loading["rl_numerical_precision"] = {
            "before_construction": precision_before,
            "after_construction": precision_after,
        }
        if marker["cursor"]["step"] != 4537 or marker["cursor"]["supervised_tokens"] != 756364650:
            raise ValueError("RL must start from the selected matched step4537 pair")
        install_rl_head(model)
        from archlab.automodel.deepseek_v41_rl_model import configure_rl_trainability

        loading["rl_trainability"] = configure_rl_trainability(
            model, freeze_router=config["freeze_router"]
        )
        from archlab.automodel.deepseek_v41_rl_memory_policy import (
            install_hc_activation_offload,
            install_inplace_moe_accumulation,
            serialize_backward_gathers,
            unshard_on_compute_stream,
        )

        loading["rl_memory_policy"] = {
            "allocator": memory_budget,
            "gather_allocation": (
                unshard_on_compute_stream(model)
                if config["unshard_on_compute_stream"]
                else {"enabled": False}
            ),
            "backward_gathers": (
                serialize_backward_gathers(model)
                if config["serialize_backward_gathers"]
                else {"enabled": False}
            ),
            "hc_activations": (
                install_hc_activation_offload(model)
                if config["hc_activation_offload"]
                else {"enabled": False}
            ),
            "moe_accumulation": (
                install_inplace_moe_accumulation(
                    model, checkpoint_activations=config["checkpoint_expert_activations"]
                )
                if config["inplace_moe_accumulation"]
                else {"enabled": False}
            ),
            "checkpoint_inputs": (
                install_checkpoint_input_offload(model)
                if config["checkpoint_input_offload"]
                else {"enabled": False}
            ),
        }
        model._archlab_rl_retain_weights = config["retain_weights"]
        model._archlab_rl_weight_reserve_gib = config["weight_reserve_gib"]
        component_gate(config, loading)
        optimizer = ShardedAdafactor(
            (p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"]
        )
        contract = make_contract(config, args.variant, loading, marker, provenance)
        if len(set(gather(digest(contract)))) != 1:
            raise ValueError("RL ranks disagree on the immutable experiment contract")
        atomic_write_json(args.output / f"rank-{rank:02d}-loading.json", loading, allow_nan=False)
        if rank == 0:
            atomic_write_json(args.output / "RUN_CONTRACT.json", contract, allow_nan=False)
            commit = subprocess.check_output(
                ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            atomic_write_json(
                args.output / "RUN_PROVENANCE.json",
                {
                    "project_commit": commit,
                    "recipe_path": str(args.recipe),
                    "mode": args.mode,
                    "cli_steps": args.steps,
                    "resume_rl": str(args.resume_rl) if args.resume_rl else None,
                },
                allow_nan=False,
            )
        if args.qualification:
            path = (
                args.qualification / "QUALIFIED.json"
                if args.qualification.is_dir()
                else args.qualification
            )
            qualification = json.loads(path.read_text())
            admit_qualification(qualification, contract)
            if rank == 0:
                atomic_write_json(args.output / "QUALIFIED.json", qualification, allow_nan=False)
        else:
            if activity:
                activity.set("qualifying_actual_actor")
            qualification = run_qualification(
                model,
                optimizer,
                indexers,
                encoded,
                tokenizer,
                config,
                contract,
                args.output,
                stops,
                pad,
            )
            admit_qualification(qualification, contract)
        if args.mode == "qualify":
            if activity:
                activity.set("qualified")
            return
        cursor = (
            restore_rl_checkpoint(args.resume_rl, model, optimizer, contract)
            if args.resume_rl
            else initial_cursor()
        )
        if cursor["prompt_cursor"] != cursor["rollout_step"] * 16 * config["prompts_per_rank"]:
            raise ValueError("RL resume cursor disagrees with fixed prompt order")
        if rank == 0:
            atomic_write_json(
                args.output / "TRAINING_ADMITTED.json",
                {
                    "passed": True,
                    "contract_digest": digest(contract),
                    "real_reward_update_still_required": cursor["optimizer_step"] == 0,
                },
                allow_nan=False,
            )
        limit = config["pilot_updates"] if args.mode == "pilot" else config["max_rollout_updates"]
        if args.steps is not None:
            limit = min(limit, args.steps)
        report = run_training_loop(
            model,
            optimizer,
            indexers,
            encoded,
            tokenizer,
            config,
            contract,
            args.output,
            mode=args.mode,
            step_limit=limit,
            cursor=cursor,
            stops=stops,
            pad=pad,
            stop_requested=lambda: requested_stop[0],
            activity_fn=activity.set if activity else lambda *_args, **_kwargs: None,
        )
        if activity:
            activity.set("stopped", **report)
    except BaseException:
        if activity:
            activity.set("failed")
        write_owned_failure(args.output, rank, owns_output, traceback.format_exc())
        raise
    finally:
        if activity:
            activity.close()
        dist.destroy_process_group()


def write_owned_failure(output, rank, owns_output, traceback_text):
    """A rejected preexisting output is never mutated by this invocation."""
    if owns_output:
        atomic_write_json(
            Path(output) / f"rank-{rank:02d}-failure.json", {"traceback": traceback_text}
        )


if __name__ == "__main__":
    main()
