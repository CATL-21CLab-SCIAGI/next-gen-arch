"""Rank-local RNG and trainer clocks for resumable native Limite GRPO."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def resize_rank_state(state, world_size):
    """Explicitly resize a resume, preserving moments/clocks but not data order.

    The original checkpoint remains untouched. Prefetched responses belonged to
    the old rank partition and must never be replayed under a new partition.
    Keep surviving learner RNGs and rewind their actor RNG before that prefetch.
    """
    previous = state["world_size"]
    if world_size == previous:
        return state
    if not 0 < world_size < previous or previous % world_size:
        raise ValueError("rank-state resize requires an evenly divisible smaller world")
    if len(state["rank_rng"]) != previous:
        raise ValueError("checkpoint is missing rank RNG states")
    resized = dict(state, world_size=world_size, rank_rng=copy.deepcopy(state["rank_rng"][:world_size]),
                   evidence=copy.deepcopy(state["evidence"]))
    discarded = sum(bool((row.get("rollout") or {}).get("pending")) for row in state["rank_rng"])
    for row in resized["rank_rng"]:
        rollout = row.get("rollout")
        if rollout and rollout.get("pending") is not None:
            rollout["generator"] = rollout["pending"]["rng_before"].clone()
            rollout["pending"] = None
    resized["evidence"]["topology_migration"] = dict(
        previous_world_size=previous, world_size=world_size,
        discarded_prefetched_batches=discarded, exact_distributed_rng_resume=False,
        optimizer_scheduler_retained=True, learner_rank_rng="surviving ranks",
        actor_rng="rewound before discarded prefetch", data_order="new rank partition",
    )
    return resized


def capture_rng():
    numpy = np.random.get_state()
    result = dict(
        python=random.getstate(),
        numpy=(numpy[0], numpy[1].tolist(), *numpy[2:]),
        cpu=torch.get_rng_state(),
    )
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state(torch.cuda.current_device())
    return result


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], torch.cuda.current_device())


def checkpoint_payloads(trainer, evidence):
    actor = getattr(trainer, "archlab_async_rollout", None)
    rollout = actor.checkpoint_state() if actor is not None else None
    local_state = capture_rng()
    if actor is not None:
        local_state["rollout"] = rollout
    ranks = [local_state]
    if torch.distributed.is_initialized():
        rendezvous = getattr(trainer, "archlab_rollout_rendezvous", None)
        if rendezvous is not None:
            # The actor's prefetched batch makes this object larger than text
            # logs. Reuse the host group and retain one complete object per rank.
            ranks = rendezvous.gather_rank_objects(local_state)
        else:
            ranks = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ranks, local_state)
    return {
        "trainer_state.json": asdict(trainer.state),
        "rl_state.pt": dict(
            world_size=len(ranks), rank_rng=ranks, evidence=dict(evidence),
            scheduler=trainer.lr_scheduler.state_dict(),
        ),
    }


def legacy_resume_state(checkpoint, output, *, max_steps, train_batch_size=1, seed=42):
    """Retain legacy weights/moments and recover clocks without claiming exact RNG.

    Old saves contain rank zero's Torch RNG only. Preserve that state, explicitly
    reseed the other ranks, and require complete rank states for future resumes.
    Canonical checkpoint files remain immutable; only the small resume view is new.
    """
    from transformers import TrainerState

    checkpoint, output = Path(checkpoint), Path(output)
    receipt = json.loads((checkpoint / "COMPLETE.json").read_text())
    step = receipt["step"]
    updates = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    update = next((row for row in reversed(updates) if row["step"] == step), None)
    if update is None:
        raise ValueError("legacy RL checkpoint has no corresponding update evidence")
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    history = [row for row in metrics if row["step"] <= step]
    epoch = next((row["epoch"] for row in reversed(history) if "loss" in row), 0.0)
    num_tokens = int(next((row["num_tokens"] for row in reversed(history) if "num_tokens" in row), 0))
    view = output / "resume-state" / "trainer-v1" / checkpoint.name
    if view.exists():
        migration = json.loads((view / "LEGACY_RESUME.json").read_text())
        if (
            migration["checkpoint"] != str(checkpoint)
            or migration["step"] != step
            or migration["seed"] != seed
            or migration["max_steps"] != max_steps
            or migration["train_batch_size"] != train_batch_size
            or json.loads((view / "COMPLETE.json").read_text()) != receipt
        ):
            raise ValueError("existing legacy resume view differs from this checkpoint")
        return view, migration
    view.mkdir(parents=True, exist_ok=False)
    for path in checkpoint.iterdir():
        if path.is_file():
            (view / path.name).symlink_to(path.resolve())
    state = TrainerState(
        global_step=step, epoch=epoch, max_steps=max_steps, log_history=history,
        train_batch_size=train_batch_size, logging_steps=1, eval_steps=100, save_steps=500,
        num_input_tokens_seen=num_tokens,
    )
    state.save_to_json(str(view / "trainer_state.json"))
    migration = dict(
        checkpoint=str(checkpoint), step=step,
        evidence={key: value for key, value in update.items() if key not in ("time", "step")},
        rank_zero_torch_rng_retained=True, other_rank_rng="explicit_reseed",
        other_rank_seed="seed + process_index", seed=seed, max_steps=max_steps,
        train_batch_size=train_batch_size, trainer_state_schema=1,
        exact_distributed_rng_resume=False,
    )
    (view / "LEGACY_RESUME.json").write_text(json.dumps(migration, indent=2))
    return view, migration
