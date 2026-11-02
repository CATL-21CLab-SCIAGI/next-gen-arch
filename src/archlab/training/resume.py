"""Continuation of an owned run using the pinned verl checkpoint contract."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from archlab.artifacts import atomic_write_json


def claim_run(root: Path, identity: dict, *, resume: Path | None = None) -> None:
    """Called while holding the run's process lock; never adopt an unrelated run."""
    marker = root / "archlab-run.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("checkpoint directory belongs to another qualified experiment")
    elif resume is not None or any(root.glob("global_step_*")):
        raise ValueError("cannot adopt checkpoints without their qualified run identity")
    else:
        atomic_write_json(marker, identity)
    if resume is None and any(root.glob("global_step_*")):
        raise ValueError("existing checkpoints require an explicit continuation")


def resumed_config(config, checkpoint: Path):
    """Preserve every scientific setting and all model/optimizer/RNG/loader state.

    Asynchronous trajectory-queue restart needs its own admission evidence; the
    existing learner-only restart probe cannot establish that property.
    """
    root = Path(config.trainer.default_local_dir).resolve()
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or checkpoint.resolve().parent != root:
        raise ValueError("resume must use a checkpoint inside this run")
    checkpoint = checkpoint.resolve()
    step = int((root / "latest_checkpointed_iteration.txt").read_text())
    if step <= 0 or checkpoint.name != f"global_step_{step}":
        raise ValueError("resume requires the latest committed checkpoint")
    if OmegaConf.select(config, "trainer.v1.trainer_mode") != "sync":
        raise ValueError("asynchronous queue resume has no deployment admission evidence yet")
    if OmegaConf.select(config, "actor_rollout_ref.actor.checkpoint.async_save", default=False):
        raise ValueError("resume requires synchronous committed checkpoint saves")
    if not (checkpoint / "data.pt").is_file():
        raise ValueError("resume is missing dataloader state")
    if any(path.is_symlink() for path in checkpoint.rglob("*")):
        raise ValueError("resume checkpoint cannot alias another run")
    actor = checkpoint / "actor"
    strategy = config.actor_rollout_ref.actor.strategy
    if strategy == "megatron":
        receipt = json.loads((actor / "ckpt_contents.json").read_text())
        if receipt["global_step"] != step:
            raise ValueError("actor checkpoint step differs from the committed step")
        for key in ("model", "optimizer", "lr_scheduler", "rng_state"):
            path = (actor / receipt["contents"][key]["path"]).resolve()
            if not path.is_relative_to(actor) or not path.exists():
                raise ValueError("checkpoint is missing contained " + key)
    elif strategy == "fsdp2":
        world = config.trainer.nnodes * config.trainer.n_gpus_per_node
        for rank in range(world):
            for kind in ("model", "optim", "extra_state"):
                if not (actor / f"{kind}_world_size_{world}_rank_{rank}.pt").is_file():
                    raise ValueError(f"checkpoint is missing {kind} rank {rank}")
    else:
        raise ValueError("unqualified resume checkpoint strategy: " + strategy)
    result = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    result.trainer.resume_mode = "resume_path"
    result.trainer.resume_from_path = str(checkpoint)
    return result
