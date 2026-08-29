"""
Utilities for saving and loading model/optim/state checkpoints.
"""

import glob
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from archlab.speedrun.models import (
    build_model_from_config_kwargs,
    patch_missing_model_state,
    patch_model_config_kwargs,
    strip_backend_extra_state,
)
from archlab.speedrun.runtime import get_base_dir, setup_default_logging
from archlab.speedrun.tokenizer import get_tokenizer

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)


def log0(message):
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(message)


def _atomic_torch_save(value: Any, path: Path) -> None:
    """Publish a torch artifact only after serialization completes."""

    temporary = path.with_suffix(path.suffix + f".tmp-rank{os.environ.get('RANK', '0')}")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def capture_rng_state() -> dict[str, Any]:
    """Capture every process-local RNG used by the speedrun backend."""

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains a CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state(state["torch_cuda"].cpu())


def load_rng_state(checkpoint_dir, step, device, rank=0):
    path = Path(checkpoint_dir) / f"rng_{step:06d}_rank{rank:d}.pt"
    if not path.is_file():
        return None
    return torch.load(path, map_location=device)


def save_checkpoint(
    checkpoint_dir,
    step,
    model_data,
    optimizer_data,
    meta_data,
    rank=0,
    rng_data=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        # Save the model state parameters
        model_path = checkpoint_dir / f"model_{step:06d}.pt"
        _atomic_torch_save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = checkpoint_dir / f"meta_{step:06d}.json"
        _atomic_json_save(meta_data, meta_path)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        optimizer_path = checkpoint_dir / f"optim_{step:06d}_rank{rank:d}.pt"
        _atomic_torch_save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")
    if rng_data is not None:
        rng_path = checkpoint_dir / f"rng_{step:06d}_rank{rank:d}.pt"
        _atomic_torch_save(rng_data, rng_path)
        logger.info(f"Saved RNG state to: {rng_path}")


def verify_checkpoint_bundle(
    checkpoint_dir: str | Path,
    step: int,
    *,
    world_size: int,
    require_optimizer: bool = True,
    require_rng: bool = False,
) -> dict[str, Any]:
    """Reject a partially published model/metadata/optimizer checkpoint bundle."""

    if world_size < 1:
        raise ValueError("world_size must be positive")
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    expected = [
        checkpoint_dir / f"model_{step:06d}.pt",
        checkpoint_dir / f"meta_{step:06d}.json",
    ]
    if require_optimizer:
        expected.extend(
            checkpoint_dir / f"optim_{step:06d}_rank{rank:d}.pt" for rank in range(world_size)
        )
    if require_rng:
        expected.extend(
            checkpoint_dir / f"rng_{step:06d}_rank{rank:d}.pt" for rank in range(world_size)
        )
    missing = [str(path) for path in expected if not path.is_file()]
    empty = [str(path) for path in expected if path.is_file() and path.stat().st_size == 0]
    if missing or empty:
        raise RuntimeError(f"incomplete checkpoint step {step}: missing={missing} empty={empty}")
    meta_path = checkpoint_dir / f"meta_{step:06d}.json"
    with meta_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if int(metadata.get("step", -1)) != step:
        raise RuntimeError(f"checkpoint metadata step mismatch: {metadata.get('step')} != {step}")
    return {
        "step": step,
        "world_size": world_size,
        "optimizer_shards": world_size if require_optimizer else 0,
        "rng_shards": world_size if require_rng else 0,
        "files": [{"path": path.name, "bytes": path.stat().st_size} for path in expected],
    }


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, step, device, load_optimizer=False
    )
    model_data = strip_backend_extra_state(model_data)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = patch_model_config_kwargs(meta_data["model_config"])
    log0(f"Building model with config: {model_config_kwargs}")
    with torch.device("meta"):
        model, model_config = build_model_from_config_kwargs(
            model_config_kwargs, runtime_backend="native"
        )
    model_data = patch_missing_model_state(model_data, model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights()  # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], (
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    )
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [
        f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))
    ]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step


# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure


def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data


def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)


def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data
