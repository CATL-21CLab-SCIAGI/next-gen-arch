"""Native publisher actors in the shared Limite RL execution engine.

These checkpoints contain the complete unchanged model topology. They share
verified staging and asynchronous publication with adapter runs, without an
adapter payload or a synthetic warmup parent.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from archlab.architectures.limite_loader import load_model
from archlab.automodel.limite_adapter_common import (
    _freeze_checkpoint_state,
    _write_checkpoint,
    file_hash,
)


def publisher_identity(snapshot):
    snapshot = Path(snapshot)
    receipt = json.loads((snapshot / "DOWNLOAD_VERIFIED.json").read_text())
    if receipt["repo"] not in {
        "paradigma-inc/limite-1b-base", "paradigma-inc/limite-1b-violetto"
    }:
        raise ValueError("native RL requires a verified Limite publisher actor")
    for spec in receipt["files"]:
        relative = Path(spec["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("publisher receipt contains an unsafe payload path")
        if file_hash(snapshot / relative) != spec["sha256"]:
            raise ValueError("publisher snapshot checksum mismatch: " + str(relative))
    return dict(
        repo=receipt["repo"], revision=receipt["revision"],
        receipt_sha256=file_hash(snapshot / "DOWNLOAD_VERIFIED.json"),
    )


def check_native_options(variant, warmup, attention_backend, trainable_mode, normal_kernel,
                         runtime_sequence_attention, allow_backend_migration):
    if variant != "native":
        return
    if warmup is not None:
        raise ValueError("native publisher RL does not have an adapter warmup parent")
    if attention_backend not in (None, "native") or normal_kernel not in (None, "shared"):
        raise ValueError("native publisher RL retains its publisher training attention")
    if trainable_mode not in (None, "full"):
        raise ValueError("native publisher RL trains all weights")
    if runtime_sequence_attention or allow_backend_migration:
        raise ValueError("adapter kernel migration does not apply to native publisher RL")


def check_native_recipe(spec, world, steps, *, correctness_fixture=False):
    if spec is None or spec.get("model", {}).get("variant") != "native":
        raise ValueError("native publisher RL requires its explicit math protocol recipe")
    execution, training = spec["execution"], spec["training"]
    responses = execution["responses_per_update"]
    if responses != training["responses_per_update"] or responses <= 0 or responses % world or responses % 4:
        raise ValueError("native RL response budget must divide ranks and four-sample groups")
    if not correctness_fixture and (world != training["world_size"] or steps != training["max_steps"]):
        raise ValueError("native RL world size and training budget differ from the recipe")
    if training["per_device_train_batch_size"] != 1 or training["learning_rate"] != 1e-5:
        raise ValueError("native RL recipe differs from the shared qualified optimizer contract")
    if training["gradient_accumulation_steps"] * training["world_size"] != responses:
        raise ValueError("native RL recipe has an inconsistent accumulation budget")
    if execution.get("async_rollouts") and (
        execution.get("overlap_actor_learner") is not False
        or execution.get("rollout_rendezvous") != "gloo_after_drain"
        or execution.get("max_policy_lag") != 1
    ):
        raise ValueError("native async RL requires the qualified drain, host rendezvous and lag contract")


def check_native_publisher(snapshot, tokenizer, spec, identity):
    if any(identity[key] != spec["model"][key] for key in ("repo", "revision")):
        raise ValueError("native publisher snapshot differs from the experiment recipe")
    if Path(tokenizer).resolve() != Path(snapshot).resolve():
        raise ValueError("native RL tokenizer must come from the verified publisher snapshot")


def native_identity_contract(model, receipt):
    if not getattr(model, "archlab_native_checkpoint", False) or hasattr(model.model, "adapters"):
        raise ValueError("native publisher RL must not contain inserted adapters")
    identity = model.archlab_publisher_identity
    if receipt["publisher_identity"] != identity:
        raise ValueError("native checkpoint belongs to a different publisher snapshot")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("full-weight RL requires every model parameter to be trainable")
    return dict(model_kind="native", trainable_mode="full", publisher_identity=identity)


def build_native_model(snapshot, device, checkpoint=None, *, checkpoint_cache=None):
    identity = publisher_identity(snapshot)
    model = load_model(snapshot, attn_implementation="sdpa", device_map=device)
    if hasattr(model.model, "adapters"):
        raise ValueError("publisher actor unexpectedly contains adapters")
    model.requires_grad_(True)
    model.model.trainable_mode = "full"
    model.archlab_native_checkpoint = True
    model.archlab_publisher_identity = identity
    if checkpoint is not None:
        from archlab.automodel.checkpoint_cache import stage_checkpoint

        checkpoint = Path(checkpoint)
        if checkpoint_cache is not None:
            checkpoint = stage_checkpoint(checkpoint, checkpoint_cache)
        receipt = json.loads((checkpoint / "COMPLETE.json").read_text())
        if receipt.get("model_kind") != "native" or receipt.get("trainable_mode") != "full":
            raise ValueError("native RL resume requires a native full-weight checkpoint")
        native_identity_contract(model, receipt)
        if "model.pt" not in receipt["files"] or "adapter.pt" in receipt["files"]:
            raise ValueError("native checkpoint has an invalid model payload")
        if checkpoint_cache is None:
            for name, checksum in receipt["files"].items():
                if Path(name).name != name or file_hash(checkpoint / name) != checksum:
                    raise ValueError("native checkpoint checksum mismatch: " + name)
        state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
        if set(state) != set(model.state_dict()) or any(
            state[name].dtype != value.dtype or state[name].shape != value.shape
            for name, value in model.state_dict().items()
        ):
            raise ValueError("native checkpoint model topology or parameter dtypes differ")
        model.load_state_dict(state, strict=True)
    return model


def save_native_checkpoint(model, optimizer, root, step, tokens, oss=None, extra=None,
                           extra_payloads=None, publisher=None):
    identity = native_identity_contract(model, {"publisher_identity": model.archlab_publisher_identity})
    if publisher is not None:
        publisher.wait()
    dest = Path(root) / "checkpoints" / f"step-{step:07d}"
    dest.mkdir(parents=True, exist_ok=False)
    payloads = dict(
        **{"model.pt": model.state_dict(), "optimizer.pt": optimizer.state_dict()},
        **{"rng.pt": dict(cpu=torch.get_rng_state(),
                          cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)},
    )
    for name, value in (extra_payloads or {}).items():
        if name in payloads:
            raise ValueError("checkpoint payload must have a unique leaf filename")
        payloads[name] = value
    manifest = {"step": step, "tokens": tokens, **(extra or {}), **identity}
    if publisher is None:
        _write_checkpoint(dest, root, oss, manifest, payloads)
    else:
        publisher.submit(_write_checkpoint, dest, root, oss, manifest, _freeze_checkpoint_state(payloads))
    return dest
