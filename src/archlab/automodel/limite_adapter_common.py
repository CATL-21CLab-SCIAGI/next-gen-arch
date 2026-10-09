"""Shared model, loss and checkpoint contract for Limite adapter SFT and RL."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from archlab.architectures.limite_adapter import (
    LimiteAdapterConfig,
    adapter_state,
    backbone_state,
    install_adapters,
    set_normal_attention_backward,
    set_normal_attention_kernel,
    set_trainable_mode,
)
from archlab.architectures.limite_loader import load_model
from archlab.artifacts import sha256_file as file_hash


def runtime_contract():
    """Record the container-owned runtime without changing any dependency."""
    versions = {}
    for package in (
        "torch",
        "transformers",
        "triton",
        "transformer-engine",
        "accelerate",
        "trl",
        "mlflow",
        "flash-attn",
        "flash-attn-4",
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-core",
        "quack-kernels",
        "megatron-core",
        "tilelang",
        "apache-tvm-ffi",
        "torch-c-dlpack-ext",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    revision = os.environ.get("ARCHLAB_SOURCE_REVISION")
    if revision is None:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    resolved_modules = {}
    for name in ("torch", "transformers", "triton", "trl", "tilelang", "tvm_ffi"):
        module = sys.modules.get(name)
        if module is not None:
            resolved_modules[name] = dict(
                version=getattr(module, "__version__", None),
                path=getattr(module, "__file__", None),
            )
        elif name in ("tilelang", "tvm_ffi"):
            spec = importlib.util.find_spec(name)
            if spec is not None:
                package = "apache-tvm-ffi" if name == "tvm_ffi" else name
                resolved_modules[name] = dict(version=versions[package], path=spec.origin)
    return dict(
        packages=versions,
        resolved_modules=resolved_modules,
        python=platform.python_version(),
        cuda=torch.version.cuda,
        nccl=torch.cuda.nccl.version(),
        device="NVIDIA B300 (NVML label is incorrect)",
        container={
            key: os.environ.get(key)
            for key in (
                "ARCHLAB_CONTAINER_IMAGE",
                "NVIDIA_PRODUCT_NAME",
                "NVIDIA_BUILD_ID",
                "NVIDIA_PYTORCH_VERSION",
                "NEMO_VERSION",
                "CUDA_VERSION",
            )
        },
        source_revision=revision,
    )


def attention_kernel_name(
    variant, attention_backend, *, normal_kernel="shared", normal_backward="tilelang"
):
    if attention_backend == "tilelang":
        if variant == "normal" and normal_kernel == "gqa":
            if normal_backward == "fa4":
                return "tilelang-forward-fa4-native-bf16-backward-v3"
            return "tilelang-gqa-key-owned-bf16-v1"
        return "tilelang-joint-bf16-v2" if variant == "simplicial" else "tilelang-gqa-bf16-v1"
    return "packed-compensated-bf16-pipelined-v4" if variant == "simplicial" else "publisher-sdpa"


def build_model(
    snapshot,
    variant,
    device,
    adapter_checkpoint=None,
    *,
    attention_backend=None,
    trainable_mode=None,
    allow_backend_migration=False,
    normal_kernel=None,
    normal_backward=None,
    checkpoint_cache=None,
):
    receipt = checkpoint_adapter = None
    if adapter_checkpoint and checkpoint_cache is not None:
        from archlab.automodel.checkpoint_cache import stage_checkpoint

        adapter_checkpoint = stage_checkpoint(adapter_checkpoint, checkpoint_cache)
    if adapter_checkpoint:
        receipt = json.loads((Path(adapter_checkpoint) / "COMPLETE.json").read_text())
        checkpoint_adapter = dict(receipt["adapter"])
        # Receipts written before backend selection describe the native path.
        checkpoint_adapter.setdefault("attention_backend", "native")
    if attention_backend is None:
        attention_backend = checkpoint_adapter["attention_backend"] if receipt else "native"
    if normal_kernel is None:
        normal_kernel = receipt.get("normal_kernel", "shared") if receipt else "shared"
    if normal_kernel not in ("shared", "gqa"):
        raise ValueError("invalid normal attention kernel")
    if normal_kernel == "gqa" and (variant != "normal" or attention_backend != "tilelang"):
        raise ValueError("gqa kernels require the normal TileLang variant")
    if normal_backward is None:
        normal_backward = receipt.get("normal_backward", "tilelang") if receipt else "tilelang"
    if normal_backward not in ("tilelang", "fa4"):
        raise ValueError("invalid normal attention backward")
    if normal_backward == "fa4" and normal_kernel != "gqa":
        raise ValueError("FA4 backward requires the normal TileLang GQA configuration")
    if trainable_mode is None:
        trainable_mode = receipt.get("trainable_mode", "adapter") if receipt else "adapter"
    if trainable_mode not in ("adapter", "full"):
        raise ValueError("invalid Limite trainable mode")
    model = load_model(snapshot, attn_implementation="sdpa", device_map=device)
    model = install_adapters(
        model, LimiteAdapterConfig(variant=variant, attention_backend=attention_backend)
    )
    if normal_kernel != "shared":
        set_normal_attention_kernel(model, normal_kernel)
    set_normal_attention_backward(model, normal_backward)
    base_snapshot_sha256 = frozen_fingerprint(model)
    model.archlab_base_snapshot_sha256 = base_snapshot_sha256
    if receipt:
        if allow_backend_migration:
            checkpoint_adapter["attention_backend"] = attention_backend
        if checkpoint_adapter != model.model.adapter_config:
            raise ValueError("checkpoint adapter geometry differs from this experiment")
        if checkpoint_cache is None:
            for name, checksum in receipt["files"].items():
                if file_hash(Path(adapter_checkpoint) / name) != checksum:
                    raise ValueError(f"checkpoint checksum mismatch: {name}")
        expected_base = receipt.get("base_snapshot_sha256", receipt.get("frozen_sha256"))
        if expected_base != base_snapshot_sha256:
            raise ValueError("checkpoint belongs to a different frozen base")
        state = torch.load(
            Path(adapter_checkpoint) / "adapter.pt", map_location="cpu", weights_only=True
        )
        model.model.adapters.load_state_dict(state, strict=True)
        if receipt.get("trainable_mode", "adapter") == "full":
            if "backbone.pt" not in receipt["files"]:
                raise ValueError("full-weight checkpoint lacks backbone state")
            state = torch.load(
                Path(adapter_checkpoint) / "backbone.pt", map_location="cpu", weights_only=True
            )
            expected_keys = {
                name for name in model.state_dict() if not name.startswith("model.adapters.")
            }
            if set(state) != expected_keys:
                raise ValueError("full-weight backbone checkpoint keys differ")
            result = model.load_state_dict(state, strict=False)
            if result.unexpected_keys or set(result.missing_keys) != {
                name for name in model.state_dict() if name.startswith("model.adapters.")
            }:
                raise ValueError("full-weight backbone checkpoint load differs")
    if trainable_mode == "full" or (receipt and receipt.get("trainable_mode") == "full"):
        set_trainable_mode(model, trainable_mode)
    return model


def frozen_fingerprint(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def make_head_part(model):
    """Bind the publisher's native logit transform and summed token loss."""

    def native_part(x, y):
        return F.cross_entropy(model._softcapped_logits(x), y, reduction="sum", ignore_index=-100)

    return native_part


def loss_sum(model, ids, targets, chunk=512, *, head_part=None, checkpoint_head=True):
    hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    flat = hidden.flatten(0, 1)
    labels = targets.flatten()
    part = make_head_part(model) if head_part is None else head_part
    losses = []
    for start in range(0, len(flat), chunk):
        x, y = flat[start : start + chunk], labels[start : start + chunk]
        losses.append(
            checkpoint(part, x, y, use_reentrant=False)
            if checkpoint_head and torch.is_grad_enabled()
            else part(x, y)
        )
    return torch.stack(losses).sum()


def publish_checkpoint(dest, root, oss, manifest):
    """Publish a completed local snapshot; no live model/optimizer references."""
    target = Path(oss) / dest.name
    target.mkdir(parents=True, exist_ok=False)
    for name, checksum in manifest["files"].items():
        shutil.copyfile(dest / name, target / name)
        if file_hash(target / name) != checksum:
            raise ValueError("OSS checksum failed")
    shutil.copyfile(dest / "COMPLETE.json", target / "COMPLETE.json")
    if manifest["trainable_mode"] == "full":
        for name in manifest["files"]:
            temporary = dest / f".{name}.oss-link"
            temporary.symlink_to(target / name)
            temporary.replace(dest / name)
    links = Path(root) / "published"
    links.mkdir(exist_ok=True)
    (links / dest.name).symlink_to(target, target_is_directory=True)


def _freeze_checkpoint_state(value):
    """Detach every mutable tensor before training can advance the optimizer."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _freeze_checkpoint_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_freeze_checkpoint_state(item) for item in value)
    return value


def _write_checkpoint(dest, root, oss, manifest, payloads):
    for name, payload in payloads.items():
        if Path(name).name != name or (dest / name).exists():
            raise ValueError("checkpoint payload must have a unique leaf filename")
        if name.endswith(".json"):
            (dest / name).write_text(json.dumps(payload, indent=2))
        elif name.endswith(".pt"):
            torch.save(payload, dest / name)
        else:
            raise ValueError("checkpoint payload must be JSON or a torch state")
    manifest["files"] = {p.name: file_hash(p) for p in dest.iterdir()}
    (dest / "COMPLETE.json").write_text(json.dumps(manifest, indent=2))
    if oss:
        publish_checkpoint(dest, root, oss, manifest)


def save_adapter(model, optimizer, root, step, tokens, oss=None, extra=None, extra_payloads=None, publisher=None):
    if publisher is not None:
        publisher.wait()
    dest = Path(root) / "checkpoints" / f"step-{step:07d}"
    dest.mkdir(parents=True, exist_ok=False)
    trainable_mode = getattr(model.model, "trainable_mode", "adapter")
    payloads = {
        "adapter.pt": adapter_state(model), "optimizer.pt": optimizer.state_dict(),
        "rng.pt": dict(
            cpu=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        ),
    }
    if trainable_mode == "full":
        payloads["backbone.pt"] = backbone_state(model)
    for name, value in (extra_payloads or {}).items():
        if name in payloads:
            raise ValueError("checkpoint payload must have a unique leaf filename")
        payloads[name] = value
    manifest = dict(
        step=step,
        tokens=tokens,
        adapter=model.model.adapter_config,
        trainable_mode=trainable_mode,
        normal_kernel=getattr(model.model, "normal_kernel", "shared"),
        normal_backward=getattr(model.model, "normal_backward", "tilelang"),
        **(extra or {}),
    )
    if trainable_mode == "full":
        manifest["base_snapshot_sha256"] = model.archlab_base_snapshot_sha256
        manifest.pop("frozen_sha256", None)
    if publisher is None:
        _write_checkpoint(dest, root, oss, manifest, payloads)
    else:
        publisher.submit(_write_checkpoint, dest, root, oss, manifest, _freeze_checkpoint_state(payloads))
    return dest
