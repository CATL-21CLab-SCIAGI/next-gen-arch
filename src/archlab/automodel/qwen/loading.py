"""Fail-closed checks around the pinned upstream pretrained checkpoint loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def audit_checkpoint_keys(model, checkpoint: Path) -> dict:
    """Account for every checkpoint key before any large allocation or load."""
    index_path = checkpoint / "model.safetensors.index.json"
    raw = index_path.read_bytes()
    index = json.loads(raw)["weight_map"]
    expected = set(model.state_dict_adapter.get_hf_state_dict_keys(model.state_dict()))
    inactive = {key for key in index if key.startswith(("model.visual.", "mtp.", "model.mtp."))}
    missing = expected - set(index)
    unexpected = set(index) - expected - inactive
    if missing or unexpected or expected & inactive:
        raise ValueError(f"checkpoint coverage mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    cache_path = checkpoint / "ARCHLAB_VERIFIED_COPY.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else None
    index_sha256 = hashlib.sha256(raw).hexdigest()
    if cache is not None and cache.get("source_index_sha256") != index_sha256:
        raise ValueError("verified cache index no longer matches its source record")
    return {"index_sha256": index_sha256, "backbone_keys": len(expected),
            "checkpoint_path": str(checkpoint.resolve()),
            "source_checkpoint": cache["source"] if cache is not None else str(checkpoint.resolve()),
            "cache_manifest_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest() if cache is not None else None,
            "inactive_vision_keys": sum(k.startswith("model.visual.") for k in inactive),
            "inactive_mtp_keys": sum(not k.startswith("model.visual.") for k in inactive),
            "missing_keys": 0, "unexpected_keys": 0}


def rebuild_nonpersistent_buffers(model, device: torch.device) -> None:
    """Use the original RoPE constructor, not new positional-encoding equations.

    Meta materialization leaves these unsaved buffers empty. The pinned generic
    checkpointer's reinitialization allowlist does not include this model.
    Reject other unsaved buffers rather than silently assuming they are valid.
    Call before FSDP installation, so no sharding hooks are replaced.
    """
    for name, module in model.named_modules():
        if module._non_persistent_buffers_set and (
            name != "model.language_model.rotary_emb"
            or module._non_persistent_buffers_set != {"inv_freq", "original_inv_freq"}
        ):
            raise ValueError(f"unaccounted nonpersistent buffers: {name}")
    language_model = model.model.language_model
    rotary = language_model.rotary_emb
    with torch.device(device):
        language_model.rotary_emb = type(rotary)(config=rotary.config)
    for buffer in language_model.rotary_emb.buffers():
        if buffer.device != device or buffer.dtype != torch.float32 or not torch.isfinite(buffer).all():
            raise ValueError("invalid reconstructed upstream RoPE buffer")


@torch.no_grad()
def poison_weights_before_load(model) -> None:
    """A missed floating-point destination must remain conspicuously invalid."""
    for parameter in model.parameters():
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        local.fill_(float("nan"))


@torch.no_grad()
def assert_loaded_weights_finite(model) -> int:
    """Check bounded chunks, including the locally owned PLE/expert shards."""
    local_elements = 0
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
        if local.is_meta:
            raise ValueError(f"unmaterialized checkpoint tensor: {name}")
        if local.is_floating_point():
            for chunk in local.reshape(-1).split(4 * 1024 * 1024):
                if not torch.isfinite(chunk).all():
                    raise ValueError(f"missing or nonfinite pretrained weight: {name}")
        local_elements += local.numel()
    return local_elements
