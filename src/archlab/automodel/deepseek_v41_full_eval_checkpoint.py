# ruff: noqa: I001  # Preserve the checkpoint-qualified import order.
"""Checksum-verified, bounded-staging weights-only restore for full V4.1 eval."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time


def marker_contract(path, *, variant, expected_tokens=None, trained_source=None):
    path = Path(path)
    marker = json.loads((path / "COMPLETE.json").read_text())
    if marker["format"] != "archlab-v41-full-sharded-v1" or marker["world_size"] != 16:
        raise ValueError(
            "evaluation requires a complete full-state checkpoint on the trained16-rank mesh"
        )
    c = marker["contract"]
    if c["variant"] != variant or (
        c["world_size"],
        c["ep_size"],
        c["expert_fsdp_size"],
        c["engram_owners"],
    ) != (16, 8, 2, 16):
        raise ValueError("checkpoint variant or ownership differs")
    if expected_tokens is not None and marker["cursor"]["supervised_tokens"] != expected_tokens:
        raise ValueError("checkpoint is not at the selected matched token count")
    source = Path(__file__).resolve().parents[1]
    if trained_source is not None:
        from archlab.automodel.deepseek_v41_control import verify_checkpoint_sources

        verify_checkpoint_sources(c, source.parents[1], trained_source)
    else:
        for relative, digest in c["implementation_sha256"].items():
            if hashlib.sha256((source / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"trained model implementation changed: {relative}")
    return marker


def restore_weights(model, path, *, variant, expected_tokens=None, trained_source=None):
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from archlab.automodel.deepseek_v41_training import emit
    from archlab.automodel.deepseek_v41_full_checkpoint import _checksum

    marker = marker_contract(
        path, variant=variant, expected_tokens=expected_tokens, trained_source=trained_source
    )
    if dist.get_world_size() != marker["world_size"]:
        raise ValueError("wrong restore mesh size")
    rank = dist.get_rank()
    rank_path = Path(path) / f"rank-{rank:02d}"
    manifest = json.loads((rank_path / "MANIFEST.json").read_text())
    if (
        manifest["rank"] != rank
        or manifest["cursor"] != marker["cursor"]
        or manifest["contract"] != marker["contract"]
    ):
        raise ValueError("checkpoint rank manifest disagrees with COMPLETE")
    named = list(model.named_parameters()) + list(model.named_buffers())
    count = 0
    loaded = 0
    threshold = 8 * 2**30
    began = time.monotonic()
    emit("eval_weight_restore_start", variant=variant, checkpoint=str(path))
    with torch.no_grad():
        for (name, tensor), entry in zip(named, manifest["tensors"], strict=True):
            local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            if (
                name != entry["name"]
                or list(local.shape) != entry["shape"]
                or list(tensor.shape) != entry["global_shape"]
                or str(local.dtype) != entry["dtype"]
            ):
                raise ValueError(f"checkpoint tensor shape/name/dtype mismatch: {name}")
            if not local.is_contiguous():
                raise ValueError(f"restore destination must be contiguous: {name}")
            flat = local.view(-1)
            first = 0
            for chunk in entry["chunks"]:
                host = torch.load(rank_path / chunk["file"], map_location="cpu", weights_only=True)
                if (
                    host.numel() != chunk["elements"]
                    or host.dtype != local.dtype
                    or _checksum(host) != chunk["sha256"]
                ):
                    raise ValueError(f"checkpoint chunk checksum or dtype mismatch: {name}")
                flat[first : first + host.numel()].copy_(host)
                first += host.numel()
                loaded += host.numel() * host.element_size()
                count += 1
                del host
            if first != flat.numel():
                raise ValueError(f"incomplete parameter restore: {name}")
            if loaded >= threshold:
                emit(
                    "eval_weight_restore_progress",
                    variant=variant,
                    loaded_local_gib=loaded / 2**30,
                    seconds=time.monotonic() - began,
                )
                threshold += 8 * 2**30
    model.requires_grad_(False)
    model.eval()
    for name, p in model.named_parameters():
        local = p.to_local() if isinstance(p, DTensor) else p
        if local.device.type != "cuda" or p.requires_grad or p.grad is not None:
            raise ValueError(f"evaluation parameters must stay frozen on GPU: {name}")
    torch.cuda.synchronize()
    dist.barrier()
    result = {
        "variant": variant,
        "rank": rank,
        "cursor": marker["cursor"],
        "tensor_entries": len(named),
        "verified_payload_chunks": count,
        "loaded_local_gib": loaded / 2**30,
        "seconds": time.monotonic() - began,
        "all_weight_payload_sha256_verified": True,
        "optimizer_loaded": False,
        "cpu_weight_offload": False,
    }
    emit("eval_weight_restore_complete", **{k: v for k, v in result.items() if k != "rank"})
    return result
