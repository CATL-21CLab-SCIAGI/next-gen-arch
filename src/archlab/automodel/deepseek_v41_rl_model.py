"""Construct GPU-resident RL actors from an exact, same-mesh parent checkpoint.

This is a weights-only phase transition, not SFT resume. The caller owns the
fresh RL optimizer, RNG seeding, policy head, and numerical admission gate.
Nothing in a construction receipt authorizes production training.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCE_CHANGE_REASONS = {
    "automodel/deepseek_v41_official_adapter.py": "right-padding-and-additional-variants",
    "automodel/deepseek_v41_full_indexer.py": "right-padding-indexer-mask",
    "automodel/deepseek_v41_full_training.py": "SFT-entrypoint-replaced-by-RL",
    "automodel/deepseek_v41_scratch_training.py": "SFT-entrypoint-replaced-by-RL",
    "automodel/deepseek_v41_scratch_construct.py": "additional-variants-same-selected-geometry",
    "automodel/deepseek_v41_control.py": "SFT-serving-control-not-used-by-RL",
    "automodel/deepseek_v41_live_window.py": "SFT-chat-control-changed-inference-head-preserved",
    "serving/openai_chat.py": "SFT-serving-control-not-used-by-RL",
}
EXECUTION_CHANGES = {
    "objective": "rl-policy-gradient",
    "optimizer_state": "fresh",
    "rng_state": "fresh-rl-seed",
    "right_padding": True,
}
RUNTIME_FIELDS = (
    "container_image",
    "packages",
    "cuda",
    "nccl",
    "automodel_commit",
    "automodel_root",
    "reproducibility",
    "world_size",
    "ep_size",
    "expert_fsdp_size",
    "engram_owners",
    "moe_precision",
    "hc_precision",
)


def configure_rl_trainability(model, *, freeze_router=False):
    """Declare the exact trainable set after parent restoration and before forward."""
    if type(freeze_router) is not bool:
        raise ValueError("freeze_router must be boolean")
    frozen = []
    for name, parameter in model.named_parameters():
        freeze = freeze_router and ".ffn.gate." in name
        parameter.requires_grad_(not freeze)
        if freeze:
            frozen.append(name)
    if freeze_router and not frozen:
        raise ValueError("router freeze requested but no V4.1 router parameters found")
    model._archlab_rl_frozen_parameter_names = tuple(sorted(frozen))
    return {
        "freeze_router": freeze_router,
        "frozen_parameter_names": sorted(frozen),
        "all_other_text_parameters_trainable": True,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def read_parent_checkpoint(checkpoint, *, family, variant, world_size, tiny=False):
    """Metadata-only admission. Full and scratch checkpoints are not resharded."""
    required_world = {"full": 16, "scratch": 8}.get(family)
    if required_world is None or world_size != required_world:
        raise ValueError("RL actor requires full16 or scratch8 parent ownership")
    if variant not in (
        ("normal", "simplicial")
        if family == "full"
        else ("normal", "simplicial", "linear", "linsimp")
    ):
        raise ValueError("Unsupported actor variant for selected family")
    path = Path(checkpoint) / "COMPLETE.json"
    marker = json.loads(path.read_text())
    contract = marker["contract"]
    expected_format = (
        "archlab-v41-full-training-v1" if family == "full" else "archlab-v41-scratch-comparison-v1"
    )
    if (
        marker.get("format") != "archlab-v41-full-sharded-v1"
        or marker.get("world_size") != world_size
        or contract.get("world_size") != world_size
        or contract.get("format") != expected_format
        or contract.get("variant") != variant
        or contract.get("tiny") is not tiny
        or contract.get("cpu_offload") is not False
    ):
        raise ValueError("Parent checkpoint format, variant, geometry class, or ownership differs")
    runtime = contract["runtime"]
    for field, expected in (
        ("world_size", world_size),
        ("ep_size", 8),
        ("expert_fsdp_size", world_size // 8),
        ("engram_owners", world_size),
    ):
        if runtime.get(field) != expected or family == "full" and contract.get(field) != expected:
            raise ValueError(f"Parent ownership differs: {field}")
    expected_manifests = [f"rank-{rank:02d}/MANIFEST.json" for rank in range(world_size)]
    if marker.get("manifests") != expected_manifests:
        raise ValueError("Parent does not list exactly the expected rank manifests")
    for relative in expected_manifests:
        if not (Path(checkpoint) / relative).is_file():
            raise ValueError(f"Parent is missing a rank manifest: {relative}")
    return marker, {
        "checkpoint": str(Path(checkpoint).resolve()),
        "marker_sha256": _sha(path),
        "parent_contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest(),
    }


def audit_parent_sources(
    marker, *, source_root=None, declared_source_changes=None, scratch_constructor_path=None
):
    """Every changed historical file needs its real old/new digest and reason.

    Mechanism, optimizer, and container files cannot be exempted. A separately
    provided scratch constructor must exactly match the parent's sealed source.
    """
    source_root = Path(source_root) if source_root else Path(__file__).resolve().parents[1]
    from archlab.source_compatibility import formatting_predecessor

    declarations = dict(declared_source_changes or {})
    hashes = marker["contract"]["implementation_sha256"]
    used, report = set(), []
    for relative, expected in hashes.items():
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Parent has an invalid source digest: {relative}")
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Invalid source provenance path")
        path = source_root / rel
        if relative == "automodel/deepseek_v41_scratch_construct.py" and scratch_constructor_path:
            path = Path(scratch_constructor_path)
            if _sha(path) != expected:
                raise ValueError("Selected scratch constructor differs from sealed parent source")
        actual = _sha(path)
        predecessor, formatting = formatting_predecessor(relative, path, actual)
        receipt = {
            "relative": relative,
            "path": str(path.resolve()),
            "parent_sha256": expected,
            "current_sha256": actual,
        }
        if formatting is not None:
            receipt["formatting_only_migration"] = formatting
        if predecessor != expected:
            required = {
                "before_sha256": expected,
                "after_sha256": predecessor,
                "reason": SOURCE_CHANGE_REASONS.get(relative),
            }
            if required["reason"] is None or declarations.get(relative) != required:
                raise ValueError(
                    f"Undeclared or disallowed parent implementation change: {relative}"
                )
            receipt["declared_change"] = declarations[relative]
            used.add(relative)
        report.append(receipt)
    if set(declarations) != used:
        raise ValueError("Source declarations include unused or unknown exemptions")
    if scratch_constructor_path and "automodel/deepseek_v41_scratch_construct.py" not in hashes:
        raise ValueError("No parent hash for selected scratch constructor")
    return report


def validate_constructed_runtime(parent, loading, *, family):
    old, new = _canonical(parent["runtime"]), _canonical(loading)
    for field in RUNTIME_FIELDS:
        if field not in old or field not in new or old[field] != new[field]:
            raise ValueError(f"Constructed actor differs from parent runtime: {field}")
    sparse_field = "sparse_backward" if family == "full" else "sparse_precision"
    if old.get(sparse_field) != new.get("sparse_precision") or new.get("sparse_precision") is None:
        raise ValueError("Constructed sparse attention backend differs from parent")
    if family == "scratch":
        for field in (
            "geometry",
            "parameters",
            "adapter_layers",
            "variant",
            "boundaries",
            "router_auxiliary_loss_coefficient",
        ):
            if old.get(field) != new.get(field) or field not in new:
                raise ValueError(f"Scratch actor geometry differs: {field}")
        if not old.get("right_padding_masked") or not new.get("right_padding_masked"):
            raise ValueError("Scratch right-padding contract changed")
    if "resolved_kernel_packages" in old and old["resolved_kernel_packages"] != new.get(
        "resolved_kernel_packages"
    ):
        raise ValueError("Resolved native kernel packages differ from parent")


def _ordered_weight_payloads(folder, entries, depth):
    """Read/checksum bounded CPU payloads concurrently; yield manifest order."""
    import torch

    jobs = iter((entry, chunk) for entry in entries for chunk in entry["chunks"])

    def read(job):
        entry, chunk = job
        host = torch.load(folder / chunk["file"], map_location="cpu", weights_only=True)
        if (
            not isinstance(host, torch.Tensor)
            or host.ndim != 1
            or str(host.dtype) != entry["dtype"]
            or host.numel() != chunk["elements"]
            or hashlib.sha256(memoryview(host.contiguous().view(torch.uint8).numpy())).hexdigest()
            != chunk["sha256"]
        ):
            raise ValueError(f"Weight payload checksum/dtype/shape differs: {entry['name']}")
        return host

    if depth == 0:
        for job in jobs:
            yield read(job)
        return
    with ThreadPoolExecutor(max_workers=depth) as pool:
        pending = deque()
        for _ in range(depth):
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(read, job))
        while pending:
            host = pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(read, job))
            yield host


def _restore_local_weights(model, checkpoint, marker, *, rank, expected_device, prefetch_chunks=2):
    """Private local reader, CPU-testable; public construction always requires CUDA."""
    import torch

    from archlab.optimizers.sharded_adafactor import local_tensor

    if type(prefetch_chunks) is not int or not 0 <= prefetch_chunks <= 2:
        raise ValueError("weight read-ahead is bounded to zero, one, or two chunks")

    checkpoint = Path(checkpoint)
    folder = checkpoint / f"rank-{rank:02d}"
    manifest_path = folder / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("rank") != rank or not 0 <= rank < marker["world_size"]:
        raise ValueError("Rank identity differs from checkpoint")
    for field in ("world_size", "cursor", "contract"):
        if manifest.get(field) != marker[field]:
            raise ValueError(f"Parent marker/rank manifest {field} mismatch")
    named = list(model.named_parameters()) + list(model.named_buffers())
    if len(named) != len(manifest["tensors"]):
        raise ValueError("Checkpoint tensor inventory differs")
    chunk_names = set()
    # Validate the entire tensor schema before copying any weights.
    for (name, tensor), entry in zip(named, manifest["tensors"], strict=True):
        local = local_tensor(tensor)
        if (
            entry["name"] != name
            or entry["shape"] != list(local.shape)
            or entry["global_shape"] != list(tensor.shape)
            or entry["dtype"] != str(local.dtype)
            or not local.is_contiguous()
        ):
            raise ValueError(f"Checkpoint tensor name/shape/dtype differs: {name}")
        elements = 0
        for chunk in entry["chunks"]:
            filename = chunk["file"]
            if (
                not re.fullmatch(r"tensor-\d+-\d+\.pt", filename)
                or filename in chunk_names
                or type(chunk["elements"]) is not int
                or chunk["elements"] <= 0
                or chunk["elements"] * local.element_size() > 256 * 1024 * 1024
                or not re.fullmatch(r"[0-9a-f]{64}", chunk["sha256"])
            ):
                raise ValueError(f"Invalid or unbounded weight chunk: {name}")
            chunk_names.add(filename)
            elements += chunk["elements"]
        if elements != local.numel():
            raise ValueError(f"Incomplete tensor chunk coverage: {name}")
    for name, parameter in model.named_parameters():
        if local_tensor(parameter).device.type != expected_device:
            raise ValueError(f"Actor parameter is not on {expected_device}: {name}")
    count = size = 0
    began = time.monotonic()
    next_progress = 8 * 2**30
    print(
        json.dumps(
            {"event": "rl_weight_restore_start", "rank": rank, "checkpoint": str(checkpoint)}
        ),
        flush=True,
    )
    from contextlib import closing

    with (
        torch.no_grad(),
        closing(_ordered_weight_payloads(folder, manifest["tensors"], prefetch_chunks)) as payloads,
    ):
        for (_, tensor), entry in zip(named, manifest["tensors"], strict=True):
            local = local_tensor(tensor)
            flat, offset = local.view(-1), 0
            for _chunk in entry["chunks"]:
                host = next(payloads)
                flat[offset : offset + host.numel()].copy_(host)
                offset += host.numel()
                size += host.numel() * host.element_size()
                count += 1
                del host
                if size >= next_progress:
                    print(
                        json.dumps(
                            {
                                "event": "rl_weight_restore_progress",
                                "rank": rank,
                                "loaded_local_gib": size / 2**30,
                                "seconds": time.monotonic() - began,
                            }
                        ),
                        flush=True,
                    )
                    next_progress += 8 * 2**30
    print(
        json.dumps(
            {
                "event": "rl_weight_restore_complete",
                "rank": rank,
                "loaded_local_gib": size / 2**30,
                "seconds": time.monotonic() - began,
            }
        ),
        flush=True,
    )
    return {
        "rank": rank,
        "rank_manifest_sha256": _sha(manifest_path),
        "tensor_entries": len(named),
        "verified_chunks": count,
        "bytes": size,
        "all_weight_checksums_verified": True,
        "optimizer_loaded": False,
        "rng_loaded": False,
        "cpu_weight_offload": False,
        "reader_prefetch_chunks": prefetch_chunks,
        "maximum_payloads_in_flight_including_current": prefetch_chunks + 1,
    }


def _scratch_constructor(path):
    if path is None:
        from archlab.automodel.deepseek_v41_scratch_construct import construct_scratch

        return construct_scratch
    spec = importlib.util.spec_from_file_location(
        "_archlab_rl_parent_scratch_constructor", Path(path)
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import sealed scratch constructor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.construct_scratch


def _construct_full_shell(*, variant, assets, weights, tiny):
    """Unrestored shell shared by strict parent loading and memory-only probes."""
    from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
    from archlab.automodel.deepseek_v41_full_boundaries import install_full_training_boundaries
    from archlab.automodel.deepseek_v41_full_eval_construct import build_eval_shell
    from archlab.automodel.deepseek_v41_full_indexer import install_trainable_indexers
    from archlab.automodel.deepseek_v41_official_adapter import install_official_adapters

    model, _, loading = build_eval_shell(weights=Path(weights), assets=Path(assets), tiny=tiny)
    adapters = install_official_adapters(
        model,
        V41AdapterConfig(width=256) if tiny else V41AdapterConfig(),
        layer_indices=(1, 3, 5) if tiny else (4, 9, 14, 19, 24, 29, 34, 39),
        device="cuda",
        variant=variant,
        allow_right_padding=True,
        backend="deterministic" if variant == "simplicial" else "flash-attn-deterministic",
    )
    loading["boundaries"] = install_full_training_boundaries(model)
    indexers = install_trainable_indexers(model, sample_queries=64)
    loading["adapter_layers"] = list(adapters)
    return model, indexers, loading


def construct_rl_actor(
    *,
    checkpoint,
    family,
    variant,
    assets,
    weights=None,
    base_config=None,
    tiny=False,
    declared_source_changes=None,
    declared_execution_changes=None,
    scratch_constructor_path=None,
    resolved_kernel_packages=None,
    expected_parent_marker_sha256=None,
):
    """Return ``(model, indexers, loading, parent_marker)`` without an optimizer.

    ``checkpoint=None,tiny=True`` constructs a qualification-only random model.
    Production requires a sealed parent and an unchanged world/EP ownership.
    A nondefault scratch backend requires its exact sealed constructor path;
    runtime validation rejects accidental backend substitution after building.
    """
    import torch
    import torch.distributed as dist

    from archlab.optimizers.sharded_adafactor import local_tensor

    world = dist.get_world_size()
    if world != {"full": 16, "scratch": 8}.get(family):
        raise ValueError("RL construction must preserve full16/scratch8 ownership")
    if dict(declared_execution_changes or {}) != EXECUTION_CHANGES:
        raise ValueError(f"Declare exactly the reviewed RL execution changes: {EXECUTION_CHANGES}")
    if checkpoint is None and not tiny:
        raise ValueError("Production RL requires an existing complete parent checkpoint")
    marker, parent_receipt, source_receipt = None, None, []
    if checkpoint is not None:
        marker, parent_receipt = read_parent_checkpoint(
            checkpoint, family=family, variant=variant, world_size=world, tiny=tiny
        )
        if (
            expected_parent_marker_sha256 is not None
            and parent_receipt["marker_sha256"] != expected_parent_marker_sha256
        ):
            raise ValueError("Parent marker differs from the selected immutable checkpoint")
        source_receipt = audit_parent_sources(
            marker,
            declared_source_changes=declared_source_changes,
            scratch_constructor_path=scratch_constructor_path,
        )
    elif declared_source_changes or scratch_constructor_path:
        raise ValueError("Tiny random qualification has no parent source exemptions")
    if family == "full":
        if variant not in ("normal", "simplicial") or weights is None:
            raise ValueError(
                "Full actors need normal/simplicial and the recorded model config directory"
            )
        model, indexers, loading = _construct_full_shell(
            variant=variant, assets=assets, weights=weights, tiny=tiny
        )
    else:
        if base_config is None:
            raise ValueError("Scratch actors require the recorded base configuration")
        model, indexers, _, loading = _scratch_constructor(scratch_constructor_path)(
            base_config=base_config, assets=assets, variant=variant, tiny=tiny
        )
    if resolved_kernel_packages is not None:
        loading["resolved_kernel_packages"] = resolved_kernel_packages
    if marker is not None:
        validate_constructed_runtime(marker["contract"], loading, family=family)
        logical = torch.tensor(
            sum(
                local_tensor(p).numel() / (world if not hasattr(p, "placements") else 1)
                for p in model.parameters()
            ),
            device="cuda",
            dtype=torch.float64,
        )
        dist.all_reduce(logical)
        expected = (
            marker["contract"]["full_text_parameters"]
            if family == "full"
            else marker["contract"]["runtime"]["parameters"]
        )
        if int(logical) != expected:
            raise ValueError("Constructed actor parameter count differs from parent")
        if _sha(Path(checkpoint) / "COMPLETE.json") != parent_receipt["marker_sha256"]:
            raise ValueError("Parent marker changed during construction")
        restore = _restore_local_weights(
            model, checkpoint, marker, rank=dist.get_rank(), expected_device="cuda"
        )
        dist.barrier()
    else:
        restore = {
            "optimizer_loaded": False,
            "rng_loaded": False,
            "qualification_random_initialization": True,
        }
    model.requires_grad_(True)
    model.train()
    if any(
        local_tensor(p).device.type != "cuda" or not p.requires_grad for p in model.parameters()
    ):
        raise ValueError("RL actor must keep every trainable weight resident on GPU")
    loading = {
        **_canonical(loading),
        "rl_transition": {
            "family": family,
            "variant": variant,
            "tiny": tiny,
            "qualification_only": bool(tiny),
            "numerical_qualification_required": True,
            "actor_weight_origin": "random-qualification"
            if marker is None
            else "parent-checkpoint",
            "parent": parent_receipt,
            "parent_contract": None if marker is None else marker["contract"],
            "source_provenance": source_receipt,
            "execution_changes": dict(declared_execution_changes),
            "restore": restore,
            "helper_sha256": _sha(Path(__file__)),
            "cpu_weight_offload": False,
            "optimizer_state_policy": "fresh-RL-optimizer-owned-by-caller",
        },
    }
    return model, indexers, loading, marker
