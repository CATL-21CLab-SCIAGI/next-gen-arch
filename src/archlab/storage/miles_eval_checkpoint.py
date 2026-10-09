"""Convert an explicitly selected native Miles DCP save to model-only evaluation.

Model storage is hardlinked, including mixed files, rather than reserialized.
The old directory is retired only after native DCP readback and full extent checks.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import io
import json
import math
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from archlab.artifacts import atomic_write_json

MODEL_PREFIXES = ("embedding.", "decoder.", "output_layer.")
FORMAT = "archlab-miles-model-only-dcp-v1"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or len(relative.parts) != 1 or name in (".", ".."):
        raise ValueError(f"DCP file must be a flat relative filename: {name}")
    result = root / relative
    if result.is_symlink() or not result.is_file():
        raise ValueError(f"missing or unsafe DCP file: {result}")
    return result


def _metadata(path):
    import torch.distributed.checkpoint as dcp

    return dcp.FileSystemReader(path).read_metadata()


def _common(path, key):
    import torch
    import torch.distributed.checkpoint as dcp

    target = {key: io.BytesIO()}
    # Native Megatron uses the default planner here. With flattening disabled,
    # some PyTorch versions replace a copied planner dict instead of the caller.
    dcp.load(target, storage_reader=dcp.FileSystemReader(path), no_dist=True)
    value = target[key]
    if isinstance(value, io.BytesIO):
        value.seek(0)
        value = torch.load(value, weights_only=False)
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError("unsupported native common-state representation")
    return value[0]


def _model_keys(metadata):
    keys = set(metadata.state_dict_metadata)
    models = {key for key in keys if key.startswith(MODEL_PREFIXES)}
    common = {key for key in keys if key.startswith("common_state/")}
    rng = {key for key in keys if key.startswith("rng_state/")}
    unknown = keys - models - common - rng
    if not models or len(common) != 1 or unknown:
        raise ValueError(f"unsupported checkpoint keys: {sorted(unknown)[:8]}")
    return models, next(iter(common)), rng


def _storage_files(metadata, keys):
    result = {}
    for index, info in metadata.storage_data.items():
        if index.fqn not in keys:
            continue
        record = result.setdefault(info.relative_path, {"entries": 0, "end": 0})
        record["entries"] += 1
        record["end"] = max(record["end"], info.offset + info.length)
    if {index.fqn for index in metadata.storage_data if index.fqn in keys} != keys:
        raise ValueError("a model metadata key has no storage entries")
    return result


def _tree_inventory(root):
    """Metadata only, refusing links before retirement."""
    files, folders, total = [], [root], 0
    while folders:
        with os.scandir(folders.pop()) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ValueError(f"symlink in retirement candidate: {entry.path}")
                if entry.is_dir(follow_symlinks=False):
                    folders.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    stat = entry.stat(follow_symlinks=False)
                    files.append((Path(entry.path), stat.st_size, stat.st_nlink))
                    total += stat.st_size
                else:
                    raise ValueError(f"unexpected retirement entry: {entry.path}")
    return files, total


def _assert_idle(source):
    for root in (source, source.parent, source.parent.parent):
        if not root.exists():
            continue
        with os.scandir(root) as entries:
            for entry in entries:
                name = entry.name.lower()
                if any(token in name for token in ("pinned", "reader", "in-use", "in_use", ".lease")):
                    raise ValueError(f"checkpoint reader or pin marker present: {entry.path}")


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_model_manifest(path, audit):
    with (audit / "model-files.jsonl").open() as records:
        for line in records:
            record = json.loads(line)
            stat = _safe_file(path, record["file"]).stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
                    record["device"], record["inode"], record["size"], record["mtime_ns"]):
                raise ValueError(f"model shard changed after validation: {record['file']}")


def _publish_eval_root(receipt):
    """A separate tracker selects step zero without changing the live tracker."""
    source, root = Path(receipt["source"]), Path(receipt["native_eval_root"])
    if root.is_symlink():
        raise ValueError("evaluation load root must not be a symlink")
    if root.exists():
        native_checkpoint = root / source.name
        if (not native_checkpoint.is_symlink() or native_checkpoint.resolve() != source
                or (root / "latest_checkpointed_iteration.txt").read_text().strip()
                != str(receipt["iteration"])):
            raise ValueError("existing evaluation root does not select the exported model")
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = root.parent / f".pending-{root.name}-{Path(receipt['audit']).name}"
        if temporary.exists():
            # Only our two small control files and the model-directory redirect
            # may exist; never traverse the redirect during crash recovery.
            for entry in temporary.iterdir():
                if entry.name not in (source.name, "latest_checkpointed_iteration.txt", "EVAL_ONLY.json"):
                    raise ValueError("unexpected pending evaluation-root entry")
                entry.unlink()
            temporary.rmdir()
        temporary.mkdir()
        (temporary / source.name).symlink_to(os.path.relpath(source, root), target_is_directory=True)
        (temporary / "latest_checkpointed_iteration.txt").write_text(str(receipt["iteration"]) + "\n")
        atomic_write_json(temporary / "EVAL_ONLY.json", receipt)
        _fsync_directory(temporary)
        os.rename(temporary, root)
        _fsync_directory(root.parent)
    atomic_write_json(root / "EVAL_ONLY.json", receipt)


def _finish_retirement(receipt, audit):
    """Replay the durable transaction after either rename or a partial deletion."""
    source, stage = Path(receipt["source"]), Path(receipt["stage"])
    retired = Path(receipt["retirement_directory"])
    tracker = source.parent / "latest_checkpointed_iteration.txt"

    def control_plane():
        if _sha(tracker) != receipt["latest_tracker_sha256"]:
            raise ValueError("latest checkpoint tracker changed during export")
        if receipt["iteration"] in receipt["protected_iterations"]:
            raise ValueError("refusing to convert an explicitly protected checkpoint")
        _assert_idle(source)

    control_plane()
    if source.exists() and not (source / "EVAL_ONLY.json").exists():
        if retired.exists() or not stage.exists():
            raise ValueError("inconsistent staged retirement state")
        if _sha(source / ".metadata") != receipt["source_metadata_sha256"]:
            raise ValueError("source metadata changed during export")
        _verify_model_manifest(source, audit)
        control_plane()
        os.rename(source, retired)
        _fsync_directory(source.parent)
    if not source.exists():
        if not retired.exists() or not (stage / "EVAL_ONLY.json").exists():
            raise ValueError("cannot recover missing canonical checkpoint")
        if _sha(stage / ".metadata") != receipt["eval_metadata_sha256"]:
            raise ValueError("staged evaluation metadata changed")
        _verify_model_manifest(stage, audit)
        control_plane()
        os.rename(stage, source)
        _fsync_directory(source.parent)
    if _sha(source / ".metadata") != receipt["eval_metadata_sha256"]:
        raise ValueError("published evaluation metadata changed")
    _verify_model_manifest(source, audit)
    receipt["phase"] = "published"
    atomic_write_json(audit / "receipt.json", receipt)
    _publish_eval_root(receipt)
    if retired.exists():
        control_plane()
        _assert_idle(retired)
        # Retry is safe after a partially completed rmtree: the replacement owns
        # all model inodes and the full pre-retirement manifest remains in audit.
        _tree_inventory(retired)
        shutil.rmtree(retired)
        _fsync_directory(source.parent)
    receipt.update(retired=True, phase="complete", latest_tracker_unchanged=True)
    atomic_write_json(source / "EVAL_ONLY.json", receipt)
    atomic_write_json(Path(receipt["native_eval_root"]) / "EVAL_ONLY.json", receipt)
    atomic_write_json(audit / "receipt.json", receipt)
    return receipt


def _sanitize_common(common):
    value = copy.deepcopy(common)
    for key in ("optimizer", "opt_param_scheduler", "lr_scheduler", "rng_state",
                "rerun_state_machine", "num_floating_point_operations_so_far"):
        value.pop(key, None)
    args = value.get("args")
    if args is not None:
        args.no_save_optim = True
        args.no_save_rng = True
        args.load_main_params_from_ckpt = False
    value["archlab_checkpoint_kind"] = {"format": FORMAT, "can_resume": False}
    return value


def _write_common(stage, key, common):
    import torch.distributed.checkpoint as dcp

    temporary = stage / ".common-build"
    dcp.save({key: [common]}, storage_writer=dcp.FileSystemWriter(
        temporary, single_file_per_rank=False),
        planner=dcp.DefaultSavePlanner(flatten_state_dict=False), no_dist=True)
    metadata = _metadata(temporary)
    if len(metadata.storage_data) != 1:
        raise ValueError("unexpected common-state shard count")
    for info in metadata.storage_data.values():
        source = _safe_file(temporary, info.relative_path)
        os.rename(source, stage / "eval_common.distcp")
        info.relative_path = "eval_common.distcp"
    shutil.rmtree(temporary)
    return metadata


def _sample_keys(metadata, keys, limit=12):
    import torch.distributed.checkpoint as dcp

    eligible = sorted(key for key in keys
                      if isinstance(metadata.state_dict_metadata[key], dcp.TensorStorageMetadata)
                      and math.prod(metadata.state_dict_metadata[key].size) <= 1_000_000)
    if not eligible:
        raise ValueError("no bounded model tensors available for native readback")
    # Cover early/middle/late layers, and adapters when present.
    selected = [key for key in eligible if ".archlab_adapter." in key][:4]
    count = max(1, limit - len(selected))
    selected.extend(eligible[round(i * (len(eligible) - 1) / max(1, count - 1))]
                    for i in range(count))
    return list(dict.fromkeys(selected))


def _load_tensors(path, metadata, keys):
    import torch
    import torch.distributed.checkpoint as dcp

    result = {key: torch.empty(metadata.state_dict_metadata[key].size,
                              dtype=metadata.state_dict_metadata[key].properties.dtype)
              for key in keys}
    dcp.load(result, storage_reader=dcp.FileSystemReader(path),
             planner=dcp.DefaultLoadPlanner(flatten_state_dict=False), no_dist=True)
    return result


def _object_fingerprint(value):
    import torch

    digest = hashlib.sha256()

    def update(item):
        digest.update(type(item).__name__.encode())
        if isinstance(item, io.BytesIO):
            digest.update(item.getvalue())
        elif isinstance(item, bytes):
            digest.update(item)
        elif isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                update(child)
        else:
            digest.update(pickle.dumps(item))

    update(value)
    return digest.hexdigest()


def _byte_state_oracle(source, stage, metadata, keys):
    import torch.distributed.checkpoint as dcp

    lengths = {}
    for index, info in metadata.storage_data.items():
        lengths[index.fqn] = max(lengths.get(index.fqn, 0), info.length)
    eligible = sorted(key for key in keys
                      if isinstance(metadata.state_dict_metadata[key], dcp.BytesStorageMetadata)
                      and lengths[key] <= 4 * 2**20)
    if not eligible:
        return []
    selected = list(dict.fromkeys((eligible[0], eligible[len(eligible) // 2], eligible[-1])))
    proofs = []
    for key in selected:
        before, after = {key: io.BytesIO()}, {key: io.BytesIO()}
        dcp.load(before, storage_reader=dcp.FileSystemReader(source), no_dist=True)
        dcp.load(after, storage_reader=dcp.FileSystemReader(stage), no_dist=True)
        fingerprint = _object_fingerprint(before[key])
        if fingerprint != _object_fingerprint(after[key]):
            raise ValueError(f"native DCP byte-state readback differs: {key}")
        proofs.append({"key": key, "sha256": fingerprint})
    return proofs


def validate_export(source, stage, source_metadata, model_keys, manifest):
    import torch

    exported = _metadata(stage)
    exported_models, common_key, rng_keys = _model_keys(exported)
    if exported_models != model_keys or rng_keys:
        raise ValueError("model/RNG metadata coverage mismatch")
    original_storage = {k: v for k, v in source_metadata.storage_data.items()
                        if k.fqn in model_keys}
    current_storage = {k: v for k, v in exported.storage_data.items() if k.fqn in model_keys}
    if original_storage != current_storage:
        raise ValueError("model storage extents changed")
    for key in model_keys:
        if source_metadata.state_dict_metadata[key] != exported.state_dict_metadata[key]:
            raise ValueError(f"model tensor metadata changed: {key}")
    for record in manifest:
        left = _safe_file(source, record["file"]).stat()
        right = _safe_file(stage, record["file"]).stat()
        if (left.st_dev, left.st_ino, left.st_size) != (right.st_dev, right.st_ino, right.st_size):
            raise ValueError(f"hardlink identity changed: {record['file']}")
        if right.st_size < record["end"]:
            raise ValueError(f"truncated DCP extent: {record['file']}")
    common = _common(stage, common_key)
    if any(key in common for key in ("optimizer", "rng_state", "opt_param_scheduler", "lr_scheduler")):
        raise ValueError("training payload remains in common state")
    if common.get("archlab_checkpoint_kind", {}).get("can_resume") is not False:
        raise ValueError("evaluation-only common marker missing")
    samples = _sample_keys(source_metadata, model_keys)
    before = _load_tensors(source, source_metadata, samples)
    after = _load_tensors(stage, exported, samples)
    tensor_proofs = []
    for key in samples:
        if not torch.equal(before[key], after[key]):
            raise ValueError(f"native DCP tensor readback differs: {key}")
        raw = after[key].contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        tensor_proofs.append({"key": key, "shape": list(after[key].shape),
                              "dtype": str(after[key].dtype), "bytes": len(raw),
                              "sha256": hashlib.sha256(raw).hexdigest()})
    return {"all_model_keys": len(model_keys), "all_model_extents": len(original_storage),
            "all_model_files": len(manifest), "native_tensor_readback": tensor_proofs,
            "native_byte_state_readback": _byte_state_oracle(source, stage, source_metadata, model_keys),
            "common_state_training_payload_removed": True, "torch_version": torch.__version__}


def export_checkpoint(source, audit, *, retire=False, workers=8, protected_iterations=()):
    source, audit = Path(source).absolute(), Path(audit).absolute()
    identity = hashlib.sha256(str(source).encode()).hexdigest()[:16]
    selected_audit = audit / identity
    selected_audit.mkdir(parents=True, exist_ok=True)
    with (selected_audit / ".conversion.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("checkpoint conversion is already running") from error
        previous = selected_audit / "receipt.json"
        if previous.exists():
            receipt = json.loads(previous.read_text())
            if receipt["source"] != str(source) or receipt["audit"] != str(selected_audit):
                raise ValueError("conversion journal identity mismatch")
            if int(source.name.removeprefix("iter_")) in protected_iterations:
                raise ValueError("refusing to convert an explicitly protected checkpoint")
            if receipt.get("phase") in ("validated", "published", "complete"):
                receipt["protected_iterations"] = sorted(set(receipt["protected_iterations"])
                                                         | set(protected_iterations))
                if retire or receipt["phase"] == "complete":
                    return _finish_retirement(receipt, selected_audit)
                return receipt
            stage = Path(receipt["stage"])
            if stage.exists():
                _tree_inventory(stage)
                shutil.rmtree(stage)
        return _export_checkpoint(source, selected_audit, retire=retire, workers=workers,
                                  protected_iterations=protected_iterations)


def _export_checkpoint(source, audit, *, retire, workers, protected_iterations):
    """Stage and validate; retirement requires the caller's explicit flag."""
    source, audit = Path(source).absolute(), Path(audit).absolute()
    if source.is_symlink() or not source.is_dir() or (source / "EVAL_ONLY.json").exists():
        raise ValueError("source must be an unconverted native checkpoint directory")
    if not source.name.startswith("iter_"):
        raise ValueError("source must be a native iteration checkpoint")
    iteration = int(source.name.removeprefix("iter_"))
    if iteration in protected_iterations:
        raise ValueError("refusing to convert an explicitly protected checkpoint")
    tracker = source.parent / "latest_checkpointed_iteration.txt"
    tracker_bytes = tracker.read_bytes()
    if tracker_bytes.strip() == str(iteration).encode():
        raise ValueError("refusing to convert the latest native checkpoint")
    _assert_idle(source)
    identity = hashlib.sha256(str(source).encode()).hexdigest()[:16]
    stage = source.parent / f".eval-export-{source.name}-{identity}"
    retired = source.parent / f".eval-retired-{source.name}-{identity}"
    if retired.exists():
        raise ValueError("retired directory exists without a validated journal")
    atomic_write_json(audit / "receipt.json", {"source": str(source), "audit": str(audit),
                                               "stage": str(stage), "phase": "staging"})
    stage.mkdir(exist_ok=False)
    metadata = _metadata(source)
    models, common_key, rng = _model_keys(metadata)
    files = _storage_files(metadata, models)
    source_inventory, source_bytes = _tree_inventory(source)
    nvme_bytes = sum(size for path, size, _ in source_inventory
                     if path.is_relative_to(source / "nvme_opt_state"))
    if not nvme_bytes:
        raise ValueError("no optimizer state found in conversion source")
    shutil.copy2(source / ".metadata", audit / "original.metadata")
    shutil.copy2(source / "metadata.json", audit / "original-metadata.json")
    for path, _, _ in source_inventory:
        if path.name == "manifest.json" and path.is_relative_to(source / "nvme_opt_state"):
            target = audit / "nvme-manifests" / path.relative_to(source / "nvme_opt_state")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    common = _common(source, common_key)
    sanitized = _sanitize_common(common)

    def link(item):
        name, extents = item
        original = _safe_file(source, name)
        os.link(original, stage / name, follow_symlinks=False)
        stat = (stage / name).stat()
        if stat.st_size < extents["end"]:
            raise ValueError(f"truncated original model shard: {name}")
        return {"file": name, "size": stat.st_size, "device": stat.st_dev,
                "mtime_ns": stat.st_mtime_ns,
                "inode": stat.st_ino, **extents}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        manifest = list(pool.map(link, sorted(files.items())))
    with (audit / "model-files.jsonl").open("w") as output:
        for record in manifest:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    common_metadata = _write_common(stage, common_key, sanitized)
    exported = copy.copy(metadata)
    exported.state_dict_metadata = {key: value for key, value in metadata.state_dict_metadata.items()
                                    if key in models}
    exported.state_dict_metadata.update(common_metadata.state_dict_metadata)
    exported.storage_data = {index: value for index, value in metadata.storage_data.items()
                             if index.fqn in models}
    exported.storage_data.update(common_metadata.storage_data)
    if metadata.planner_data is not None:
        exported.planner_data = {key: value for key, value in metadata.planner_data.items()
                                 if key in models}
        exported.planner_data.update(common_metadata.planner_data or {})
    with (stage / ".metadata").open("wb") as output:
        pickle.dump(exported, output)
        output.flush()
        os.fsync(output.fileno())
    shutil.copy2(source / "metadata.json", stage / "metadata.json")
    _fsync_directory(stage)
    proof = validate_export(source, stage, metadata, models, manifest)
    receipt = {"format": FORMAT, "can_resume": False,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "source": str(source), "stage": str(stage), "iteration": iteration,
               "source_metadata_sha256": _sha(source / ".metadata"),
               "eval_metadata_sha256": _sha(stage / ".metadata"),
               "source_bytes": source_bytes, "optimizer_retired_bytes": nvme_bytes,
               "model_payload_bytes": sum(row["size"] for row in manifest),
               "rng_metadata_keys_removed": len(rng), "validation": proof,
               "native_load_required_flags": ["--no-load-optim", "--no-load-rng"],
               "load_main_params_from_ckpt": False, "retired": False,
               "audit": str(audit), "phase": "validated",
               "native_eval_root": str(source.parent.parent / "eval-checkpoints" / source.name),
               "retirement_directory": str(retired),
               "protected_iterations": list(protected_iterations),
               "latest_tracker_sha256": hashlib.sha256(tracker_bytes).hexdigest(),
               "optimizer_private_bytes": sum(size for path, size, links in source_inventory
                                              if path.is_relative_to(source / "nvme_opt_state")
                                              and links == 1)}
    atomic_write_json(stage / "EVAL_ONLY.json", receipt)
    atomic_write_json(audit / "receipt.json", receipt)
    if not retire:
        return receipt
    return _finish_retirement(receipt, audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--retire", action="store_true")
    parser.add_argument("--protected-iteration", type=int, action="append", default=[])
    args = parser.parse_args()
    result = export_checkpoint(args.source, args.audit, retire=args.retire,
                               protected_iterations=args.protected_iteration)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
