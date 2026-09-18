"""Newest-complete checkpoint retention for explicitly configured experiment roots.

Runs outside immutable training code. Completion markers and every shard's payload
lengths are checked before an older checkpoint is removed. Incomplete saves remain.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def confined(root, relative):
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"unsafe checkpoint path: {path}")
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"symlink checkpoint parent: {parent}")
    return path


def validate(path):
    if path.is_symlink():
        raise ValueError(f"symlink checkpoint: {path}")
    marker = json.loads(confined(path, "COMPLETE.json").read_text())
    if marker["format"] != "archlab-v41-full-sharded-v1":
        raise ValueError("unsupported checkpoint format")
    if (
        len(marker["manifests"]) != marker["world_size"]
        or len(set(marker["manifests"])) != marker["world_size"]
    ):
        raise ValueError("missing or duplicated rank manifests")
    sizes = {
        "torch.bfloat16": 2,
        "torch.float16": 2,
        "torch.float32": 4,
        "torch.float64": 8,
        "torch.int64": 8,
        "torch.int32": 4,
        "torch.int16": 2,
        "torch.int8": 1,
        "torch.uint8": 1,
        "torch.bool": 1,
    }
    payloads = 0
    names = None
    for rank, relative in enumerate(marker["manifests"]):
        manifest_path = confined(path, relative)
        manifest = json.loads(manifest_path.read_text())
        for key in ("contract", "cursor", "world_size"):
            if manifest[key] != marker[key]:
                raise ValueError(f"rank {rank} {key} mismatch")
        if manifest["rank"] != rank:
            raise ValueError("rank mismatch")
        current_names = [entry["name"] for entry in manifest["tensors"]]
        if names is not None and current_names != names:
            raise ValueError("tensor names differ across ranks")
        names = current_names
        for entry in manifest["tensors"]:
            if sum(c["elements"] for c in entry["chunks"]) != math.prod(entry["shape"]):
                raise ValueError("tensor chunks incomplete")
            for chunk in entry["chunks"]:
                payload = confined(
                    path, str(manifest_path.parent.relative_to(path) / chunk["file"])
                )
                size = payload.stat().st_size
                raw = chunk["elements"] * sizes[entry["dtype"]]
                if not raw <= size < raw + 65536:
                    raise ValueError(f"bad payload length: {payload}")
                payloads += 1
        for filename in [*manifest["optimizer_states"], "rng.pt"]:
            payload = confined(path, str(manifest_path.parent.relative_to(path) / filename))
            if payload.stat().st_size <= 0:
                raise ValueError(f"empty payload: {payload}")
            payloads += 1
    return {
        "path": str(path),
        "cursor": marker["cursor"],
        "ranks": marker["world_size"],
        "payload_files_checked": payloads,
        "validated_utc": utc(),
        "validation": "all rank identities, tensor element counts and payload lengths; optimizer/RNG presence",
    }


def complete_paths(group):
    paths = [Path(p) for p in group.get("paths", [])]
    for root in group.get("roots", []):
        parent = Path(root)
        if parent.is_symlink() or parent.resolve() != parent:
            raise ValueError("root must be absolute and contain no symlinks")
        paths.extend(p for p in parent.glob("step-*") if re.fullmatch(r"step-\d+", p.name))
    result = []
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"symlink candidate: {path}")
        marker = path / "COMPLETE.json"
        if marker.exists():
            data = json.loads(marker.read_text())
            result.append((data["cursor"]["step"], data["cursor"]["supervised_tokens"], path))
    return sorted(result, key=lambda x: (x[0], x[1], str(x[2])))


def inventory_tree(path):
    # DirEntry type checks use directory metadata; avoid three serial NAS stat
    # round trips for each of the tens of thousands of files in a checkpoint.
    files = []
    folders = [path]
    while folders:
        with os.scandir(folders.pop()) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ValueError(f"symlink inside deletion candidate: {entry.path}")
                if entry.is_dir(follow_symlinks=False):
                    folders.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files.append(Path(entry.path))
                else:
                    raise ValueError(f"unexpected checkpoint entry: {entry.path}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        size = sum(pool.map(lambda p: p.stat().st_size, files))
    return files, size


def delete_tree(path):
    # Rank subdirectories are independent. Limit concurrency to eight metadata
    # workers; no model payloads are read and no GPU memory is used.
    def remove(child):
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(remove, list(path.iterdir())))
    path.rmdir()


def remove_checkpoint(path, replacement, audit):
    # Archive manifests before deletion. Refuse symlinks anywhere in the tree.
    files, size = inventory_tree(path)
    key = hashlib.sha256(str(path).encode()).hexdigest()[:20]
    archive = audit / "deleted" / key
    archive.mkdir(parents=True, exist_ok=True)
    for source in files:
        if source.name in ("COMPLETE.json", "MANIFEST.json"):
            target = archive / source.relative_to(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    staged = path.parent / (".retention-delete-" + key)
    receipt = {
        "original": str(path),
        "staged": str(staged),
        "replacement": str(replacement),
        "bytes": size,
        "started_utc": utc(),
        "complete": False,
    }
    atomic_json(archive / "RECEIPT.json", receipt)
    path.rename(staged)
    delete_tree(staged)
    receipt.update(complete=True, completed_utc=utc())
    atomic_json(archive / "RECEIPT.json", receipt)
    return receipt


def sweep(config, audit, apply=False):
    audit.mkdir(parents=True, exist_ok=True)
    # A killed deletion is resumed only from a recorded, exact rename destination.
    if apply:
        for file in (audit / "deleted").glob("*/RECEIPT.json"):
            receipt = json.loads(file.read_text())
            if receipt["complete"]:
                continue
            staged = Path(receipt["staged"])
            if staged.exists():
                if staged.is_symlink() or not staged.name.startswith(".retention-delete-"):
                    raise ValueError("unsafe pending deletion")
                delete_tree(staged)
                receipt.update(complete=True, completed_utc=utc())
                atomic_json(file, receipt)
    result = {"utc": utc(), "apply": apply, "groups": [], "errors": []}
    for group in config["groups"]:
        try:
            candidates = complete_paths(group)
            if not candidates:
                continue
            newest = candidates[-1][2]
            older = [x[2] for x in candidates[:-1] if str(x[2]) not in config.get("protected", {})]
            # Validate once for each newly completed save; immutable completed files
            # are not repeatedly read while the training shares NAS bandwidth.
            digest = hashlib.sha256((newest / "COMPLETE.json").read_bytes()).hexdigest()
            cache = audit / (
                "verified-" + hashlib.sha256(str(newest).encode()).hexdigest()[:20] + ".json"
            )
            if not cache.exists() or json.loads(cache.read_text()).get("marker_sha256") != digest:
                verification = validate(newest)
                verification["marker_sha256"] = digest
                atomic_json(cache, verification)
            record = {
                "name": group["name"],
                "newest": str(newest),
                "delete": [str(x) for x in older],
            }
            if apply:
                for root in group.get("roots", []):
                    atomic_json(
                        Path(root) / "LATEST.json",
                        {"path": str(newest), "marker_sha256": digest, "utc": utc()},
                    )
                record["removed"] = [remove_checkpoint(path, newest, audit) for path in older]
            result["groups"].append(record)
        except Exception as error:
            result["errors"].append({"group": group["name"], "error": repr(error)})
    atomic_json(audit / "LATEST.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        raise ValueError("interval must be 1–60 seconds")
    args.audit.mkdir(parents=True, exist_ok=True)
    with (args.audit / "LOCK").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Retention is already running for this policy.", flush=True)
            return
        while True:
            result = sweep(json.loads(args.config.read_text()), args.audit, args.apply)
            print(json.dumps(result), flush=True)
            if not args.watch:
                if result["errors"]:
                    raise SystemExit(1)
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
