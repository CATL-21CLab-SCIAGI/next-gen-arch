"""Copy and verify an immutable model directory before replacing it with a link."""

import argparse
import hashlib
import json
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Source must be a real directory")
    result = {}
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(parent) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not (
                stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
            ):
                raise ValueError(f"Unsupported source entry: {path}")
            if stat.S_ISREG(info.st_mode):
                result[str(path.relative_to(root))] = {
                    "bytes": info.st_size,
                    "mtime_ns": info.st_mtime_ns,
                    "inode": info.st_ino,
                }
    return result


def target_file(destination, relative):
    path = destination / relative
    for parent in [path, *path.parents]:
        if parent.is_symlink():
            raise ValueError(f"Destination symlink would redirect a write: {parent}")
        if parent == destination:
            break
    return path


def prepare(source, destination, journal):
    source, destination = source.absolute(), destination.absolute()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be disjoint")
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("Destination must already be an explicit real directory")
    files = inventory(source)
    before = sorted(path.name for path in destination.iterdir())
    journal.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "source": str(source),
        "destination": str(destination),
        "phase": "copying",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "destination_children_before": before,
        "files": dict(files),
        "bytes": sum(info["bytes"] for info in files.values()),
    }
    save(journal, state)
    existing = {}
    for relative, info in files.items():
        target = target_file(destination, relative)
        if target.exists():
            if not target.is_file() or target.stat().st_size != info["bytes"]:
                raise ValueError(f"Destination collision: {relative}")
            source_hash = digest(source / relative)
            if source_hash != digest(target):
                raise ValueError(f"Destination content differs: {relative}")
            existing[relative] = source_hash

    def copy_one(item):
        relative, info = item
        src, dst = source / relative, target_file(destination, relative)
        if relative in existing:
            return relative, {**info, "sha256": existing[relative], "reused_existing": True}
        dst.parent.mkdir(parents=True, exist_ok=True)
        checksum = hashlib.sha256()
        with src.open("rb") as inp, dst.open("xb") as out:
            while chunk := inp.read(8 * 1024 * 1024):
                out.write(chunk)
                checksum.update(chunk)
            out.flush()
            os.fsync(out.fileno())
        if src.stat().st_size != info["bytes"] or src.stat().st_mtime_ns != info["mtime_ns"]:
            raise ValueError(f"Source changed during copy: {relative}")
        if dst.stat().st_size != info["bytes"] or digest(dst) != checksum.hexdigest():
            raise ValueError(f"Destination verification failed: {relative}")
        print(json.dumps({"copied_and_verified": relative, "bytes": info["bytes"]}), flush=True)
        return relative, {**info, "sha256": checksum.hexdigest(), "reused_existing": False}

    with ThreadPoolExecutor(max_workers=3) as pool:
        for relative, info in pool.map(copy_one, files.items()):
            state["files"][relative] = info
            save(journal, state)
    if inventory(source) != files:
        raise ValueError("Source tree changed during migration")
    if not set(before) <= set(path.name for path in destination.iterdir()):
        raise ValueError("Existing destination entries disappeared")
    state.update(phase="verified", verified_at_utc=datetime.now(timezone.utc).isoformat())
    save(journal, state)
    return state


def commit(journal):
    state = json.loads(journal.read_text())
    source, destination = Path(state["source"]), Path(state["destination"])
    if state["phase"] != "verified":
        raise ValueError("Only a fully verified copy can be committed")
    expected = {
        key: {name: value[name] for name in ("bytes", "mtime_ns", "inode")}
        for key, value in state["files"].items()
    }
    if inventory(source) != expected:
        raise ValueError("Source changed after verification")
    for relative, info in state["files"].items():
        target = target_file(destination, relative)
        if target.stat().st_size != info["bytes"] or digest(target) != info["sha256"]:
            raise ValueError(f"Verified destination changed: {relative}")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = source.parent / f".{source.name}.nas-backup-{tag}"
    link = source.parent / f".{source.name}.oss-link-{tag}"
    if backup.exists() or link.exists() or link.is_symlink():
        raise ValueError("Migration staging path exists")
    link.symlink_to(destination, target_is_directory=True)
    state.update(phase="switch_prepared", backup=str(backup), temporary_link=str(link))
    save(journal, state)
    # NFSv3 cannot exchange a directory and symlink atomically. Keep a verified
    # backup and restore it if publishing the link fails.
    source.rename(backup)
    try:
        link.rename(source)
    except BaseException:
        backup.rename(source)
        link.unlink(missing_ok=True)
        raise
    if not source.is_symlink() or source.resolve() != destination.resolve():
        raise ValueError("Symlink did not resolve to the verified destination")
    for relative, info in state["files"].items():
        if (source / relative).stat().st_size != info["bytes"]:
            raise ValueError("Original model path does not resolve correctly")
    if inventory(backup) != expected:
        raise ValueError("NAS backup changed; keeping it for recovery")
    state.update(phase="linked", linked_at_utc=datetime.now(timezone.utc).isoformat())
    save(journal, state)
    shutil.rmtree(backup)
    state.update(
        phase="complete",
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        removed_nas_bytes=state["bytes"],
    )
    save(journal, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "commit"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None or args.destination is None:
            parser.error("prepare requires --source and --destination")
        result = prepare(args.source, args.destination, args.journal)
    else:
        result = commit(args.journal)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
