"""Create a verified read-only-input checkpoint cache; never transform weights."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage_checkpoint(source: Path, destination: Path, workers: int = 4, *, resume: bool = False):
    source, destination = source.resolve(), destination.resolve()
    if workers < 1 or destination == source or source in destination.parents:
        raise ValueError("use a new independent cache directory and positive worker count")
    index = source / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    filenames = set(weight_map.values())
    if any(Path(name).name != name for name in filenames):
        raise ValueError("checkpoint index must reference direct child files")
    files = sorted(path for path in source.iterdir() if path.is_file())
    if not filenames.issubset({path.name for path in files}):
        raise ValueError("source checkpoint is incomplete")
    if resume:
        if not destination.is_dir() or (destination / "ARCHLAB_VERIFIED_COPY.json").exists():
            raise ValueError("resume requires an existing unfinished cache")
        if any(p.is_symlink() or not p.is_file() or p.name not in {f.name for f in files}
               for p in destination.iterdir()):
            raise ValueError("refusing to resume a cache with unexpected entries")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()

    def copy_and_verify(path):
        target = destination / path.name
        stat = path.stat()
        digest = sha256(path)
        reusable = resume and target.exists() and target.stat().st_size == stat.st_size and sha256(target) == digest
        if not reusable:
            shutil.copyfile(path, target)
        if stat.st_size != target.stat().st_size or digest != sha256(target):
            raise RuntimeError(f"checkpoint cache verification failed: {path.name}")
        if stat.st_size != path.stat().st_size or stat.st_mtime_ns != path.stat().st_mtime_ns:
            raise RuntimeError(f"source changed during staging: {path.name}")
        return path.name, {"bytes": stat.st_size, "sha256": digest}

    manifest = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(copy_and_verify, path) for path in files]
        for future in as_completed(pending):
            name, evidence = future.result()
            manifest[name] = evidence
            print(json.dumps({"event": "verified_copy", "name": name, **evidence}), flush=True)
    record = {"source": str(source), "destination": str(destination), "files": manifest,
              "source_index_sha256": sha256(index), "seconds": time.monotonic() - start,
              "total_bytes": sum(item["bytes"] for item in manifest.values()),
              "weight_shard_files": len(filenames), "weight_transformation": False}
    # Written last: an interrupted or incomplete copy never has this sentinel.
    with (destination / "ARCHLAB_VERIFIED_COPY.json").open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"event": "checkpoint_cache_complete", **{k: v for k, v in record.items() if k != "files"}}),
          flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="repair only a known unfinished cache, never the source")
    args = parser.parse_args()
    stage_checkpoint(args.source, args.destination, args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
