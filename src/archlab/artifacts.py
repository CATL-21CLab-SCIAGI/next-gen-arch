"""Lightweight file identities and atomic metadata publication.

JSON content hashing deliberately remains separate: callers already have
different persisted Unicode/canonicalization contracts.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


def transfer_plan(source: str | Path, destination: str) -> tuple[list[PurePosixPath], list[tuple[Path, str]]]:
    """Map a file or directory to remote paths before performing any uploads.

    Both sandbox transports preserve empty directories and reject links or
    special files inside directory trees before uploading a partial task.
    """
    source = Path(source)
    if source.is_file():
        target = destination + source.name if destination.endswith("/") else destination
        return [], [(source, target)]
    if not source.is_dir() or source.is_symlink():
        raise ValueError("transfer source must be a file or regular directory")
    entries = sorted(source.rglob("*"))
    if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in entries):
        raise ValueError("directory transfer supports regular files and directories only")
    root = PurePosixPath(destination)
    directories = [root, *(root / path.relative_to(source) for path in entries if path.is_dir())]
    files = [(path, str(root / path.relative_to(source))) for path in entries if path.is_file()]
    return directories, files


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    if chunk_bytes < 1:
        raise ValueError("hash chunk size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any, *, ensure_ascii: bool = True,
                      allow_nan: bool = True, create_parents: bool = True,
                      sort_keys: bool = True) -> None:
    """Publish indented JSON plus LF using a same-directory replacement.

    Preserve legacy serialization defaults; stricter callers explicitly request
    ``allow_nan=False``; insertion-order formats request ``sort_keys=False``.
    Failed serialization/replacement leaves the old target
    intact and removes only this call's temporary file. Unique names also work
    for concurrent writers in the same process.
    """
    path = Path(path)
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=sort_keys,
                      ensure_ascii=ensure_ascii, allow_nan=allow_nan)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
