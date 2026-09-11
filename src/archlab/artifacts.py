"""Lightweight file identities and atomic metadata publication.

JSON content hashing deliberately remains separate: callers already have
different persisted Unicode/canonicalization contracts.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    if chunk_bytes < 1:
        raise ValueError("hash chunk size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any, *, ensure_ascii: bool = True,
                      allow_nan: bool = True, create_parents: bool = True) -> None:
    """Publish indented, sorted JSON plus LF using a same-directory replacement.

    Preserve legacy serialization defaults; stricter callers explicitly request
    ``allow_nan=False``. Failed serialization/replacement leaves the old target
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
            json.dump(payload, stream, indent=2, sort_keys=True,
                      ensure_ascii=ensure_ascii, allow_nan=allow_nan)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
