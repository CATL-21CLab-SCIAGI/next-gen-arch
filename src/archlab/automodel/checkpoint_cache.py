"""Once-per-node, content-addressed staging of verified checkpoint payloads."""

from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from archlab.artifacts import atomic_write_json, sha256_file


def _identity(path):
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)


def stage_checkpoint(source, cache_root):
    """Keep the source immutable; concurrent local ranks share one verified copy.

    The launch must choose node-local storage. A receipt hash identifies the
    cache entry; payload identities detect modification after verification.
    No partially copied entry is ever published or accepted as a resume.
    """
    source, cache_root = Path(source), Path(cache_root)
    receipt_bytes = (source / "COMPLETE.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    key = hashlib.sha256(receipt_bytes).hexdigest()
    for name in receipt["files"]:
        if Path(name).name != name or name in ("COMPLETE.json", "VERIFIED.json"):
            raise ValueError("checkpoint receipt contains an unsafe payload name")
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / key
    with (cache_root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            if (destination / "COMPLETE.json").read_bytes() != receipt_bytes:
                raise ValueError("cached checkpoint receipt changed")
            proof = json.loads((destination / "VERIFIED.json").read_text())
            if proof["receipt_sha256"] != key or set(proof["payloads"]) != set(receipt["files"]):
                raise ValueError("cached checkpoint verification differs from its receipt")
            for name, expected in proof["payloads"].items():
                if _identity(destination / name) != expected:
                    raise ValueError("verified checkpoint payload changed: " + name)
            return destination
        temporary = Path(tempfile.mkdtemp(prefix=key + ".partial-", dir=cache_root))
        try:
            for name, checksum in receipt["files"].items():
                shutil.copyfile(source / name, temporary / name)
                if sha256_file(temporary / name) != checksum:
                    raise ValueError("staged checkpoint checksum mismatch: " + name)
            if (source / "COMPLETE.json").read_bytes() != receipt_bytes:
                raise ValueError("source checkpoint receipt changed during staging")
            (temporary / "COMPLETE.json").write_bytes(receipt_bytes)
            atomic_write_json(temporary / "VERIFIED.json", dict(
                receipt_sha256=key,
                source=str(source),
                payloads={name: _identity(temporary / name) for name in receipt["files"]},
            ))
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return destination
