"""Verify Limite export files without importing the model runtime."""

import json
from pathlib import Path

from archlab.artifacts import sha256_file


def verify_export(path):
    """Return the receipt after validating its required files and checksums."""
    root = Path(path)
    receipt = json.loads((root / "EXPORT.json").read_text())
    if receipt.get("format") != "archlab-limite-hf-v1":
        raise ValueError("unknown Limite export format")
    if not {"config.json", "model.safetensors"} <= set(receipt["files"]):
        raise ValueError("incomplete Limite export")
    for name, digest in receipt["files"].items():
        if Path(name).name != name or sha256_file(root / name) != digest:
            raise ValueError("export checksum mismatch: " + name)
    return receipt
