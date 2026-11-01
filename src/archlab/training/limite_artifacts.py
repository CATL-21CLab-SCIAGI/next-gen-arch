"""Verify Limite export files without importing the model runtime."""

import json
from pathlib import Path

from safetensors import safe_open

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


def tensor_inventory(path):
    """Map each stored tensor name to its shape and safetensors dtype without reading data."""
    with safe_open(path, framework="pt", device="cpu") as source:
        return {name: (source.get_slice(name).get_shape(), source.get_slice(name).get_dtype())
                for name in source.keys()}
