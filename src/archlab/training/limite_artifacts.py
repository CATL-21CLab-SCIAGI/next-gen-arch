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


def verify_trained_insertion(path):
    """Return the receipt of adapter-trained preludes after checking their recorded lineage.

    The weights must descend from a fresh zero-output, adapter-scope insertion export with
    the same prelude geometry, through a checkpoint receipt that is unchanged.
    """
    receipt = verify_export(path)
    insertion = receipt.get("insertion") or {}
    parent, source = receipt.get("parent_export") or {}, receipt.get("source_checkpoint") or {}
    if insertion.get("output_state") != "trained" or not parent or not source.get("receipt"):
        raise ValueError("export lacks trained-insertion lineage")
    if sha256_file(Path(parent["path"]) / "EXPORT.json") != parent["receipt_sha256"]:
        raise ValueError("trained insertion parent receipt changed")
    parent_receipt = verify_export(parent["path"])
    fresh = parent_receipt.get("insertion") or {}
    if (fresh.get("output_state") != "zero" or fresh.get("trainable_mode") != "adapter"
            or parent_receipt["state_sha256"] != parent["state_sha256"]
            or insertion != {**fresh, "output_state": "trained", "trainable_mode": insertion["trainable_mode"]}):
        raise ValueError("trained insertion does not descend from its fresh adapter-scope parent")
    if sha256_file(Path(source["path"]) / source["receipt"]) != source["receipt_sha256"]:
        raise ValueError("trained insertion checkpoint receipt changed")
    return receipt


def tensor_inventory(path):
    """Map each stored tensor name to its shape and safetensors dtype without reading data."""
    with safe_open(path, framework="pt", device="cpu") as source:
        return {name: (source.get_slice(name).get_shape(), source.get_slice(name).get_dtype())
                for name in source.keys()}
