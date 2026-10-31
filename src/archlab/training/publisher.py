"""Verify native publisher artifacts without converting model weights."""

from __future__ import annotations

import json
import re
from pathlib import Path

from archlab.artifacts import sha256_file


def verify_publisher(root: Path, manifest: Path, *, model_id: str) -> dict:
    """Bind a local native checkpoint to its pinned, verified publisher manifest.

    The manifest is prepared from immutable publisher metadata before launch.
    Rehash the payloads here rather than accepting an earlier readiness receipt.
    Shared weight symlinks are permitted only to their explicitly recorded target.
    """
    root, manifest = Path(root).resolve(), Path(manifest).resolve()
    record = json.loads(manifest.read_text())
    if (record.get("schema_version") != 1 or record.get("model_id") != model_id
            or not re.fullmatch(r"[0-9a-f]{40}", record.get("publisher_revision", ""))):
        raise ValueError("native checkpoint needs a pinned publisher manifest")
    files = record.get("assets", {})
    required = {"config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"}
    if not required <= files.keys():
        raise ValueError("publisher manifest is missing native model/tokenizer assets")
    for name, expected in files.items():
        if (Path(name).name != name or name in {".", ".."}
                or not re.fullmatch(r"[0-9a-f]{64}", expected.get("sha256", ""))):
            raise ValueError("invalid publisher artifact name or hash")
        path = root / name
        if path.is_symlink() and path.resolve() != Path(expected.get("target", "")).resolve():
            raise ValueError("publisher artifact symlink target changed: " + name)
        if (not path.is_file() or path.stat().st_size != expected.get("bytes")
                or sha256_file(path) != expected["sha256"]):
            raise ValueError("publisher artifact bytes changed: " + name)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    if not shards or not shards <= files.keys():
        raise ValueError("publisher manifest does not cover every weight shard")
    if any(Path(name).name != name or not name.endswith(".safetensors") for name in shards):
        raise ValueError("invalid publisher weight shard")
    if any((root / name).exists() for name in ("EXPORT.json", "COMPLETE.json", "rl_state.pt")):
        raise ValueError("native publisher initialization cannot contain experiment state")
    return dict(path=str(root), model_id=model_id, revision=record["publisher_revision"],
                manifest=str(manifest), manifest_sha256=sha256_file(manifest),
                files={name: item["sha256"] for name, item in files.items()},
                shards=len(shards), initialization="native publisher weights; fresh RL state")
