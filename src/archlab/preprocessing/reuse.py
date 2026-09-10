"""Explicit, checksum-verified import of the reviewed v1 tokenizer's READY parts.

No existing manifest is changed. Only the pinned v1 implementation is admitted:
the new renderer extends its accepted domain and leaves its successful rows alone.
Payloads are copied (not mutable hardlinks) into the new dataset version.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

LEGACY_IMPLEMENTATION_SHA256 = "8d77664898eceb1813714c6774ffb7484b3b2d2dc5a157c3bbd8fb66fa2162ba"
LEGACY_RENDERER_SHA256 = "24ec8341d94ab955a3d5e80fa553f30135b6eda8d02ac95d237187fa00ba5867"
POLICY = "deepseek-v41-v1-success-domain-unchanged-lossless-source-extension-v3"


def validate_legacy_contract(current, legacy):
    if (legacy.get("schema_version") != 2 or current.get("schema_version") != 3
            or legacy.get("tokenizer_format") != "deepseek-v41"
            or current.get("tokenizer_format") != "deepseek-v41"
            or legacy.get("implementation_sha256") != LEGACY_IMPLEMENTATION_SHA256
            or legacy.get("renderer_sha256") != LEGACY_RENDERER_SHA256):
        raise ValueError("Only the audited DeepSeek V4.1 v1 contract can be imported")
    allowed = {"schema_version", "implementation_sha256", "renderer_sha256", "assistant_message_policy", "reuse_completed"}
    if {k: v for k, v in current.items() if k not in allowed} != {k: v for k, v in legacy.items() if k not in allowed}:
        raise ValueError("Legacy source/tokenizer/split/runtime contract is not identical")


def import_contract(current, directory):
    from archlab.preprocessing.nemotron_math import file_sha, json_sha

    root = Path(directory).resolve()
    legacy = json.loads((root / "manifest.json").read_text())
    validate_legacy_contract(current, legacy)
    legacy_hash = json_sha(legacy)
    parts = {}
    for task in current["tasks"]:
        marker = root / "parts" / task["id"] / "READY.json"
        if not marker.exists():
            continue
        part = json.loads(marker.read_text())
        validate_legacy_part(task, part, legacy_hash)
        parts[task["id"]] = file_sha(marker)
    return dict(directory=str(root), contract_sha256=legacy_hash, policy=POLICY,
                implementation_sha256=file_sha(__file__), parts=parts)


def validate_legacy_part(task, part, contract_hash):
    from archlab.preprocessing.nemotron_math import json_sha

    if (part["id"] != task["id"] or part["task_sha256"] != json_sha(task)
            or part["contract_sha256"] != contract_hash or part["documents"] != task["rows"]
            or part["source"] != task["source"] or part["row_start"] != task["row_start"]):
        raise ValueError("Legacy part coverage/provenance mismatch")
    expected_files = set()
    for summary in part["partitions"].values():
        for suffix in (".bin", ".idx", ".metadata.jsonl.gz"):
            expected_files.add(summary["prefix"] + suffix)
    names = [item["name"] for item in part["files"]]
    if (len(names) != len(set(names)) or set(names) != expected_files
            or any(Path(name).name != name or name in ("", ".", "..") for name in names)
            or sum(p["documents"] for p in part["partitions"].values()) != part["documents"]
            or sum(p["tokens"] for p in part["partitions"].values()) != part["tokens"]):
        raise ValueError("Legacy part payload inventory is invalid")


def reuse_part(task, settings, local, destination):
    from archlab.preprocessing.nemotron_math import file_sha, publish_part, verify_part, write_json

    origin = settings["reuse_completed"]
    source = Path(origin["directory"]) / "parts" / task["id"]
    marker = source / "READY.json"
    if file_sha(marker) != origin["parts"][task["id"]]:
        raise ValueError("Legacy READY marker changed since inventory")
    legacy = json.loads(marker.read_text())
    validate_legacy_part(task, legacy, origin["contract_sha256"])
    verify_part(source, legacy)
    for item in legacy["files"]:
        shutil.copyfile(source / item["name"], local / item["name"])
    verify_part(local, legacy)
    manifest = copy.deepcopy(legacy)
    manifest["contract_sha256"] = settings["contract_sha256"]
    manifest["reused_from"] = dict(
        directory=str(source), contract_sha256=origin["contract_sha256"],
        ready_sha256=origin["parts"][task["id"]], policy=POLICY,
        original_manifest=legacy,
    )
    write_json(local / "ORIGIN_READY.json", legacy)
    write_json(local / "READY.json", manifest)
    publish_part(local, destination, manifest)
    return manifest
