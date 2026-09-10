"""Explicit, checksum-verified import of reviewed tokenizer READY parts.

No existing manifest is changed. Only pinned implementations are admitted:
the new renderer extends their accepted domains and leaves successful rows alone.
Payloads are copied (not mutable hardlinks) into the new dataset version.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

LEGACY_IMPLEMENTATION_SHA256 = "8d77664898eceb1813714c6774ffb7484b3b2d2dc5a157c3bbd8fb66fa2162ba"
LEGACY_RENDERER_SHA256 = "24ec8341d94ab955a3d5e80fa553f30135b6eda8d02ac95d237187fa00ba5867"
V3_IMPLEMENTATION_SHA256 = "ecdb44fa506b1450109107315b8309b1c03fd23ff4182729fd514927a79cb19c"
V3_RENDERER_SHA256 = "a6f17b6681c3b30e632f4c38c7fc766ac935d09ff122c373dfee8611dce26178"
POLICY = "deepseek-v41-reviewed-v1-v3-success-domain-unchanged-native-endings-v4"


def validate_legacy_contract(current, legacy):
    signature = (legacy.get("schema_version"), legacy.get("implementation_sha256"),
                 legacy.get("renderer_sha256"), legacy.get("assistant_message_policy"))
    reviewed = {
        (2, LEGACY_IMPLEMENTATION_SHA256, LEGACY_RENDERER_SHA256, None),
        (3, V3_IMPLEMENTATION_SHA256, V3_RENDERER_SHA256, "lossless-assistant-sequences-and-terminal-calls-v2"),
    }
    if (signature not in reviewed or current.get("schema_version") != 4
            or legacy.get("tokenizer_format") != "deepseek-v41"
            or current.get("tokenizer_format") != "deepseek-v41"
            or legacy.get("ending") != "One native BOS; native assistant EOS; no extra EOD"
            or current.get("ending") != "Exact native source ending; no added EOS/EOD; incomplete trajectories flagged"):
        raise ValueError("Only audited DeepSeek V4.1 v1/v3 contracts can be imported")
    allowed = {"schema_version", "implementation_sha256", "renderer_sha256", "assistant_message_policy", "reuse_completed", "ending"}
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
