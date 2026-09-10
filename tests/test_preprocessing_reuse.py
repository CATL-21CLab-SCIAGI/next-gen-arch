"""Dataset migration tests: reject drift and preserve original payloads/provenance."""

import copy
import json

import pytest

from archlab.preprocessing.nemotron_math import file_sha, json_sha, write_json
from archlab.preprocessing.reuse import (
    LEGACY_IMPLEMENTATION_SHA256,
    LEGACY_RENDERER_SHA256,
    reuse_part,
    validate_legacy_contract,
)


def contracts():
    legacy = dict(schema_version=2, tokenizer_format="deepseek-v41", tokenizer_files={"tokenizer.json": "hash"},
                  split={"seed": "fixed"}, sources=["same"], tasks=["same"],
                  implementation_sha256=LEGACY_IMPLEMENTATION_SHA256, renderer_sha256=LEGACY_RENDERER_SHA256)
    current = {**legacy, "schema_version": 3, "implementation_sha256": "new", "renderer_sha256": "new",
               "assistant_message_policy": "lossless-assistant-sequences-and-terminal-calls-v2"}
    return current, legacy


def test_only_reviewed_extension_is_importable():
    current, legacy = contracts()
    validate_legacy_contract(current, legacy)
    for key, value in (("sources", ["different"]), ("split", {}), ("tokenizer_files", {}),
                       ("tokenizer_format", "qwen")):
        with pytest.raises(ValueError):
            validate_legacy_contract({**current, key: value}, legacy)
    with pytest.raises(ValueError):
        validate_legacy_contract(current, {**legacy, "renderer_sha256": "unreviewed"})
    with pytest.raises(ValueError):
        validate_legacy_contract(current, {**legacy, "schema_version": 3})


def test_reuse_copies_checked_payloads_and_keeps_original_manifests(tmp_path):
    source = tmp_path / "old" / "parts" / "part1"
    local, output = tmp_path / "stage" / "part1", tmp_path / "output" / "part1"
    source.mkdir(parents=True)
    local.mkdir(parents=True)
    task = dict(id="part1", source="source.parquet", rows=1, row_start=0)
    prefix = "train-no-tools_text_document"
    files = []
    for suffix in (".bin", ".idx", ".metadata.jsonl.gz"):
        path = source / (prefix + suffix)
        path.write_bytes(b"data")
        files.append(dict(name=path.name, bytes=4, sha256=file_sha(path)))
    manifest = dict(id="part1", task_sha256=json_sha(task), contract_sha256="old-contract",
                    source=task["source"], row_start=0, documents=1, tokens=1, files=files,
                    partitions={"train-no-tools": dict(prefix=prefix, documents=1, tokens=1)})
    write_json(source / "READY.json", manifest)
    original_marker = (source / "READY.json").read_bytes()
    settings = dict(contract_sha256="new-contract", reuse_completed=dict(
        directory=str(tmp_path / "old"), contract_sha256="old-contract",
        parts={"part1": file_sha(source / "READY.json")},
    ))
    result = reuse_part(task, settings, local, output)
    assert result["contract_sha256"] == "new-contract"
    assert result["reused_from"]["original_manifest"] == manifest
    assert json.loads((output / "READY.json").read_text()) == result
    assert (source / "READY.json").read_bytes() == original_marker
    for item in files:
        assert (output / item["name"]).read_bytes() == (source / item["name"]).read_bytes()
        assert (output / item["name"]).stat().st_ino != (source / item["name"]).stat().st_ino
    # Corruption and changed markers must fail closed, never import silently.
    (source / files[0]["name"]).write_bytes(b"oops")
    with pytest.raises(ValueError, match="checksum"):
        reuse_part(task, settings, local, output)
    changed = copy.deepcopy(manifest)
    changed["documents"] = 2
    write_json(source / "READY.json", changed)
    with pytest.raises(ValueError, match="READY marker changed"):
        reuse_part(task, settings, local, output)
