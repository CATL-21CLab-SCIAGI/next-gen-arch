from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from archlab.dlc_controller import (
    DENSE_27B_LAUNCHER,
    LAUNCHER,
    Controller,
    publish_request,
    validate_request,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(
    tmp_path: Path,
    *,
    exit_code: int = 0,
    launcher: str = LAUNCHER,
) -> tuple[Path, str]:
    root = tmp_path / "repos" / "next-gen-arch"
    (root / "scripts").mkdir(parents=True)
    (root / launcher).write_text(f"#!/usr/bin/env bash\nexit {exit_code}\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    return root, _git(root, "rev-parse", "HEAD")


def _payload(
    repository: Path,
    commit: str,
    *,
    launcher: str = LAUNCHER,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "generation": "20260902T200000Z-ca5e75c",
        "action": "run",
        "requested_at_utc": "2026-09-02T12:00:00Z",
        "requested_from": "dsw-evergreen",
        "repository": {"root": str(repository), "commit": commit},
        "launcher": launcher,
        "environment": {
            "NGA_SOURCE_DATA": "/mnt/oss-dataset/datasets/AI-ModelScope/fineweb-edu/sample/100BT",
            "NGA_SOURCE_MANIFEST": "/mnt/oss/datasets/fineweb/source.json",
            "NGA_DATA_ROOT": "/mnt/oss/datasets/fineweb-qwen38",
            "NGA_TOKENIZER": "/mnt/oss/models/qwen38",
            "NGA_OUTPUT_ROOT": "/mnt/nas/evergreen/output",
            "NGA_EXPECTED_NODES": "4",
            "NGA_GPUS_PER_NODE": "8",
            "NGA_TOKENIZER_WORKERS": "8",
            "NGA_EXPECTED_TRAIN_PARTS": "32",
            "NGA_DATA_WAIT_SECONDS": "604800",
            "NGA_PREPARE_DATA": "1",
            "NGA_RUNTIME_PREFLIGHT": "1",
            "NGA_PREFLIGHT_DATA_ROOT": "/mnt/oss/datasets/preflight",
            "NGA_SEQUENCE_LENGTH": "2048",
            "NGA_MICRO_BATCH_SIZE": "1",
            "NGA_GLOBAL_BATCH_SIZE": "512",
            "NGA_TARGET_TRAIN_TOKENS": "100000000000",
            "NGA_CHECKPOINT_INTERVAL_TOKENS": "10000000000",
            "NGA_PRECISION": "bf16",
        },
    }


def test_validate_request_binds_the_full_32_gpu_shape(tmp_path: Path) -> None:
    root = tmp_path / "repos" / "next-gen-arch"
    payload = _payload(root, "a" * 40)
    request = validate_request(
        payload,
        allowed_repo_root=tmp_path / "repos",
        expected_nodes=4,
        expected_gpus_per_node=8,
    )
    assert request.environment["NGA_GLOBAL_BATCH_SIZE"] == "512"
    assert len(request.digest) == 64

    payload["environment"].pop("NGA_PRECISION")  # type: ignore[union-attr]
    legacy_controller_request = validate_request(payload, allowed_repo_root=tmp_path / "repos")
    assert "NGA_PRECISION" not in legacy_controller_request.environment


def test_validate_request_rejects_wrong_source_and_node_count(tmp_path: Path) -> None:
    root = tmp_path / "repos" / "next-gen-arch"
    payload = _payload(root, "a" * 40)
    payload["requested_from"] = "somewhere-else"
    with pytest.raises(ValueError, match="dsw-evergreen"):
        validate_request(payload, allowed_repo_root=tmp_path / "repos")

    payload = _payload(root, "a" * 40)
    payload["environment"]["NGA_EXPECTED_NODES"] = "1"  # type: ignore[index]
    with pytest.raises(ValueError, match="controller owns 4"):
        validate_request(
            payload,
            allowed_repo_root=tmp_path / "repos",
            expected_nodes=4,
            expected_gpus_per_node=8,
        )


def test_validate_request_rejects_unknown_precision(tmp_path: Path) -> None:
    root = tmp_path / "repos" / "next-gen-arch"
    payload = _payload(root, "a" * 40)
    payload["environment"]["NGA_PRECISION"] = "fp16"  # type: ignore[index]

    with pytest.raises(ValueError, match="NGA_PRECISION"):
        validate_request(payload, allowed_repo_root=tmp_path / "repos")


def test_validate_request_accepts_dense_27b_launcher_without_data_conversion_keys(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repos" / "next-gen-arch"
    payload = _payload(root, "a" * 40, launcher=DENSE_27B_LAUNCHER)
    environment = payload["environment"]
    assert isinstance(environment, dict)
    for key in (
        "NGA_SOURCE_DATA",
        "NGA_SOURCE_MANIFEST",
        "NGA_TOKENIZER_WORKERS",
        "NGA_EXPECTED_TRAIN_PARTS",
        "NGA_DATA_WAIT_SECONDS",
        "NGA_PREPARE_DATA",
        "NGA_PRECISION",
    ):
        environment.pop(key)
    environment["NGA_PREFLIGHT_STEPS"] = "400"

    request = validate_request(
        payload,
        allowed_repo_root=tmp_path / "repos",
        expected_nodes=4,
        expected_gpus_per_node=8,
    )

    assert request.payload["launcher"] == DENSE_27B_LAUNCHER
    assert request.environment["NGA_PREFLIGHT_STEPS"] == "400"

    environment["NGA_PREFLIGHT_STEPS"] = "0"
    with pytest.raises(ValueError, match="NGA_PREFLIGHT_STEPS"):
        validate_request(payload, allowed_repo_root=tmp_path / "repos")


def test_publish_is_atomic_idempotent_and_rejects_generation_reuse(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(_payload(repository, commit)))
    control_root = tmp_path / "control"

    first = publish_request(request_file, control_root, tmp_path / "repos")
    second = publish_request(request_file, control_root, tmp_path / "repos")
    assert first["request_sha256"] == second["request_sha256"]
    assert (
        json.loads((control_root / "desired.json").read_text())["generation"] == first["generation"]
    )

    payload = _payload(repository, commit)
    payload["environment"]["NGA_GLOBAL_BATCH_SIZE"] = "1024"  # type: ignore[index]
    request_file.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="different request"):
        publish_request(request_file, control_root, tmp_path / "repos")


def test_publish_rejects_dirty_repository(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)
    (repository / LAUNCHER).write_text("#!/usr/bin/env bash\nexit 9\n")
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(_payload(repository, commit)))
    with pytest.raises(RuntimeError, match="not clean"):
        publish_request(request_file, tmp_path / "control", tmp_path / "repos")


def test_controller_records_child_failure_without_exiting(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path, exit_code=7)
    request = validate_request(
        _payload(repository, commit),
        allowed_repo_root=tmp_path / "repos",
        expected_nodes=4,
        expected_gpus_per_node=8,
    )
    control_root = tmp_path / "control"
    controller = Controller(
        control_root=control_root,
        allowed_repo_root=tmp_path / "repos",
        rank=0,
        world_size=4,
        expected_nodes=4,
        expected_gpus_per_node=8,
        poll_seconds=1,
    )

    controller._run(request)

    result = json.loads(
        (control_root / "runs" / request.generation / "node-00000.json").read_text()
    )
    assert result["exit_code"] == 7
    assert result["succeeded"] is False
    assert controller.stop_requested is False
