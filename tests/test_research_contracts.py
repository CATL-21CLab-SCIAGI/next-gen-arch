from __future__ import annotations

import json

import pytest
import torch

from archlab.contracts import (
    ComparisonContract,
    ComparisonRegime,
    ContractError,
    assert_paired_controls,
    resolve_training_budget,
)
from archlab.failures import classify_failure
from archlab.performance import ThroughputProtocol, summarize_step_timestamps
from archlab.provenance import (
    ProvenanceError,
    create_dataset_manifest,
    hash_named_tensors,
    verify_dataset_manifest,
    write_dataset_manifest,
)


def test_comparison_contract_requires_an_explicit_regime():
    with pytest.raises(ContractError, match="regime is required"):
        ComparisonContract.from_mapping({})

    contract = ComparisonContract.from_mapping(
        {
            "regime": "controlled",
            "baseline_variant": "baseline",
            "shared_initialization": "bit_identical",
        }
    )
    assert contract.regime is ComparisonRegime.CONTROLLED


def test_budget_regimes_cannot_be_silently_mixed():
    controlled = resolve_training_budget(
        "controlled",
        batch_tokens=128,
        parameter_count=1_000,
        target_train_tokens=1_000,
    )
    assert controlled.steps == 7
    assert controlled.effective_training_tokens == 896

    scaling = resolve_training_budget(
        "scaling",
        batch_tokens=128,
        parameter_count=1_000,
        tokens_per_parameter=12.0,
    )
    assert scaling.steps == 94
    assert scaling.tokens_per_parameter == 12.032

    compute = resolve_training_budget(
        "fixed_compute",
        batch_tokens=128,
        parameter_count=1_000,
        algorithmic_flops_per_token=10.0,
        target_model_flops=10_000.0,
    )
    assert compute.steps == 7
    assert compute.effective_model_flops == 8_960.0

    with pytest.raises(ContractError, match="do not mix budgets"):
        resolve_training_budget(
            "controlled",
            batch_tokens=128,
            parameter_count=1_000,
            target_train_tokens=1_000,
            tokens_per_parameter=12.0,
        )


def test_controlled_pair_rejects_data_or_budget_drift():
    baseline = {
        "seed": 42,
        "dataset_manifest_sha256": "data-a",
        "tokenizer_sha256": "tok",
        "data_order_id": "seed-42",
        "training_tokens": 1_000,
        "sequence_length": 2_048,
        "global_batch_tokens": 128,
        "optimizer_contract_sha256": "opt",
    }
    variant = dict(baseline)
    assert_paired_controls(baseline, variant)
    variant["training_tokens"] = 1_128
    with pytest.raises(ContractError, match="training_tokens"):
        assert_paired_controls(baseline, variant)


def test_dataset_manifest_detects_same_size_content_changes(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "train.bin").write_bytes(b"abcd")
    (root / "val.bin").write_bytes(b"efgh")
    manifest = create_dataset_manifest(
        root,
        dataset="fixture",
        revision="commit-1",
        patterns=("*.bin",),
    )
    path = tmp_path / "dataset.manifest.json"
    write_dataset_manifest(path, manifest)

    metadata = verify_dataset_manifest(root, path, mode="metadata")
    assert metadata["files_verified"] == 2
    assert metadata["content_rehashed"] is False
    verify_dataset_manifest(root, path, mode="full")

    (root / "train.bin").write_bytes(b"wxyz")
    verify_dataset_manifest(root, path, mode="metadata")
    with pytest.raises(ProvenanceError, match="content mismatch"):
        verify_dataset_manifest(root, path, mode="full")


def test_dataset_manifest_rejects_tampered_inventory_hash(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "train.bin").write_bytes(b"data")
    manifest = create_dataset_manifest(root, dataset="fixture", revision="r1")
    path = tmp_path / "manifest.json"
    write_dataset_manifest(path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["inventory_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="inventory_sha256"):
        verify_dataset_manifest(root, path)


def test_initialization_hash_includes_bfloat16_tensor_bytes():
    tensors = [
        ("weight", torch.arange(8, dtype=torch.bfloat16).view(2, 4)),
        ("bias", torch.arange(2, dtype=torch.float32)),
    ]
    first = hash_named_tensors(tensors)
    second = hash_named_tensors(reversed(tensors))
    assert first == second
    tensors[0][1][0, 0] = 7
    assert hash_named_tensors(tensors) != first


def test_throughput_protocol_excludes_warmup_and_limits_window():
    timestamps = [0.0, 10.0, 20.0, 21.0, 22.0, 23.0, 24.0]
    summary = summarize_step_timestamps(
        timestamps,
        tokens_per_step=100,
        protocol=ThroughputProtocol(warmup_steps=2, measurement_steps=3),
    )
    assert summary["throughput_sample_intervals"] == 3
    assert summary["median_step_seconds"] == 1.0
    assert summary["steady_state_tokens_per_second"] == 100.0


@pytest.mark.parametrize(
    ("error", "category", "retriable"),
    (
        (RuntimeError("non-finite gradient"), "numerical", False),
        (RuntimeError("NCCL connection reset"), "operational", True),
        (RuntimeError("CUDA out of memory"), "capacity", False),
        (ValueError("manifest mismatch"), "contract", False),
    ),
)
def test_failure_classification_forbids_blind_numerical_retries(error, category, retriable):
    classification = classify_failure(error)
    assert classification.category == category
    assert classification.retriable is retriable
