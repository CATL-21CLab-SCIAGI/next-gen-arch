import json
from pathlib import Path

import pytest

from archlab.architectures.qwen38_27b import Qwen38DenseConfig
from archlab.megatron.qwen38_27b_train import (
    RESUME_IMMUTABLE_FIELDS,
    _attach_resume_contract,
    _megatron_argv,
    _native_muon_contract,
    _parser,
    _record_run_contract,
)


def _contract_payload() -> dict[str, object]:
    payload = {field: {"value": field} for field in RESUME_IMMUTABLE_FIELDS}
    payload["created_at_unix"] = 1.0
    _attach_resume_contract(payload)
    return payload


def test_qwen38_27b_training_defaults_are_long_run_bf16(tmp_path: Path):
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )

    assert args.gdn_kernel == "fla"
    assert args.model_scale == "quarter"
    assert args.micro_batch_size == 4
    assert args.global_batch_size == 512
    assert args.target_train_tokens == 100_000_000_000
    assert args.checkpoint_interval_tokens == 10_000_000_000
    assert args.learning_rate == 5e-5
    assert args.minimum_learning_rate == 5e-6
    assert args.resume is True


def test_qwen38_27b_megatron_argv_uses_dense_geometry_muon_and_speedups(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("WORLD_SIZE", "32")
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )
    config = Qwen38DenseConfig()
    argv = _megatron_argv(args, config)

    expected_pairs = {
        "--num-layers": "16",
        "--hidden-size": "1280",
        "--ffn-hidden-size": "4352",
        # Megatron validates a 64-wide hidden partition; the custom model owns six Q heads.
        "--num-attention-heads": "20",
        "--optimizer": "muon",
        "--muon-momentum": "0.95",
        "--muon-scale-mode": "spectral",
        "--muon-extra-scale-factor": "0.2",
        "--muon-fp32-matmul-prec": "medium",
        "--muon-coefficient-type": "polar_express",
        "--muon-num-ns-steps": "8",
    }
    for flag, value in expected_pairs.items():
        assert argv[argv.index(flag) + 1] == value
    for flag in (
        "--bf16",
        "--muon-nesterov",
        "--use-distributed-optimizer",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
        "--ddp-pad-buckets-for-high-nccl-busbw",
    ):
        assert flag in argv
    assert "--optimizer-cuda-graph" not in argv


def test_qwen38_27b_uses_container_owned_native_muon():
    contract = _native_muon_contract()

    assert contract["fp32_matmul_precision"] == "medium"
    assert "TensorParallelMuon" in str(contract["implementation"])
    assert "no repository-local optimizer adapter" in str(contract["integration"])
    assert contract["optimizer_cuda_graph"] is False


def test_qwen38_27b_full_megatron_argv_uses_released_geometry(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("WORLD_SIZE", "32")
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--run-dir",
            str(tmp_path / "run"),
            "--model-scale",
            "full",
            "--micro-batch-size",
            "1",
        ]
    )
    config = Qwen38DenseConfig.for_scale("full")
    argv = _megatron_argv(args, config)

    assert argv[0] == "qwen38-27b-full-bf16"
    assert argv[argv.index("--num-layers") + 1] == "64"
    assert argv[argv.index("--hidden-size") + 1] == "5120"
    assert argv[argv.index("--ffn-hidden-size") + 1] == "17408"
    assert argv[argv.index("--num-attention-heads") + 1] == "20"


def test_qwen38_27b_probe_does_not_write_a_large_checkpoint(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "32")
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--run-dir",
            str(tmp_path / "run"),
            "--probe-steps",
            "5",
        ]
    )

    argv = _megatron_argv(args, Qwen38DenseConfig())

    assert argv[argv.index("--train-iters") + 1] == "5"
    assert argv[argv.index("--save-interval") + 1] == "6"


def test_resume_contract_retains_attempts_without_overwriting_canonical(tmp_path: Path):
    run_dir = tmp_path / "run"
    payload = _contract_payload()
    _record_run_contract(run_dir, payload, resume=True)
    canonical = (run_dir / "RUN_CONTRACT.json").read_bytes()

    compatible = json.loads(json.dumps(payload))
    compatible["created_at_unix"] = 2.0
    _record_run_contract(run_dir, compatible, resume=True)

    assert (run_dir / "RUN_CONTRACT.json").read_bytes() == canonical
    assert len(list((run_dir / "contracts").glob("attempt-*.json"))) == 2


def test_resume_contract_rejects_immutable_drift_before_checkpoint_load(tmp_path: Path):
    run_dir = tmp_path / "run"
    payload = _contract_payload()
    _record_run_contract(run_dir, payload, resume=True)
    canonical = (run_dir / "RUN_CONTRACT.json").read_bytes()
    checkpoint_marker = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    checkpoint_marker.parent.mkdir(parents=True)
    checkpoint_marker.write_text("10")

    incompatible = json.loads(json.dumps(payload))
    incompatible["model_config"] = {"value": "changed"}
    _attach_resume_contract(incompatible)
    with pytest.raises(RuntimeError, match="model_config"):
        _record_run_contract(run_dir, incompatible, resume=True)

    assert (run_dir / "RUN_CONTRACT.json").read_bytes() == canonical
    assert len(list((run_dir / "contracts").glob("attempt-*.json"))) == 2
