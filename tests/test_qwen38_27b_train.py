from pathlib import Path

from archlab.architectures.qwen38_27b import Qwen38DenseConfig
from archlab.megatron.qwen38_27b_train import _megatron_argv, _parser


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
    assert args.micro_batch_size == 4
    assert args.global_batch_size == 512
    assert args.target_train_tokens == 100_000_000_000
    assert args.checkpoint_interval_tokens == 10_000_000_000
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
        "--optimizer-cuda-graph",
    ):
        assert flag in argv


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
