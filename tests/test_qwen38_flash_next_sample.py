from archlab.architectures.qwen38_flash_next_full import Qwen38FlashNextFullConfig
from archlab.megatron.qwen38_flash_next_full_train import _parser
from archlab.megatron.qwen38_flash_next_sample import _sampling_argv, sampling_config


def test_sampling_disables_training_only_gradient_fusion_and_state_loading(tmp_path):
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--tokenizer",
            str(tmp_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--model-variant",
            "1b-depth48-no-mtp",
            "--parallelism",
            "dp-only",
            "--global-batch-size",
            "1",
            "--probe-steps",
            "1",
        ]
    )
    argv = _sampling_argv(args, Qwen38FlashNextFullConfig.billion_depth48_no_mtp())
    assert "--no-gradient-accumulation-fusion" in argv
    assert "--no-load-optim" in argv
    assert "--no-load-rng" in argv
    for flag in (
        "--tensor-model-parallel-size",
        "--pipeline-model-parallel-size",
        "--expert-model-parallel-size",
        "--context-parallel-size",
    ):
        assert argv[argv.index(flag) + 1] == "1"


def test_width320_sampling_selects_exact_checkpoint_without_repointing_training(tmp_path):
    config = sampling_config("w320-e32-depth48-no-mtp")
    args = _parser().parse_args([
        "--data-root", str(tmp_path), "--tokenizer", str(tmp_path),
        "--run-dir", str(tmp_path / "run"), "--model-variant", "w320-e32-depth48-no-mtp",
        "--parallelism", "dp-only", "--global-batch-size", "1", "--probe-steps", "1",
    ])
    argv = _sampling_argv(args, config, checkpoint_step=1192, attention_backend="unfused")
    assert argv[argv.index("--ckpt-step") + 1] == "1192"
    assert argv[argv.index("--attention-backend") + 1] == "unfused"
    assert argv[argv.index("--hidden-size") + 1] == "320"
    assert "--attention-output-gate" in argv
    assert "--qk-layernorm" in argv
    assert config.num_experts == 32 and config.residual_streams == 4
    assert "--no-load-optim" in argv and "--no-load-rng" in argv
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()


def test_sampling_rejects_unknown_geometry_and_nonpositive_step(tmp_path):
    try:
        sampling_config("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown geometry accepted")
    args = _parser().parse_args([
        "--data-root", str(tmp_path), "--tokenizer", str(tmp_path),
        "--run-dir", str(tmp_path / "run"), "--model-variant", "1b-depth48-no-mtp",
        "--parallelism", "dp-only", "--global-batch-size", "1", "--probe-steps", "1",
    ])
    try:
        _sampling_argv(args, sampling_config("1b-depth48-no-mtp"), checkpoint_step=0)
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive checkpoint step accepted")
