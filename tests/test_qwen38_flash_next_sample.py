from archlab.architectures.qwen38_flash_next_full import Qwen38FlashNextFullConfig
from archlab.megatron.qwen38_flash_next_full_train import _parser
from archlab.megatron.qwen38_flash_next_sample import _sampling_argv


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
