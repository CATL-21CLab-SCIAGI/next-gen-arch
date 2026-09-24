"""Consume the upstream V4.1 recipe and driver with checkpoint-only hooks."""

import argparse
import json
import os
import runpy
import shlex
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "train"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--miles", type=Path, required=True)
    parser.add_argument("--address", required=True)
    cli = parser.parse_args()
    from scripts import run_deepseek_v41 as recipe

    root = cli.run_root
    captured = {}
    original = recipe.U.execute_train

    def capture(**kwargs):
        captured.update(kwargs)

    recipe.U.execute_train = capture
    extra = [
        "--custom-model-provider-path", "archlab.megatron.miles_v41_model.model_provider",
        "--custom-megatron-init-path", "archlab.megatron.miles_v41_stock_init.initialize",
        "--megatron-to-hf-mode", "raw", "--moe-router-dtype", "fp32",
        "--make-vocab-size-divisible-by", "1", "--global-batch-size", "128",
        "--prompt-data", str(root / "train.jsonl"), "--input-key", "prompt",
        "--save", str(root / "checkpoints"), "--save-interval", "20",
        "--save-trigger-sentinel", str(root / "SAVE_REQUEST"),
        "--stream-optimizer-state-moment-dtype", "bf16",
        "--no-offload-train", "--offload-train-target", "cpu",
        "--sglang-load-format", "dummy", "--sglang-device", "cuda",
        "--sglang-moe-runner-backend", "triton",
        "--sglang-disable-shared-experts-fusion", "--sglang-enable-fp32-lm-head",
        "--sglang-json-model-override-args", json.dumps({"architectures": ["DeepseekV4ForCausalLM"]}),
        "--sglang-context-length", "2560", "--sglang-max-total-tokens", "32768",
        "--sglang-chunked-prefill-size", "1024", "--sglang-dist-timeout", "7200",
        "--distributed-timeout-minutes", "120", "--update-weight-buffer-size", str(128 * 2**20),
        "--save-debug-rollout-data", str(root / "rollout-{rollout_id}.pt"),
        "--tensorboard-dir", str(root / "tensorboard"), "--no-load-optim", "--no-load-rng",
    ]
    try:
        recipe._train(recipe.ScriptArgs(
            run_id="normal-fp8", hf_checkpoint=str(root / "model"),
            model_dir=str(root), data_dir=str(root), save_dir=str(root),
            num_nodes=4, num_gpus_per_node=8, hardware="B300",
            tp_size=8, pp_size=4, ep_size=8, task="dapo_aime",
            num_rollout=64, rollout_batch_size=16, n_samples_per_prompt=8,
            rollout_max_response_len=2001, max_tokens_per_gpu=2560,
            load_from_hf=True, disk_offload=True, offload_disk_dir=str(root / "offload"),
            colocate_memory_peak_device="cpu", recompute="full", grad_reduce_bf16=True,
            rollout_gpus_per_engine=8, sglang_mem_fraction_static=0.95,
            sglang_max_running_requests=16, check_weight_update=False,
            extra_args=shlex.join(extra),
        ))
    finally:
        recipe.U.execute_train = original
    argv = shlex.split(captured["train_args"])
    # The checkpoint's tokenizer prompts are already formatted, with its trained
    # non-thinking prefix. Do not apply the public checkpoint's template again.
    argv.remove("--apply-chat-template")
    index = argv.index("--apply-chat-template-kwargs")
    del argv[index:index + 2]
    env = captured["extra_env_vars"] | {
        "RAY_ADDRESS": cli.address, "RAY_USAGE_STATS_ENABLED": "0",
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "archlab.serving.sglang",
        "ARCHLAB_SGLANG_MODEL_SHA256": "11ac4bec0aac9e2d094cc38211e1744ee7cdc5d8a9920b6af12d9c92f24a5163",
        "ARCHLAB_MILES_LIVE_WEIGHTS": "1", "ARCHLAB_MILES_STOCK_FP8": "1",
        "ARCHLAB_MILES_RESIDENT_POLICY": "1",
        "ARCHLAB_MILES_RUN_ROOT": str(root),
        "ARCHLAB_RL_FREEZE_ENGRAM": "1", "SGLANG_DIAG_BYPASS_HEALTH_GENERATE": "1",
        "SGLANG_SHARED_EXPERT_TP1": "1", "SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED": "0",
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1", "SGLANG_OPT_FP8_WO_A_GEMM": "0",
        "WANDB_MODE": "disabled",
    }
    os.environ.update(env)
    from miles.utils.external_utils.model_args_utils import load_model_args

    argv = shlex.split(load_model_args("deepseek-v4.1")) + argv
    (root / "launch-argv.json").write_text(json.dumps(argv, indent=2))
    sys.argv = [str(cli.miles / "train.py"), *argv]
    if cli.action == "check":
        from miles.utils.arguments import parse_args
        parse_args()
        print("STOCK_MILES_ARGUMENTS_ACCEPTED", flush=True)
    else:
        import ray
        ray.init(address=cli.address, runtime_env={"env_vars": env | {"PYTHONPATH": os.environ["PYTHONPATH"]}})
        runpy.run_path(str(cli.miles / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
