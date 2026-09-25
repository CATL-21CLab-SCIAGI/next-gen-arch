"""Historical execution path; not the supported stock Adam/FP8 baseline.

Use archlab.megatron.miles_v41_stock_launch; see docs/MILES_BASELINE.md.

Launch the paired MiMo math-RL actors inside the qualified GPU namespace.

Run ``head`` on each arm's first node, ``worker`` on its second node, and
``train`` on its head. Paths, addresses and the arm identity are explicit.
"""

import argparse
import json
import os
import runpy
import shlex
import subprocess
import sys
from pathlib import Path


def environment(args):
    cache = args.run_root / f"cache-{args.node}"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "tmp").mkdir(exist_ok=True)
    result = {
        "PATH": str(args.runtime / "site-packages/bin") + ":" + os.environ["PATH"],
        "TMPDIR": "/tmp", "RAY_TMPDIR": "/tmp/ray",
        "RAY_USAGE_STATS_ENABLED": "0", "RAY_ADDRESS": args.address,
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "archlab.serving.sglang",
        "ARCHLAB_SGLANG_MODEL_SHA256": "11ac4bec0aac9e2d094cc38211e1744ee7cdc5d8a9920b6af12d9c92f24a5163",
        "ARCHLAB_MILES_LIVE_WEIGHTS": "1",
        # Miles probes /health_generate before its first policy transfer. Keep
        # that probe readiness-only; the model gate forbids dummy generation.
        "SGLANG_DIAG_BYPASS_HEALTH_GENERATE": "1",
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "0", "SGLANG_OPT_FUSE_WQA_WKV": "0",
        "SGLANG_SHARED_EXPERT_TP1": "1", "SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED": "0",
        "TOKENIZERS_PARALLELISM": "false", "NCCL_CUMEM_ENABLE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "4",
    }
    os.environ.update(result)
    return result


def training_argv(args):
    from miles.utils.external_utils.model_args_utils import load_model_args

    model = args.run_root / f"model-{args.variant}"
    output = args.run_root / f"production-{args.variant}-v1"
    base = shlex.split(load_model_args("deepseek-v4.1"))
    options = {
        "--hf-checkpoint": str(model), "--load": str(model), "--ref-load": str(model), "--save": str(output / "checkpoints"),
        "--model-name": "deepseekv41", "--megatron-to-hf-mode": "raw", "--train-backend": "megatron",
        "--custom-model-provider-path": "archlab.megatron.miles_v41_model.model_provider",
        "--custom-megatron-init-path": "archlab.megatron.miles_v41_init.initialize",
        "--actor-num-nodes": 2, "--actor-num-gpus-per-node": 8, "--num-gpus-per-node": 8,
        "--tensor-model-parallel-size": 8, "--pipeline-model-parallel-size": 2,
        "--expert-model-parallel-size": 8, "--expert-tensor-parallel-size": 1, "--context-parallel-size": 1,
        "--moe-router-dtype": "fp32",
        "--micro-batch-size": 1, "--global-batch-size": 256,
        "--make-vocab-size-divisible-by": 1,
        "--seq-length": 2560, "--max-position-embeddings": 2560, "--qkv-format": "bshd",
        "--dsv4-impl": "miles", "--recompute-granularity": "full",
        "--recompute-method": "uniform", "--recompute-num-layers": 1,
        "--optimizer": "muon", "--lr": "3e-6", "--lr-decay-style": "constant",
        "--lr-warmup-iters": 0, "--weight-decay": 0, "--clip-grad": 1,
        "--adam-beta1": 0.95, "--adam-beta2": 0.95, "--adam-eps": "1e-8",
        "--offload-train-target": "disk", "--offload-train-disk-dir": str(output / "offload"),
        "--offload-train-disk-chunk-mb": 64,
        "--train-env-vars": json.dumps({"MILES_WEIGHT_BACKUP_DIR": str(output / "weight-backup")}),
        "--prompt-data": str(args.run_root / "train.jsonl"), "--input-key": "prompt", "--label-key": "label",
        "--custom-rm-path": "archlab.rl.miles_mimo.reward",
        "--custom-convert-samples-to-train-data-path": "archlab.rl.miles_mimo.convert_samples",
        "--loss-type": "custom_loss", "--custom-loss-function-path": "archlab.rl.miles_mimo.loss",
        "--advantage-estimator": "grpo", "--kl-coef": 0, "--kl-loss-coef": 0, "--entropy-coef": 0,
        "--rollout-batch-size": 16, "--n-samples-per-prompt": 16, "--num-steps-per-rollout": 1,
        "--num-rollout": 128, "--rollout-max-response-len": 2001, "--rollout-max-prompt-len": 559,
        "--rollout-temperature": 1, "--rollout-top-p": 1, "--seed": 20260922,
        "--save-interval": 16, "--eval-interval": 8, "--n-samples-per-eval-prompt": 1,
        "--eval-max-response-len": 2001, "--eval-temperature": 0,
        "--rollout-num-gpus-per-engine": 8, "--sglang-tp-size": 8, "--sglang-ep-size": 8,
        "--sglang-dp-size": 1, "--sglang-device": "cuda", "--sglang-attention-backend": "dsv4", "--sglang-moe-runner-backend": "triton",
        "--sglang-max-running-requests": 16, "--sglang-mem-fraction-static": 0.86,
        "--sglang-dist-timeout": 7200,
        "--distributed-timeout-minutes": 120,
        "--sglang-context-length": 2560, "--sglang-max-total-tokens": 32768,
        "--sglang-chunked-prefill-size": 256, "--sglang-load-format": "dummy",
        "--sglang-json-model-override-args": json.dumps({"architectures": ["DeepseekV4ForCausalLM"]}),
        "--update-weight-buffer-size": 128 * 2**20, "--ckpt-format": "torch_dist",
        "--tensorboard-dir": str(output / "tensorboard"),
    }
    flags = ["--bf16", "--grad-reduce-in-bf16", "--sequence-parallel", "--colocate", "--offload-train", "--offload-rollout", "--use-rollout-logprobs",
             "--disable-rewards-normalization", "--rollout-shuffle",
             "--sglang-disable-cuda-graph", "--sglang-disable-radix-cache", "--sglang-disable-custom-all-reduce",
             "--sglang-disable-shared-experts-fusion", "--sglang-enable-fp32-lm-head", "--no-load-optim",
             "--no-load-rng"]
    return base + [str(x) for pair in options.items() for x in pair] + flags + [
        "--eval-prompt-data", "heldout", str(args.run_root / "heldout-64.jsonl")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["head", "worker", "train", "check"])
    parser.add_argument("--variant", choices=["normal", "simplicial"], required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--miles", type=Path, required=True)
    args = parser.parse_args()
    environment(args)
    if args.action in ("head", "worker"):
        command = [sys.executable, "-m", "ray.scripts.scripts", "start", "--num-gpus=8", "--num-cpus=56",
                   "--object-store-memory=4294967296", "--min-worker-port=30000", "--max-worker-port=30999",
                   f"--node-ip-address={args.node_ip}", "--disable-usage-stats"]
        if args.action == "head":
            command += ["--head", f"--port={args.address.rsplit(':', 1)[1]}", "--include-dashboard=false",
                        f"--temp-dir={os.environ['RAY_TMPDIR']}"]
        else:
            command += [f"--address={args.address}"]
        subprocess.run(command, check=True)
        return
    argv = training_argv(args)
    output = args.run_root / f"production-{args.variant}-v1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "launch-argv.json").write_text(json.dumps(argv, indent=2))
    sys.argv = [str(args.miles / "train.py"), *argv]
    if args.action == "check":
        from miles.utils.arguments import parse_args
        parse_args()
        print("MILES_ARGUMENTS_ACCEPTED", flush=True)
    else:
        import ray
        from miles.ray.specs import inference

        original_engine_env = inference.compute_inference_engine_env_vars

        def engine_env(train_args):
            return original_engine_env(train_args) | {"SGLANG_DIAG_BYPASS_HEALTH_GENERATE": "1"}

        inference.compute_inference_engine_env_vars = engine_env
        ray.init(address=args.address)
        runpy.run_path(str(args.miles / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
