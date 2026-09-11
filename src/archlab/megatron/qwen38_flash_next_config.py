"""Flash-Next execution arguments, shared by training and sampling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from archlab.architectures.qwen38_flash_next_full import Qwen38FlashNextFullConfig
from archlab.megatron.simplicial_production import (
    GLOBAL_ATTENTION,
    SIMPLICIAL_ATTENTION,
    attention_variant_contract,
)

TRAIN_STEPS = 11_921
TOKENS_PER_STEP = 8_388_608
EFFECTIVE_TOKENS = TRAIN_STEPS * TOKENS_PER_STEP
CHECKPOINT_INTERVAL_STEPS = 1_192
CHECKPOINT_WRITER_THREADS = 8
DISTRIBUTED_TIMEOUT_MINUTES = 60
NATIVE_MUON_FP32_MATMUL_PRECISION = "medium"
FULL_MODEL_VARIANT = "full"
QUARTER_DEPTH48_NO_MTP_MODEL_VARIANT = "quarter-depth48-no-mtp"
BILLION_DEPTH48_NO_MTP_MODEL_VARIANT = "1b-depth48-no-mtp"
WIDTH320_E32_MODEL_VARIANT = "w320-e32-depth48-no-mtp"
LOSS_NORMALIZATION = "global-valid-token-mean-v1"
ATTENTION_GROUPING = "explicit-gqa-v1"


def _native_muon_contract() -> dict[str, Any]:
    return {
        "implementation": "container-owned megatron.core.optimizer TensorParallelMuon",
        "integration": "Megatron --optimizer muon; no adapter or runtime patch",
        "momentum": 0.95,
        "nesterov": True,
        "coefficient": "polar_express",
        "newton_schulz_steps": 8,
        "scale_mode": "spectral",
        "extra_scale_factor": 0.2,
        "fp32_matmul_precision": NATIVE_MUON_FP32_MATMUL_PRECISION,
        "qkv_split": "native query-group Q/K/V split; coarser than Qwen per-head splitting",
        "released_private_optimizer": "Canzona unavailable",
    }


def _megatron_argv(args: argparse.Namespace, config: Qwen38FlashNextFullConfig) -> list[str]:
    attention_variant_contract(args)
    if config.attention_output_gate and args.parallelism != "dp-only":
        raise ValueError("the width-scaled gated variant requires DP-only execution")
    if args.parallelism == "dp-only" and len(config.pipeline_layers) != 1:
        raise ValueError("DP-only execution requires a single-stage model layout")
    tokens_per_step = args.global_batch_size * config.sequence_len
    if args.global_batch_size < args.micro_batch_size:
        raise ValueError("global batch must be at least one microbatch")
    if args.probe_steps:
        train_steps = args.probe_steps
        save_interval = args.probe_save_interval or args.probe_steps
    else:
        if tokens_per_step != TOKENS_PER_STEP:
            raise ValueError(f"production tokens per step must be {TOKENS_PER_STEP}")
        if args.target_train_tokens != EFFECTIVE_TOKENS:
            raise ValueError(f"production target must be exactly {EFFECTIVE_TOKENS} tokens")
        train_steps = TRAIN_STEPS
        save_interval = CHECKPOINT_INTERVAL_STEPS
    argv = [
        f"qwen38-flash-next-{config.arch_family}-bf16",
        "--use-mcore-models",
        "--num-layers",
        str(config.num_hidden_layers),
        "--hidden-size",
        str(config.hidden_size),
        "--ffn-hidden-size",
        str(config.moe_intermediate_size),
        "--num-attention-heads",
        str(config.attention_heads),
        "--num-query-groups",
        str(config.attention_kv_heads),
        "--group-query-attention",
        "--kv-channels",
        str(config.attention_head_dim),
        "--seq-length",
        str(config.sequence_len),
        "--max-position-embeddings",
        str(config.max_position_embeddings),
        "--position-embedding-type",
        "rope",
        "--rotary-percent",
        str(config.partial_rotary_factor),
        "--rotary-base",
        str(int(config.rope_theta)),
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        str(config.rms_norm_eps),
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--swiglu",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--global-batch-size",
        str(args.global_batch_size),
        "--train-iters",
        str(train_steps),
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        str(len(config.pipeline_layers)),
        "--expert-model-parallel-size",
        "1" if args.parallelism == "dp-only" else "8",
        "--expert-tensor-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--distributed-backend",
        "nccl",
        "--distributed-timeout-minutes",
        str(DISTRIBUTED_TIMEOUT_MINUTES),
        "--transformer-impl",
        "transformer_engine",
        "--num-experts",
        str(config.num_experts),
        "--moe-router-topk",
        str(config.num_experts_per_token),
        "--moe-ffn-hidden-size",
        str(config.moe_intermediate_size),
        "--moe-shared-expert-intermediate-size",
        str(config.shared_expert_intermediate_size),
        "--moe-shared-expert-gate",
        "--moe-router-load-balancing-type",
        "aux_loss",
        "--moe-aux-loss-coeff",
        str(config.router_aux_loss_coefficient),
        "--moe-token-dispatcher-type",
        "alltoall",
        "--optimizer",
        "muon",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--adam-eps",
        "1e-8",
        "--muon-momentum",
        "0.95",
        "--muon-nesterov",
        "--muon-scale-mode",
        "spectral",
        "--muon-extra-scale-factor",
        "0.2",
        "--muon-fp32-matmul-prec",
        NATIVE_MUON_FP32_MATMUL_PRECISION,
        "--muon-coefficient-type",
        "polar_express",
        "--muon-num-ns-steps",
        "8",
        "--muon-scalar-optimizer",
        "adam",
        "--lr",
        str(args.learning_rate),
        "--min-lr",
        str(args.minimum_learning_rate),
        "--lr-decay-style",
        "cosine",
        "--lr-warmup-fraction",
        str(args.warmup_fraction),
        "--weight-decay",
        str(args.weight_decay),
        "--clip-grad",
        str(args.clip_grad),
        "--bf16",
        "--use-distributed-optimizer",
        "--no-use-layer-wise-param-layout",
        "--overlap-grad-reduce",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        str(config.vocab_size),
        "--dataloader-type",
        "external",
        "--num-workers",
        "0",
        "--eval-interval",
        str(args.eval_interval),
        "--eval-iters",
        str(args.eval_iters),
        "--log-interval",
        str(args.log_interval),
        "--log-throughput",
        "--calculate-per-token-loss",
        "--rerun-mode",
        "disabled",
        "--no-masked-softmax-fusion",
        "--no-bias-gelu-fusion",
        "--no-bias-swiglu-fusion",
        "--no-bias-dropout-fusion",
        "--no-rope-fusion",
        "--save",
        str(args.run_dir / "checkpoints"),
        "--save-interval",
        str(save_interval),
        "--ckpt-format",
        "torch_dist",
        "--dist-ckpt-workers",
        str(CHECKPOINT_WRITER_THREADS),
        "--exit-signal-handler",
        "--tensorboard-dir",
        str(args.run_dir / "tensorboard"),
        "--seed",
        str(args.seed),
    ]
    if args.parallelism != "dp-only":
        # Preserve the legacy launch contract. In the frozen runtime, DP-only
        # grouped experts can revisit an already-started parameter-gather bucket
        # in DDP's forward pre-hooks. Native Muon's non-overlapped path gathers
        # updated parameters synchronously after the optimizer step instead.
        argv.append("--overlap-param-gather")
    if config.attention_output_gate:
        argv.append("--attention-output-gate")
    if config.qk_layernorm:
        argv.append("--qk-layernorm")
    if config.zero_centered_gamma:
        argv.append("--apply-layernorm-1p")
    if len(config.pipeline_layers) > 1:
        argv.extend(
            (
                "--decoder-first-pipeline-num-layers",
                str(config.pipeline_layers[0]),
                "--decoder-last-pipeline-num-layers",
                str(config.pipeline_layers[-1]),
            )
        )
    if config.router_z_loss_coefficient:
        argv.extend(("--moe-z-loss-coeff", str(config.router_z_loss_coefficient)))
    if config.mtp_num_layers:
        argv.extend(
            (
                "--mtp-num-layers",
                str(config.mtp_num_layers),
                "--mtp-use-repeated-layer",
                "--mtp-loss-scaling-factor",
                str(config.mtp_loss_scaling_factor),
            )
        )
    if args.fused_moe:
        argv.extend(("--moe-permute-fusion", "--moe-router-fusion"))
    if args.fused_cross_entropy:
        # The frozen runtime forbids the TE implementation for stability reasons.
        argv.extend(("--cross-entropy-loss-fusion", "--cross-entropy-fusion-impl", "native"))
    load_dir = args.load_dir or args.run_dir / "checkpoints"
    marker = load_dir / "latest_checkpointed_iteration.txt"
    if args.load_dir is not None and (not args.resume or not marker.is_file()):
        raise ValueError("an explicit load-dir requires resume and a completed checkpoint marker")
    if args.resume and marker.is_file():
        argv.extend(("--load", str(load_dir)))
        if args.probe_steps:
            # Probe horizons are intentionally short and may grow between the
            # save and reload gates. Keep the checkpoint's optimizer tensors,
            # but use the resumed probe's native scheduler horizon. Production
            # always retains the fixed 11,921-step contract and never overrides.
            argv.append("--override-opt-param-scheduler")
    return argv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Megatron-native trainer for the full Qwen3.8-Flash-Next text variant.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--attention-variant", choices=(GLOBAL_ATTENTION, SIMPLICIAL_ATTENTION),
                        default=GLOBAL_ATTENTION)
    parser.add_argument("--initialization-reference", type=Path,
                        help="Require byte-identical common initial weights from INITIALIZATION.json")
    parser.add_argument(
        "--load-dir", type=Path, help="Native checkpoint root; defaults to run-dir/checkpoints"
    )
    parser.add_argument("--parallelism", choices=("legacy", "dp-only"), default="legacy")
    parser.add_argument("--fused-moe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fused-cross-entropy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--model-variant",
        choices=(
            FULL_MODEL_VARIANT,
            QUARTER_DEPTH48_NO_MTP_MODEL_VARIANT,
            BILLION_DEPTH48_NO_MTP_MODEL_VARIANT,
            WIDTH320_E32_MODEL_VARIANT,
        ),
        default=FULL_MODEL_VARIANT,
    )
    parser.add_argument("--sequence-length", type=int, default=2_048)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=4_096)
    parser.add_argument("--target-train-tokens", type=int, default=EFFECTIVE_TOKENS)
    parser.add_argument("--learning-rate", type=float, default=1.76e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.76e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=1_192)
    parser.add_argument("--eval-iters", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--probe-save-interval", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser
