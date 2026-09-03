"""Megatron lifecycle adapter for quarter-shape Qwen3.8-27B text pretraining."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from functools import partial
from pathlib import Path

import torch

from archlab.architectures.qwen38_27b import (
    SOURCE_CONFIG_SHA256,
    SOURCE_MODEL,
    SOURCE_REVISION,
    TOKENIZER_SHA256,
    Qwen38Dense,
    Qwen38DenseConfig,
)
from archlab.megatron.backend import validate_runtime
from archlab.megatron.qwen38_muon import install_qwen38_muon_adapter, muon_recipe_contract
from archlab.megatron.qwen38_train import (
    BinaryTokenBatches,
    _architecture_from_model,
    _atomic_json,
    _current_iteration,
    _data_prefixes,
    _distributed_rank,
    _distributed_world_size,
    _invoke_pretrain,
    _loss_func,
    _partition_prefixes,
    _sha256,
)
from archlab.speedrun.precision import resolve_precision_backend


def _forward_step(data_iterator, model, return_schedule_plan: bool = False):
    if return_schedule_plan:
        raise NotImplementedError("the Qwen3.8-27B adapter does not use schedule plans")
    batch = next(data_iterator)
    losses = model(batch["tokens"], labels=batch["labels"])
    component_metrics = dict(_architecture_from_model(model).last_loss_metrics)
    return losses, partial(_loss_func, component_metrics=component_metrics)


def _megatron_argv(args: argparse.Namespace, config: Qwen38DenseConfig) -> list[str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.global_batch_size % (world_size * args.micro_batch_size):
        raise ValueError("global batch must divide by world size times micro batch")
    tokens_per_step = args.global_batch_size * config.sequence_len
    train_steps = args.probe_steps or math.ceil(args.target_train_tokens / tokens_per_step)
    save_interval = (
        train_steps + 1
        if args.probe_steps
        else max(1, round(args.checkpoint_interval_tokens / tokens_per_step))
    )
    # Megatron validates hidden_size / num_attention_heads even though this
    # adapter owns its projections.  Twenty is the source-compatible 64-wide
    # hidden partition (1280 / 64); the model itself retains six Q heads.
    megatron_validation_heads = config.hidden_size // config.attention_head_dim
    argv = [
        "qwen38-27b-quarter-bf16",
        "--use-mcore-models",
        "--num-layers",
        str(config.num_hidden_layers),
        "--hidden-size",
        str(config.hidden_size),
        "--ffn-hidden-size",
        str(config.intermediate_size),
        "--num-attention-heads",
        str(megatron_validation_heads),
        "--seq-length",
        str(config.sequence_len),
        "--max-position-embeddings",
        str(config.max_position_embeddings),
        "--position-embedding-type",
        "rope",
        "--normalization",
        "RMSNorm",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
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
        "1",
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--distributed-backend",
        "nccl",
        "--seed",
        str(args.seed),
        "--transformer-impl",
        "local",
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
        "--muon-no-split-qkv",
        "--muon-scale-mode",
        "spectral",
        "--muon-extra-scale-factor",
        "0.2",
        "--muon-fp32-matmul-prec",
        "medium",
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
        "--overlap-grad-reduce",
        "--overlap-param-gather",
        "--ddp-pad-buckets-for-high-nccl-busbw",
        "--optimizer-cuda-graph",
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
        "--tensorboard-dir",
        str(args.run_dir / "tensorboard"),
    ]
    checkpoint_marker = args.run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    if args.resume and checkpoint_marker.is_file():
        argv.extend(("--load", str(args.run_dir / "checkpoints")))
    return argv


def _write_contract(args: argparse.Namespace, config: Qwen38DenseConfig) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    with torch.device("meta"):
        meta_model = Qwen38Dense(config, gdn_kernel=args.gdn_kernel)
        counts = meta_model.num_scaling_params()
        optimizer_partition = meta_model.optimizer_contract()
    tokens_per_step = args.global_batch_size * config.sequence_len
    train_steps = args.probe_steps or math.ceil(args.target_train_tokens / tokens_per_step)
    save_interval = (
        train_steps + 1
        if args.probe_steps
        else max(1, round(args.checkpoint_interval_tokens / tokens_per_step))
    )
    runtime = validate_runtime(require_pretrain=False)
    tokenizer_json = args.tokenizer / "tokenizer.json"
    data_ready = args.data_root / "DATA_READY.json"
    tokenizer_sha256 = _sha256(tokenizer_json)
    if tokenizer_sha256 != TOKENIZER_SHA256:
        raise RuntimeError(f"Qwen3.8-27B tokenizer drift: {tokenizer_sha256} != {TOKENIZER_SHA256}")
    payload = {
        "model": "Qwen3.8-27B quarter-shape text pretraining",
        "model_config": config.to_dict(),
        "parameter_count": counts,
        "source": {
            "model": SOURCE_MODEL,
            "revision": SOURCE_REVISION,
            "config_sha256": SOURCE_CONFIG_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "scope": "text backbone only; the source vision encoder is excluded from FineWeb-Edu pretraining",
        },
        "quarter_scaling": {
            "divided_by_four": [
                "text layers 64 -> 16",
                "hidden width 5120 -> 1280",
                "FFN width 17408 -> 4352",
                "full-attention Q heads 24 -> 6",
                "full-attention KV heads 4 -> 1",
                "full-attention head dimension 256 -> 64",
                "GDN QK heads 16 -> 4",
                "GDN value heads 48 -> 12",
                "GDN key/value head dimensions 128 -> 32",
            ],
            "preserved": [
                "248320-token vocabulary and token IDs",
                "262144-position limit",
                "three GDN layers then one gated full-attention layer",
                "four-token causal convolution kernel",
                "one MTP layer",
            ],
        },
        "precision": {
            "compute": "BF16",
            "recipe": "Megatron BF16 with Transformer Engine BF16 linears",
            "runtime_backend": "te_bf16",
            "model_parameters": "BF16 under Megatron mixed precision",
            "optimizer_master_parameters": "FP32 distributed optimizer",
            "optimizer_state": "distributed FP32 Muon momentum plus AdamW moments",
        },
        "optimizer": {
            "name": "Qwen-style hybrid Muon/AdamW",
            **muon_recipe_contract(),
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "warmup_fraction": args.warmup_fraction,
            "weight_decay": args.weight_decay,
            "clip_grad": args.clip_grad,
            "partition": optimizer_partition,
        },
        "kernel_acceleration": {
            "gated_delta_net": args.gdn_kernel,
            "gated_full_attention": "PyTorch scaled_dot_product_attention causal fused backend",
            "dense_gemm": "Transformer Engine BF16",
            "cross_entropy": "Transformer Engine Triton fused cross entropy",
            "optimizer_step": "Megatron whole-step CUDA graph after warmup",
            "input_pipeline": "background CPU memmap prefetch, pinned memory, nonblocking H2D",
            "distributed_overlap": [
                "gradient reduce-scatter overlapped with backward",
                "parameter gather overlap",
                "DP bucket padding for NCCL bandwidth",
            ],
        },
        "parallelism": {
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "data": int(os.environ.get("WORLD_SIZE", "1")),
            "tensor": 1,
            "pipeline": 1,
            "context": 1,
            "expert": 1,
        },
        "batch": {
            "sequence_length": config.sequence_len,
            "micro_batch_sequences": args.micro_batch_size,
            "global_batch_sequences": args.global_batch_size,
            "tokens_per_step": tokens_per_step,
        },
        "training": {
            "seed": args.seed,
            "target_tokens": args.target_train_tokens,
            "train_steps": train_steps,
            "effective_tokens": train_steps * tokens_per_step,
            "checkpoint_interval_steps": save_interval,
            "checkpoint_interval_tokens": save_interval * tokens_per_step,
            "probe_steps": args.probe_steps,
        },
        "source_commit": os.environ.get("NGA_EXPECTED_COMMIT"),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": tokenizer_sha256,
        "data_root": str(args.data_root),
        "data_ready_sha256": _sha256(data_ready),
        "runtime": runtime,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "created_at_unix": time.time(),
    }
    _atomic_json(args.run_dir / "RUN_CONTRACT.json", payload)


def _run(args: argparse.Namespace) -> None:
    config = Qwen38DenseConfig(sequence_len=args.sequence_length)
    _data_prefixes(args.data_root, "train")
    _data_prefixes(args.data_root, "val")
    _write_contract(args, config)
    sys.argv = _megatron_argv(args, config)

    import megatron.training.training as training_module
    from megatron.core.enums import ModelType
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.module import MegatronModule
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    install_qwen38_muon_adapter(config)

    class Qwen38DenseMegatronModel(MegatronModule):
        def __init__(self, transformer_config, pg_collection):
            super().__init__(transformer_config)
            self.pg_collection = (
                pg_collection
                if pg_collection is not None
                else ProcessGroupCollection.use_mpu_process_groups()
            )
            self.pre_process = True
            self.post_process = True
            self.share_embeddings_and_output_weights = False
            self.input_tensor = None
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            self.precision = resolve_precision_backend(
                "bf16",
                device_type="cuda",
                gpu_name=torch.cuda.get_device_name(torch.cuda.current_device()),
                stochastic_rounding="on",
            )
            self.architecture = Qwen38Dense(
                config,
                runtime_backend="te_bf16",
                gdn_kernel=args.gdn_kernel,
            )
            self.architecture.init_weights()
            optimizer_partition = self.architecture.optimizer_contract(
                require_two_dimensional_muon=True
            )
            if _distributed_rank() == 0:
                _atomic_json(args.run_dir / "OPTIMIZER_PARTITION.json", optimizer_partition)

        def set_input_tensor(self, input_tensor) -> None:
            self.input_tensor = input_tensor
            empty = isinstance(input_tensor, list) and all(item is None for item in input_tensor)
            if input_tensor is not None and not empty:
                raise RuntimeError("Qwen3.8-27B adapter requires PP=1")

        def forward(
            self,
            input_ids,
            position_ids=None,
            attention_mask=None,
            labels=None,
            loss_mask=None,
            packed_seq_params=None,
            **_kwargs,
        ):
            del position_ids, attention_mask, loss_mask, packed_seq_params
            if labels is None:
                raise RuntimeError("labels are required for pretraining")
            with self.precision.training_context():
                return self.architecture(input_ids, labels, loss_reduction="none")

    def model_provider(
        pre_process=True,
        post_process=True,
        vp_stage=None,
        config=None,
        pg_collection=None,
    ):
        if not pre_process or not post_process or vp_stage is not None:
            raise ValueError("Qwen3.8-27B adapter requires TP=PP=CP=1")
        transformer_config = config or core_transformer_config_from_args(get_args())
        return Qwen38DenseMegatronModel(transformer_config, pg_collection)

    def datasets_provider(_sample_counts):
        rank, world_size = _distributed_rank(), _distributed_world_size()
        train = _partition_prefixes(_data_prefixes(args.data_root, "train"), rank, world_size)
        validation = _partition_prefixes(_data_prefixes(args.data_root, "val"), rank, world_size)
        accumulation = args.global_batch_size // (world_size * args.micro_batch_size)

        def train_batches():
            start = _current_iteration() * accumulation
            yield from BinaryTokenBatches(
                train,
                batch_size=args.micro_batch_size,
                sequence_len=config.sequence_len,
                start_batch=start,
                device=torch.device("cuda", torch.cuda.current_device()),
            )

        def validation_batches():
            yield from BinaryTokenBatches(
                validation,
                batch_size=args.micro_batch_size,
                sequence_len=config.sequence_len,
                start_batch=rank * args.eval_iters * accumulation,
                device=torch.device("cuda", torch.cuda.current_device()),
            )

        return train_batches(), validation_batches(), None

    datasets_provider.is_distributed = True
    _invoke_pretrain(
        training_module,
        datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
    )

    if _distributed_rank() == 0:
        _atomic_json(
            args.run_dir
            / ("PROBE_COMPLETE.json" if args.probe_steps else "TRAINING_COMPLETE.json"),
            {"iteration": _current_iteration(), "completed_at_unix": time.time()},
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--gdn-kernel", choices=("fla",), default="fla")
    parser.add_argument("--sequence-length", type=int, default=2_048)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--target-train-tokens", type=int, default=100_000_000_000)
    parser.add_argument("--checkpoint-interval-tokens", type=int, default=10_000_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-fraction", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (
        min(
            args.sequence_length,
            args.micro_batch_size,
            args.global_batch_size,
            args.target_train_tokens,
            args.checkpoint_interval_tokens,
            args.eval_interval,
            args.eval_iters,
            args.log_interval,
        )
        < 1
    ):
        raise SystemExit("all integer training controls must be positive")
    if args.probe_steps < 0:
        raise SystemExit("probe steps must be non-negative")
    for required in (
        args.data_root / "DATA_READY.json",
        args.tokenizer / "tokenizer.json",
    ):
        if not required.is_file():
            raise SystemExit(f"required artifact is missing: {required}")
    ready = json.loads((args.data_root / "DATA_READY.json").read_text())
    if ready.get("tokenizer_sha256") != TOKENIZER_SHA256:
        raise SystemExit("FineWeb-Edu data was not encoded with the Qwen3.8-27B tokenizer")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "checkpoints").mkdir(exist_ok=True)
    _run(args)


if __name__ == "__main__":
    main()
