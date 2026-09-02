"""Megatron lifecycle adapter for the quarter-shape Qwen3.8 FP4 pretraining run."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from archlab.architectures.qwen38_flash_next import (
    Qwen38FlashNext,
    Qwen38FlashNextConfig,
)
from archlab.megatron.backend import validate_runtime
from archlab.speedrun.precision import resolve_precision_backend


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class BinaryTokenBatches:
    """Rank-local deterministic iterator over raw int32 Megatron ``.bin`` tokens."""

    def __init__(
        self,
        prefixes: list[Path],
        *,
        batch_size: int,
        sequence_len: int,
        start_batch: int,
        device: torch.device,
    ):
        if not prefixes:
            raise ValueError("at least one indexed-data prefix is required")
        self.arrays = [np.memmap(f"{prefix}.bin", mode="r", dtype=np.int32) for prefix in prefixes]
        if any(array.size <= sequence_len for array in self.arrays):
            raise ValueError("each indexed-data part must contain more than one sequence")
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.batch_index = start_batch
        self.device = device

    @staticmethod
    def _cyclic_slice(array: np.memmap, start: int, length: int) -> np.ndarray:
        start %= array.size
        if start + length <= array.size:
            return np.asarray(array[start : start + length])
        first = np.asarray(array[start:])
        remainder = length - first.size
        chunks = [first]
        while remainder >= array.size:
            chunks.append(np.asarray(array[:]))
            remainder -= array.size
        if remainder:
            chunks.append(np.asarray(array[:remainder]))
        return np.concatenate(chunks)

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        array = self.arrays[self.batch_index % len(self.arrays)]
        flat_tokens = self.batch_size * self.sequence_len
        start = (self.batch_index // len(self.arrays)) * flat_tokens
        window = self._cyclic_slice(array, start, flat_tokens + 1)
        # Copy because memmap slices are read-only and then pin for nonblocking H2D.
        tensor = torch.from_numpy(np.array(window, dtype=np.int64, copy=True))
        if self.device.type == "cuda":
            tensor = tensor.pin_memory()
        tokens = tensor[:-1].view(self.batch_size, self.sequence_len)
        labels = tensor[1:].view(self.batch_size, self.sequence_len)
        self.batch_index += 1
        return {
            "tokens": tokens.to(self.device, non_blocking=True),
            "labels": labels.to(self.device, non_blocking=True),
        }


def _distributed_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _distributed_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def _partition_prefixes(prefixes: list[Path], rank: int, world_size: int) -> list[Path]:
    assigned = prefixes[rank::world_size]
    if assigned:
        return assigned
    return [prefixes[rank % len(prefixes)]]


def _data_prefixes(data_root: Path, split: str) -> list[Path]:
    prefixes = sorted(path.with_suffix("") for path in (data_root / split).glob("part-*.bin"))
    if not prefixes:
        raise FileNotFoundError(f"no {split} part-*.bin files under {data_root}")
    for prefix in prefixes:
        if not Path(f"{prefix}.idx").is_file() or not Path(f"{prefix}.json").is_file():
            raise FileNotFoundError(f"incomplete indexed-data prefix: {prefix}")
    return prefixes


def _current_iteration() -> int:
    from megatron.training import get_args

    args = get_args()
    return int(getattr(args, "curr_iteration", getattr(args, "iteration", 0)))


def _loss_func(output_tensor: torch.Tensor):
    losses = output_tensor.reshape(-1).float()
    count = torch.tensor(losses.numel(), dtype=torch.float32, device=losses.device)
    loss_sum = losses.sum()
    return loss_sum / count, {"lm loss": torch.stack((loss_sum.detach(), count))}


def _forward_step(data_iterator, model, return_schedule_plan: bool = False):
    if return_schedule_plan:
        raise NotImplementedError("the Qwen3.8 adapter does not use schedule plans")
    batch = next(data_iterator)
    losses = model(batch["tokens"], labels=batch["labels"])
    return losses, _loss_func


def _invoke_pretrain(training_module, datasets_provider, model_provider, model_type) -> None:
    parameters = inspect.signature(training_module.pretrain).parameters
    if "cfg_container" in parameters:
        from megatron.training.argument_utils import pretrain_cfg_container_from_args
        from megatron.training.arguments import parse_and_validate_args

        args = parse_and_validate_args(args_defaults={"tokenizer_type": "NullTokenizer"})
        config = pretrain_cfg_container_from_args(args)
        training_module.pretrain(
            config, datasets_provider, model_provider, model_type, _forward_step
        )
        return
    training_module.pretrain(
        datasets_provider,
        model_provider,
        model_type,
        _forward_step,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )


def _megatron_argv(args: argparse.Namespace, config: Qwen38FlashNextConfig) -> list[str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.global_batch_size % (world_size * args.micro_batch_size):
        raise ValueError("global batch must divide by world size times micro batch")
    tokens_per_step = args.global_batch_size * config.sequence_len
    train_steps = args.probe_steps or math.ceil(args.target_train_tokens / tokens_per_step)
    save_interval = max(1, round(args.checkpoint_interval_tokens / tokens_per_step))
    argv = [
        "qwen38-quarter-fp4",
        "--use-mcore-models",
        "--num-layers",
        str(config.num_hidden_layers),
        "--hidden-size",
        str(config.hidden_size),
        "--ffn-hidden-size",
        str(4 * config.hidden_size),
        "--num-attention-heads",
        "10",
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
        "--transformer-impl",
        "local",
        "--optimizer",
        "adam",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--adam-eps",
        "1e-8",
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
        "--no-gradient-accumulation-fusion",
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


def _write_contract(args: argparse.Namespace, config: Qwen38FlashNextConfig) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    with torch.device("meta"):
        counts = Qwen38FlashNext(config).num_scaling_params()
    tokens_per_step = args.global_batch_size * config.sequence_len
    train_steps = args.probe_steps or math.ceil(args.target_train_tokens / tokens_per_step)
    save_interval = max(1, round(args.checkpoint_interval_tokens / tokens_per_step))
    runtime = validate_runtime(require_pretrain=False)
    tokenizer_json = args.tokenizer / "tokenizer.json"
    data_ready = args.data_root / "DATA_READY.json"
    payload = {
        "model": "Qwen3.8-Flash-Next quarter-shape text pretraining",
        "model_config": config.to_dict(),
        "parameter_count": counts,
        "source_model": "Qwen/Qwen3.8-Flash-Next",
        "precision": {
            "compute": "NVFP4 block scaling",
            "recipe": "Transformer Engine NVFP4BlockScaling",
            "stochastic_rounding": True,
            "two_dimensional_quantization": True,
            "bf16_exceptions": [
                "routing logits",
                "shared-expert scalar gate",
                "linear weights with K not divisible by 32 or N not divisible by 16",
                "normalization and loss",
            ],
            "master_parameters": "BF16 under Megatron mixed precision",
            "optimizer_state": "distributed FP32 AdamW",
        },
        "optimizer": {
            "name": "AdamW stabilization recipe",
            "note": "The source model's hybrid Muon/AdamW optimizer is not claimed in this run.",
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "warmup_fraction": args.warmup_fraction,
            "weight_decay": args.weight_decay,
            "clip_grad": args.clip_grad,
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
            "target_tokens": args.target_train_tokens,
            "train_steps": train_steps,
            "effective_tokens": train_steps * tokens_per_step,
            "checkpoint_interval_steps": save_interval,
            "checkpoint_interval_tokens": save_interval * tokens_per_step,
            "probe_steps": args.probe_steps,
        },
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": _sha256(tokenizer_json),
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
    config = Qwen38FlashNextConfig(sequence_len=args.sequence_length)
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

    class Qwen38MegatronModel(MegatronModule):
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
                "fp4_blackwell",
                device_type="cuda",
                gpu_name=torch.cuda.get_device_name(torch.cuda.current_device()),
                stochastic_rounding="on",
            )
            self.architecture = Qwen38FlashNext(config, runtime_backend="te_fp4")
            self.architecture.init_weights()

        def set_input_tensor(self, input_tensor) -> None:
            self.input_tensor = input_tensor
            empty = isinstance(input_tensor, list) and all(item is None for item in input_tensor)
            if input_tensor is not None and not empty:
                raise RuntimeError("Qwen3.8 adapter requires PP=1")

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
            raise ValueError("Qwen3.8 adapter requires TP=PP=CP=1")
        transformer_config = config or core_transformer_config_from_args(get_args())
        return Qwen38MegatronModel(transformer_config, pg_collection)

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
    parser.add_argument("--sequence-length", type=int, default=2_048)
    parser.add_argument("--micro-batch-size", type=int, default=1)
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
        args.tokenizer / "config.json",
    ):
        if not required.is_file():
            raise SystemExit(f"required artifact is missing: {required}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "checkpoints").mkdir(exist_ok=True)
    _run(args)


if __name__ == "__main__":
    main()
