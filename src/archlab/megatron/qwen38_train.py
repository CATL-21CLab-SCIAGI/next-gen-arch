"""Megatron lifecycle adapter for quarter-shape Qwen3.8 pretraining."""

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
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from archlab.architectures.qwen38_flash_next import (
    Qwen38FlashNext,
    Qwen38FlashNextConfig,
)
from archlab.megatron.backend import validate_runtime
from archlab.megatron.qwen38_muon import install_qwen38_muon_adapter, muon_recipe_contract
from archlab.speedrun.precision import resolve_precision_backend

PRECISION_RECIPES = {"bf16": "bf16", "fp4": "fp4_blackwell"}
RUNTIME_BACKENDS = {"bf16": "te_bf16", "fp4": "te_fp4"}


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
        repeat_window_batches: int | None = None,
    ):
        if not prefixes:
            raise ValueError("at least one indexed-data prefix is required")
        self.arrays = [np.memmap(f"{prefix}.bin", mode="r", dtype=np.int32) for prefix in prefixes]
        if any(array.size <= sequence_len for array in self.arrays):
            raise ValueError("each indexed-data part must contain more than one sequence")
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.batch_index = start_batch
        self.start_batch = start_batch
        if repeat_window_batches is not None and repeat_window_batches < 1:
            raise ValueError("repeat_window_batches must be positive")
        self.repeat_window_batches = repeat_window_batches
        self.device = device
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future | None = None
        if self.device.type == "cuda":
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="token-prefetch")
            self._future = self._executor.submit(
                self._cpu_batch, self._source_batch_index(self.batch_index)
            )

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

    def _source_batch_index(self, batch_index: int) -> int:
        if self.repeat_window_batches is None:
            return batch_index
        return self.start_batch + (
            (batch_index - self.start_batch) % self.repeat_window_batches
        )

    def _cpu_batch(self, batch_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        array = self.arrays[batch_index % len(self.arrays)]
        flat_tokens = self.batch_size * self.sequence_len
        start = (batch_index // len(self.arrays)) * flat_tokens
        window = self._cyclic_slice(array, start, flat_tokens + 1)
        # Copy because memmap slices are read-only and then pin for nonblocking H2D.
        tensor = torch.from_numpy(np.array(window, dtype=np.int64, copy=True))
        if self.device.type == "cuda":
            tensor = tensor.pin_memory()
        tokens = tensor[:-1].view(self.batch_size, self.sequence_len)
        labels = tensor[1:].view(self.batch_size, self.sequence_len)
        return tokens, labels

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._future is None:
            tokens, labels = self._cpu_batch(self._source_batch_index(self.batch_index))
        else:
            tokens, labels = self._future.result()
        self.batch_index += 1
        if self._executor is not None:
            self._future = self._executor.submit(
                self._cpu_batch, self._source_batch_index(self.batch_index)
            )
        return {
            "tokens": tokens.to(self.device, non_blocking=True),
            "labels": labels.to(self.device, non_blocking=True),
        }

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def _distributed_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _distributed_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def _partition_prefixes(
    prefixes: list[Path],
    rank: int,
    world_size: int,
    *,
    require_distinct: bool = True,
) -> list[Path]:
    if not prefixes:
        raise ValueError("at least one indexed-data prefix is required")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid rank/world size: {rank}/{world_size}")
    if require_distinct and len(prefixes) < world_size:
        raise ValueError(
            f"training requires at least one distinct indexed-data part per rank: "
            f"parts={len(prefixes)}, world_size={world_size}"
        )
    assigned = prefixes[rank::world_size]
    if assigned:
        return assigned
    return [prefixes[rank % len(prefixes)]]


def _data_prefixes(data_root: Path, split: str) -> list[Path]:
    prefixes = sorted(path.with_suffix("") for path in (data_root / split).glob("part-*.bin"))
    if not prefixes:
        raise FileNotFoundError(f"no {split} part-*.bin files under {data_root}")
    for prefix in prefixes:
        for suffix in (".bin", ".idx", ".json"):
            artifact = Path(f"{prefix}{suffix}")
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                raise FileNotFoundError(f"missing or empty indexed-data artifact: {artifact}")
    return prefixes


def _validated_data_prefixes(data_root: Path) -> tuple[list[Path], list[Path]]:
    """Return artifacts only when DATA_READY declares their exact membership."""
    data_root = data_root.expanduser().resolve()
    ready_path = data_root / "DATA_READY.json"
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid indexed-data manifest: {ready_path}") from error
    if not isinstance(ready, dict):
        raise RuntimeError(f"indexed-data manifest must contain an object: {ready_path}")

    validated = []
    for split, manifest_key in (("train", "train_parts"), ("val", "valid_parts")):
        declared_values = ready.get(manifest_key)
        if not isinstance(declared_values, list) or not declared_values:
            raise RuntimeError(f"indexed-data manifest lacks nonempty {manifest_key}")
        if not all(isinstance(value, str) and value for value in declared_values):
            raise RuntimeError(f"indexed-data manifest has invalid {manifest_key}")
        declared = []
        for value in declared_values:
            prefix = Path(value).expanduser()
            if not prefix.is_absolute():
                prefix = data_root / prefix
            declared.append(prefix.resolve())
        if len(set(declared)) != len(declared):
            raise RuntimeError(f"indexed-data manifest has duplicate {manifest_key}")
        discovered = [prefix.resolve() for prefix in _data_prefixes(data_root, split)]
        if set(declared) != set(discovered):
            missing = sorted(str(path) for path in set(declared) - set(discovered))
            undeclared = sorted(str(path) for path in set(discovered) - set(declared))
            raise RuntimeError(
                f"indexed-data manifest membership changed for {split}: "
                f"missing={missing}, undeclared={undeclared}"
            )
        validated.append(discovered)
    return validated[0], validated[1]


def _current_iteration() -> int:
    from megatron.training import get_args

    args = get_args()
    return int(getattr(args, "curr_iteration", getattr(args, "iteration", 0)))


def _loss_func(
    output_tensor: torch.Tensor,
    component_metrics: dict[str, torch.Tensor] | None = None,
):
    losses = output_tensor.reshape(-1).float()
    count = torch.tensor(losses.numel(), dtype=torch.float32, device=losses.device)
    loss_sum = losses.sum()
    report = {"lm loss": torch.stack((loss_sum.detach(), count))}
    for name, value in (component_metrics or {}).items():
        metric_count = torch.ones((), dtype=torch.float32, device=value.device)
        report[name] = torch.stack((value.float(), metric_count))
    return loss_sum / count, report


def _architecture_from_model(model):
    current = model
    while hasattr(current, "module"):
        current = current.module
    return current.architecture


def _forward_step(data_iterator, model, return_schedule_plan: bool = False):
    if return_schedule_plan:
        raise NotImplementedError("the Qwen3.8 adapter does not use schedule plans")
    batch = next(data_iterator)
    losses = model(batch["tokens"], labels=batch["labels"])
    component_metrics = dict(_architecture_from_model(model).last_loss_metrics)
    return losses, partial(_loss_func, component_metrics=component_metrics)


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
        f"qwen38-quarter-{args.precision}",
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


def _write_contract(args: argparse.Namespace, config: Qwen38FlashNextConfig) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    with torch.device("meta"):
        meta_model = Qwen38FlashNext(config, gdn_kernel=args.gdn_kernel)
        if args.freeze_ngram_tables:
            meta_model.freeze_ngram_tables()
        counts = meta_model.num_scaling_params()
        optimizer_partition = meta_model.optimizer_contract()
    tokens_per_step = args.global_batch_size * config.sequence_len
    train_steps = args.probe_steps or math.ceil(args.target_train_tokens / tokens_per_step)
    save_interval = max(1, round(args.checkpoint_interval_tokens / tokens_per_step))
    runtime = validate_runtime(require_pretrain=False)
    tokenizer_json = args.tokenizer / "tokenizer.json"
    data_ready = args.data_root / "DATA_READY.json"
    precision = (
        {
            "compute": "BF16",
            "recipe": "Megatron BF16 with Transformer Engine BF16 kernels",
            "runtime_backend": "te_bf16",
            "model_parameters": "BF16 under Megatron mixed precision",
            "optimizer_master_parameters": "FP32 distributed optimizer",
            "optimizer_state": "distributed FP32 Muon momentum plus AdamW moments",
            "moe_grouped_token_padding_multiple": 1,
            "moe_grouped_linear_max_experts": config.num_experts,
            "mtp_token_padding_multiple": 1,
        }
        if args.precision == "bf16"
        else {
            "compute": "NVFP4 block scaling",
            "recipe": "Transformer Engine NVFP4BlockScaling",
            "runtime_backend": "te_fp4",
            "stochastic_rounding": True,
            "two_dimensional_quantization": True,
            "moe_grouped_token_padding_multiple": 64,
            "moe_grouped_linear_max_experts": 64,
            "mtp_token_padding_multiple": 8,
            "dense_linear_alignment": 128,
            "bf16_exceptions": [
                "routing logits",
                "shared-expert scalar gate",
                "dense linear weights with K or N not divisible by 128",
                "normalization and loss",
            ],
            "master_parameters": "BF16 under Megatron mixed precision",
            "optimizer_state": "distributed FP32 Muon momentum plus AdamW moments",
        }
    )
    payload = {
        "model": "Qwen3.8-Flash-Next quarter-shape text pretraining",
        "model_config": config.to_dict(),
        "parameter_count": counts,
        "source_model": "Qwen/Qwen3.8-Flash-Next",
        "precision": precision,
        "optimizer": {
            "name": "Qwen3.8 hybrid Muon/AdamW",
            **muon_recipe_contract(),
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "warmup_fraction": args.warmup_fraction,
            "weight_decay": args.weight_decay,
            "clip_grad": args.clip_grad,
            "partition": optimizer_partition,
        },
        "expert_routing": {
            "top_k": config.num_experts_per_token,
            "auxiliary_balance_formula": "E * sum(mean(top-k assignments) * mean(router probabilities))",
            "auxiliary_balance_coefficient": config.router_aux_loss_coefficient,
            "router_z_loss_coefficient": config.router_z_loss_coefficient,
            "source_recipe_difference": (
                "the released full-size Qwen3.8 config uses balance coefficient 1e-3 and "
                "no router z-loss; quarter-scale from-scratch training uses 1e-2 and 1e-3 "
                "after the source values collapsed routing and caused non-finite gradients"
            ),
            "capacity_limit": None,
            "dropped_tokens": False,
            "logged_diagnostics": [
                "expert balance loss",
                "expert load cv",
                "expert max load / mean",
                "router entropy",
            ],
        },
        "qsa_training": {
            "mode": "joint from-scratch sparse-index training",
            "selection_budget": config.indexer_budget,
            "implementation": (
                "auditable PyTorch top-k indexer with gathered sparse K/V attention; "
                "no dense value-attention matrix"
            ),
            "source_recipe_difference": (
                "the published Qwen3.8 model introduced QSA during continued pretraining with "
                "dense-attention distillation; this from-scratch run has no dense teacher checkpoint"
            ),
        },
        "kernel_acceleration": {
            "gdn": args.gdn_kernel,
            "flash_qla_source_commit": os.environ.get("NGA_FLASHQLA_COMMIT"),
            "dense_and_grouped_gemm": "Transformer Engine BF16",
            "cross_entropy": "Transformer Engine Triton fused cross entropy",
            "ple": (
                "GPU hashed lookup, Transformer Engine key/value projections, "
                "cuDNN dilated depthwise convolution"
            ),
            "ngram_table_training": (
                "frozen forward-only root-cause control"
                if args.freeze_ngram_tables
                else "dense Megatron Adam gradients"
            ),
            "optimizer_step": "Megatron whole-step CUDA graph after warmup",
            "gradient_accumulation": "Megatron fusion enabled",
            "input_pipeline": "background CPU memmap prefetch, pinned memory, nonblocking H2D",
            "distributed_overlap": [
                "gradient reduce-scatter overlapped with backward",
                "parameter gather overlap",
                "DP bucket padding for NCCL bandwidth",
            ],
        },
        "speedrun_exclusions": {
            "fp8_or_fp4_compute": "disabled by the explicit 16-bit training request",
            "qsa_fused_training_kernel": (
                "Qwen's production fused sparse-attention/KL kernel is not public in the fresh "
                "container; the 2K run uses the auditable PyTorch QSA path"
            ),
            "whole_model_torch_compile": "dynamic expert token counts and host dispatch are not graph-safe",
            "nccl_user_buffers": "not enabled without a topology-specific registration preflight",
            "sparse_ngram_gradient_communication": (
                "not applicable: hash tables are frozen in this root-cause control; "
                "PLE projections and convolution remain trainable"
                if args.freeze_ngram_tables
                else "modded-nanogpt's custom bigram protocol is model-specific; native "
                "Megatron requires dense gradients for this four-table PLE"
            ),
            "ngram_host_offload": (
                "the quartered 3.2B table fits B300 HBM; direct device lookup avoids host transfer"
            ),
            "normuon_and_cautious_weight_decay": (
                "not substituted for the published Qwen3.8 Polar-Express Muon recipe without "
                "a controlled quality/stability sweep"
            ),
            "batch_or_sequence_ramp": (
                "Qwen3.8 reports a constant large batch outperforming batch-size warmup"
            ),
            "flash_qla": (
                "official SM103 kernel requires key dimension 128; the required quarter shape uses 32"
                if args.gdn_kernel != "flash_qla"
                else None
            ),
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
    train_prefixes, validation_prefixes = _validated_data_prefixes(args.data_root)
    _write_contract(args, config)
    sys.argv = _megatron_argv(args, config)

    import megatron.training.training as training_module
    from megatron.core.enums import ModelType
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.module import MegatronModule
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    install_qwen38_muon_adapter(config)

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
                PRECISION_RECIPES[args.precision],
                device_type="cuda",
                gpu_name=torch.cuda.get_device_name(torch.cuda.current_device()),
                stochastic_rounding="on",
            )
            self.architecture = Qwen38FlashNext(
                config,
                runtime_backend=RUNTIME_BACKENDS[args.precision],
                gdn_kernel=args.gdn_kernel,
            )
            self.architecture.init_weights()
            if args.freeze_ngram_tables:
                self.architecture.freeze_ngram_tables()
            optimizer_partition = self.architecture.optimizer_contract(
                require_two_dimensional_muon=True
            )
            if _distributed_rank() == 0:
                _atomic_json(args.run_dir / "OPTIMIZER_PARTITION.json", optimizer_partition)

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
        train = _partition_prefixes(train_prefixes, rank, world_size)
        validation = _partition_prefixes(
            validation_prefixes,
            rank,
            world_size,
            require_distinct=False,
        )
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
            window_batches = args.eval_iters * accumulation
            yield from BinaryTokenBatches(
                validation,
                batch_size=args.micro_batch_size,
                sequence_len=config.sequence_len,
                start_batch=rank * window_batches,
                device=torch.device("cuda", torch.cuda.current_device()),
                repeat_window_batches=window_batches,
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
    parser.add_argument("--precision", choices=tuple(PRECISION_RECIPES), default="bf16")
    parser.add_argument("--gdn-kernel", choices=("flash_qla", "fla"), default="fla")
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
    parser.add_argument(
        "--freeze-ngram-tables",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="retain PLE forward lookups but exclude the four hash tables from DDP/Adam",
    )
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
