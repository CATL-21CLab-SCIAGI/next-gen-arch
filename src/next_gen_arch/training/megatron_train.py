"""Megatron pretrain-loop adapter for controlled small-scale architecture runs.

This adapter deliberately keeps the architecture math and historical mixed
Muon/Adam policy in this package while delegating distributed initialization,
DDP gradient accumulation, pipeline scheduling, finite checks, and reporting to
the pinned Megatron-LM submodule. It is a comparison adapter, not a claim that
every mechanism has a tensor-parallel-native MCore layer implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from next_gen_arch.backends.megatron import MEGATRON_COMMIT, MEGATRON_ROOT, validate_submodule
from next_gen_arch.training.campaigns import (
    TEN_M_BATCH_TOKENS,
    TEN_M_EVAL_TOKENS,
    TEN_M_GLOBAL_BATCH_SIZE,
    TEN_M_MICRO_BATCH_SIZE,
    TEN_M_SEEDS,
    TEN_M_SEQUENCE_LENGTH,
    TenMVariant,
    get_ten_m_variant,
    ten_m_model_config_kwargs,
)
from next_gen_arch.training.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from next_gen_arch.training.models import (
    build_engram_token_map,
    build_model_config,
    instantiate_model,
)
from next_gen_arch.training.optim import setup_model_optimizer
from next_gen_arch.training.tokenizer import get_token_bytes, get_tokenizer

_EVAL_NATS = 0.0
_EVAL_BYTES = 0
_TOKEN_BYTES: torch.Tensor | None = None


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _claim_run_directory(run_dir: Path, variant: TenMVariant, seed: int) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "RUNNING.json"
    descriptor = {
        "backend": "megatron",
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "variant": variant.name,
        "seed": seed,
        "started_at_unix": time.time(),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor_fd = os.open(marker, flags, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"run directory is already claimed: {run_dir}") from error
    with os.fdopen(descriptor_fd, "w", encoding="utf-8") as handle:
        json.dump(descriptor, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return marker


def _megatron_arguments(variant: TenMVariant) -> list[str]:
    eval_iters = TEN_M_EVAL_TOKENS // TEN_M_BATCH_TOKENS
    return [
        "next-gen-arch-megatron",
        "--use-mcore-models",
        "--num-layers",
        "5",
        "--hidden-size",
        "56",
        "--ffn-hidden-size",
        "224",
        "--num-attention-heads",
        "7",
        "--seq-length",
        str(TEN_M_SEQUENCE_LENGTH),
        "--max-position-embeddings",
        str(TEN_M_SEQUENCE_LENGTH),
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
        str(TEN_M_MICRO_BATCH_SIZE),
        "--global-batch-size",
        str(TEN_M_GLOBAL_BATCH_SIZE),
        "--train-iters",
        str(variant.steps),
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--distributed-backend",
        "nccl",
        "--transformer-impl",
        "local",
        "--optimizer",
        "adam",
        "--lr",
        "0.02",
        "--min-lr",
        "0.001",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.28",
        "--clip-grad",
        "0.0",
        "--bf16",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "32768",
        "--dataloader-type",
        "external",
        "--num-workers",
        "0",
        "--eval-interval",
        str(variant.steps),
        "--eval-iters",
        str(eval_iters),
        "--log-interval",
        "10",
        "--log-throughput",
        "--rerun-mode",
        "disabled",
        "--no-gradient-accumulation-fusion",
        "--no-masked-softmax-fusion",
        "--no-bias-gelu-fusion",
        "--no-bias-swiglu-fusion",
        "--no-bias-dropout-fusion",
        "--no-rope-fusion",
    ]


class SpeedrunSchedule:
    """The historical warmup/warmdown, momentum, and decay schedule."""

    def __init__(self, optimizer, variant: TenMVariant):
        self.optimizer = optimizer
        self.variant = variant
        self.iteration = 0
        self.step_timestamps: list[float] = []
        self._apply(0)

    def _lr_multiplier(self, iteration: int) -> float:
        if iteration < self.variant.warmup_steps:
            return (iteration + 1) / self.variant.warmup_steps
        warmdown = round(0.65 * self.variant.steps)
        if iteration <= self.variant.steps - warmdown:
            return 1.0
        progress = (self.variant.steps - iteration) / warmdown
        return progress + (1.0 - progress) * 0.05

    def _muon_momentum(self, iteration: int) -> float:
        if iteration < 400:
            fraction = iteration / 400
            return (1.0 - fraction) * 0.85 + fraction * 0.97
        warmdown = round(0.65 * self.variant.steps)
        warmdown_start = self.variant.steps - warmdown
        if iteration >= warmdown_start:
            progress = (iteration - warmdown_start) / warmdown
            return 0.97 * (1.0 - progress) + 0.90 * progress
        return 0.97

    def _apply(self, iteration: int) -> None:
        lr_multiplier = self._lr_multiplier(iteration)
        momentum = self._muon_momentum(iteration)
        weight_decay = 0.28 * 0.5 * (1.0 + math.cos(math.pi * iteration / self.variant.steps))
        for group in self.optimizer.param_groups:
            if group.get("dsa_indexer"):
                group["lr"] = group["dsa_sparse_lr"] if iteration >= 40 else group["dsa_warmup_lr"]
            else:
                group["lr"] = group["initial_lr"] * lr_multiplier
            if group["kind"] == "muon":
                group["momentum"] = momentum
                group["weight_decay"] = weight_decay

    def step(self, increment: int) -> None:
        del increment
        self.iteration += 1
        self.step_timestamps.append(time.perf_counter())
        self._apply(min(self.iteration, self.variant.steps))

    def state_dict(self) -> dict[str, int]:
        return {"iteration": self.iteration}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.iteration = int(state_dict["iteration"])
        self._apply(self.iteration)

    def measured_throughput(self) -> tuple[float, float]:
        warm_steps = 10
        if len(self.step_timestamps) <= warm_steps:
            return 0.0, 0.0
        elapsed = self.step_timestamps[-1] - self.step_timestamps[warm_steps - 1]
        measured_steps = len(self.step_timestamps) - warm_steps
        return elapsed, measured_steps * TEN_M_BATCH_TOKENS / elapsed


def _install_optimizer_adapter(variant: TenMVariant) -> dict[str, SpeedrunSchedule]:
    import megatron.training.training as training_module
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    schedule_holder: dict[str, SpeedrunSchedule] = {}

    def setup_model_and_optimizer(model_provider_func, model_type, checkpointing_context=None):
        del checkpointing_context
        args = training_module.get_args()
        timers = training_module.get_timers()
        model = training_module.get_model(model_provider_func, model_type, wrap_with_ddp=True)
        unwrapped = training_module.unwrap_model(model)
        if len(unwrapped) != 1 or not hasattr(unwrapped[0], "architecture"):
            raise RuntimeError("the 10M adapter requires one non-pipelined architecture model")
        architecture = unwrapped[0].architecture
        raw_optimizer = setup_model_optimizer(
            architecture,
            unembedding_lr=0.008,
            embedding_lr=0.3,
            matrix_lr=0.02,
            scalar_lr=0.5,
            weight_decay=0.28,
            distributed=False,
        )
        canonical_group_seen = False
        for group in raw_optimizer.param_groups:
            is_canonical = group["kind"] == "muon" and not canonical_group_seen
            group["default_config"] = is_canonical
            canonical_group_seen = canonical_group_seen or is_canonical
        optimizer_config, _ = training_module.get_megatron_optimizer_config(args)
        optimizer_config.timers = timers
        optimizer_config.clip_grad = 0.0
        optimizer = Float16OptimizerWithFloat16Params(
            raw_optimizer,
            optimizer_config,
            grad_scaler=None,
            init_state_fn=lambda *_args, **_kwargs: None,
        )
        schedule = SpeedrunSchedule(optimizer, variant)
        schedule_holder["schedule"] = schedule
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0
        return model, optimizer, schedule

    training_module.setup_model_and_optimizer = setup_model_and_optimizer
    return schedule_holder


def _external_batch_loader(tokenizer, split: str):
    if split == "train":
        source = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer,
            TEN_M_MICRO_BATCH_SIZE,
            TEN_M_SEQUENCE_LENGTH,
            split="train",
            device="cuda",
        )
        for tokens, labels, _state in source:
            yield {"tokens": tokens, "labels": labels}
    else:
        source = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer,
            TEN_M_MICRO_BATCH_SIZE,
            TEN_M_SEQUENCE_LENGTH,
            split="val",
            device="cuda",
        )
        for tokens, labels in source:
            yield {"tokens": tokens, "labels": labels}


def _loss_func(labels: torch.Tensor, training: bool, output_tensor: torch.Tensor):
    global _EVAL_BYTES, _EVAL_NATS, _TOKEN_BYTES
    losses = output_tensor.reshape(-1).float()
    labels = labels.reshape(-1)
    loss_sum = losses.sum()
    token_count = torch.tensor(labels.numel(), device=labels.device, dtype=torch.int64)
    if _TOKEN_BYTES is None or _TOKEN_BYTES.device != labels.device:
        _TOKEN_BYTES = get_token_bytes(device=labels.device)
    byte_counts = _TOKEN_BYTES[labels]
    nats = (losses * (byte_counts > 0)).sum()
    bytes_sum = byte_counts.sum()
    if not training:
        _EVAL_NATS += float(nats.detach())
        _EVAL_BYTES += int(bytes_sum.detach())
    report = {
        "lm loss": torch.cat(
            (loss_sum.detach().view(1), token_count.to(dtype=torch.float32).view(1))
        ),
        "bpb": torch.cat(
            (
                nats.detach().view(1),
                (bytes_sum * math.log(2.0)).to(dtype=torch.float32).view(1),
            )
        ),
    }
    return loss_sum, token_count, report


def _forward_step(data_iterator, model, return_schedule_plan: bool = False):
    if return_schedule_plan:
        raise NotImplementedError("the comparison adapter does not use schedule plans")
    batch = next(data_iterator)
    tokens = batch["tokens"]
    labels = batch["labels"]
    output_tensor = model(tokens, labels=labels)
    from functools import partial

    return output_tensor, partial(_loss_func, labels, bool(model.training))


def _run_megatron(variant: TenMVariant, seed: int, tokenizer):
    sys.path.insert(0, str(MEGATRON_ROOT))
    sys.argv = _megatron_arguments(variant) + ["--seed", str(seed)]

    from megatron.core.datasets import utils as dataset_utils
    from megatron.core.enums import ModelType
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.module import MegatronModule
    from megatron.training import get_args, pretrain, print_rank_0
    from megatron.training.arguments import core_transformer_config_from_args

    def skip_unused_dataset_helper_build() -> None:
        print_rank_0("> external ClimbMix loader: skipping unused dataset-index helper build")

    dataset_utils.compile_helpers = skip_unused_dataset_helper_build

    model_kwargs = ten_m_model_config_kwargs(variant)

    class ArchitectureMegatronModel(MegatronModule):
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
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model_config = build_model_config(**model_kwargs)
            self.architecture = instantiate_model(model_config)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.architecture.init_weights()
            nonfinite_parameters = [
                name
                for name, parameter in self.architecture.named_parameters()
                if not torch.isfinite(parameter).all()
            ]
            if nonfinite_parameters:
                raise RuntimeError(
                    "non-finite parameters after initialization: "
                    + ", ".join(nonfinite_parameters[:5])
                )
            if variant.name == "engram":
                token_map, compressed_vocab_size = build_engram_token_map(tokenizer, 32_768)
                self.architecture.configure_engram_token_map(
                    token_map, tokenizer.get_bos_token_id()
                )
                print_rank_0(f"Engram compressed vocabulary: {compressed_vocab_size}")
            actual_parameters = self.architecture.num_scaling_params()["total"]
            if actual_parameters != variant.parameter_count:
                raise RuntimeError(
                    f"parameter count drift for {variant.name}: "
                    f"{actual_parameters} != {variant.parameter_count}"
                )

        def set_input_tensor(self, input_tensor) -> None:
            self.input_tensor = input_tensor
            is_empty_pipeline_input = isinstance(input_tensor, list) and all(
                tensor is None for tensor in input_tensor
            )
            if input_tensor is not None and not is_empty_pipeline_input:
                raise RuntimeError("the 10M adapter requires PP=1")

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
                raise RuntimeError("labels are required during the 10M comparison")
            args = get_args()
            if hasattr(self.architecture, "set_training_step"):
                self.architecture.set_training_step(int(args.iteration))
            return self.architecture(
                input_ids,
                labels,
                loss_reduction="none",
            ).view_as(labels)

    def model_provider(
        pre_process=True,
        post_process=True,
        vp_stage=None,
        config=None,
        pg_collection=None,
    ):
        args = get_args()
        if not pre_process or not post_process or vp_stage is not None:
            raise ValueError("the comparison adapter requires TP=PP=CP=1")
        if any(
            value != 1
            for value in (
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.context_parallel_size,
            )
        ):
            raise ValueError("the comparison adapter requires TP=PP=CP=1")
        transformer_config = config or core_transformer_config_from_args(args)
        return ArchitectureMegatronModel(transformer_config, pg_collection)

    def datasets_provider(_sample_counts):
        return (
            _external_batch_loader(tokenizer, "train"),
            _external_batch_loader(tokenizer, "val"),
            None,
        )

    datasets_provider.is_distributed = True
    schedule_holder = _install_optimizer_adapter(variant)
    pretrain(
        datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        _forward_step,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    return schedule_holder["schedule"]


def _environment(variant: TenMVariant, seed: int, mode: str) -> dict[str, Any]:
    repository = _repository_root()
    return {
        "backend": "megatron",
        "support_tier": "mcore_training_wrapper",
        "variant": asdict(variant),
        "seed": seed,
        "mode": mode,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_commit": _git_output(repository, "rev-parse", "HEAD"),
        "megatron_commit": MEGATRON_COMMIT,
        "data_root": os.environ.get("NANOCHAT_BASE_DIR"),
        "triton_ptxas_path": os.environ.get("TRITON_PTXAS_PATH"),
        "semantic_equivalence": (
            "same architecture, ClimbMix packing, tokenizer, seed, batch budget, and "
            "mixed Muon/Adam schedule; MCore owns initialization, DDP accumulation, "
            "finite checks, and evaluation scheduling; bitwise equivalence is not claimed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=TEN_M_SEEDS)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--probe-steps", type=int, default=0)
    args = parser.parse_args()
    if args.probe_steps < 0:
        parser.error("--probe-steps must be non-negative")
    contract_variant = get_ten_m_variant(args.variant)
    variant = (
        replace(contract_variant, steps=args.probe_steps) if args.probe_steps else contract_variant
    )
    mode = "probe" if args.probe_steps else "full"
    run_dir = args.run_dir.expanduser().resolve()
    marker = _claim_run_directory(run_dir, variant, args.seed)
    try:
        submodule = validate_submodule()
        environment = _environment(variant, args.seed, mode)
        environment["submodule"] = submodule
        _write_json(run_dir / "resolved_run.json", environment)
        tokenizer = get_tokenizer()
        started = time.perf_counter()
        schedule = _run_megatron(variant, args.seed, tokenizer)
        wall_seconds = time.perf_counter() - started
        measured_seconds, tokens_per_second = schedule.measured_throughput()
        if _EVAL_BYTES <= 0:
            raise RuntimeError("Megatron completed without a validation BPB denominator")
        final_bpb = _EVAL_NATS / (math.log(2.0) * _EVAL_BYTES)
        if not math.isfinite(final_bpb):
            raise RuntimeError("validation produced a non-finite final BPB")
        result = {
            **environment,
            "status": "complete",
            "parameter_count": variant.parameter_count,
            "training_steps": variant.steps,
            "training_tokens": variant.training_tokens,
            "contract_training_steps": contract_variant.steps,
            "contract_training_tokens": contract_variant.training_tokens,
            "validation_tokens": TEN_M_EVAL_TOKENS,
            "final_bpb": final_bpb,
            "wall_seconds": wall_seconds,
            "measured_training_seconds": measured_seconds,
            "tokens_per_second": tokens_per_second,
            "completed_at_unix": time.time(),
        }
        _write_json(run_dir / "result.json", result)
        marker.replace(run_dir / "COMPLETE.json")
    except BaseException as error:
        _write_json(
            run_dir / "FAILED.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_at_unix": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()
