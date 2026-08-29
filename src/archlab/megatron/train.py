"""Megatron pretrain-loop adapter for controlled small-scale architecture runs.

This adapter deliberately keeps the architecture math and historical mixed
Muon/Adam policy in this package while delegating distributed initialization,
DDP gradient accumulation, pipeline scheduling, finite checks, and reporting to
the system-provided Megatron Core runtime. It is a comparison adapter, not a claim
that every mechanism has a tensor-parallel-native MCore layer implementation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import platform
import socket
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from archlab.contracts import (
    BudgetResolution,
    ComparisonRegime,
    ContractError,
    resolve_training_budget,
)
from archlab.failures import classify_failure
from archlab.megatron.backend import validate_runtime
from archlab.optimizers.recipes import (
    OPTIMIZATION_RECIPES,
    OptimizationRecipe,
    get_optimization_recipe,
)
from archlab.optimizers.speedrun import setup_model_optimizer
from archlab.performance import ThroughputProtocol, summarize_step_timestamps
from archlab.provenance import (
    hash_named_tensors,
    hash_tokenizer_vocabulary,
    sha256_file,
    source_provenance,
    stable_json_sha256,
    verify_dataset_manifest,
)
from archlab.speedrun.campaigns import (
    CAMPAIGN_VARIANTS,
    COMPARISON_BATCH_TOKENS,
    COMPARISON_EVAL_TOKENS,
    COMPARISON_GLOBAL_BATCH_SIZE,
    COMPARISON_SEEDS,
    COMPARISON_SEQUENCE_LENGTH,
    FINEWEB_CAMPAIGN_VARIANTS,
    FINEWEB_FIXED_TOKEN_TARGETS,
    FINEWEB_TOKENS_PER_PARAMETER,
    FINEWEB_VOCAB_SIZE,
    HISTORICAL_MICRO_BATCH_SIZE,
    CampaignVariant,
    campaign_model_config_kwargs,
    fineweb_model_config_kwargs,
    get_campaign_variant,
    get_fineweb_variant_template,
)
from archlab.speedrun.dataloader import (
    fineweb_distributed_data_loader,
    fixed_fineweb_validation_loader,
    inspect_fineweb_dataset,
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
    tokenizing_replicated_global_batch_loader_with_state_bos_bestfit,
)
from archlab.speedrun.models import (
    build_engram_token_map,
    build_model_config,
    instantiate_model,
)
from archlab.speedrun.runtime import resolve_climbmix_data_dir
from archlab.speedrun.tokenizer import (
    get_pretrained_tokenizer,
    get_token_bytes,
    get_tokenizer,
    token_bytes_for_tokenizer,
)

_EVAL_NATS = 0.0
_EVAL_BYTES = 0
_TRAIN_LOSS_SUM = 0.0
_TRAIN_TOKENS = 0
_TRAIN_METRICS_ENABLED = False
_TOKEN_BYTES: torch.Tensor | None = None
_RUN_ID = ""
_ATTEMPT_ID = ""
DATASETS = ("climbmix", "fineweb10b", "fineweb100b")
FINEWEB_DATASETS = frozenset({"fineweb10b", "fineweb100b"})
FINEWEB_EXPECTED_TRAIN_SHARDS = {"fineweb10b": 103, "fineweb100b": 1_028}


def _is_fineweb(dataset: str) -> bool:
    return dataset in FINEWEB_DATASETS


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _megatron_resume_iteration() -> int:
    """Read the restored optimizer-step cursor when an external loader first advances."""

    try:
        from megatron.training import get_args

        return int(get_args().iteration)
    except (ImportError, AttributeError):
        return 0


def _append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None or _global_rank() != 0:
        return
    enriched = dict(payload)
    if _RUN_ID:
        enriched.setdefault("run_id", _RUN_ID)
    if _ATTEMPT_ID:
        enriched.setdefault("attempt_id", _ATTEMPT_ID)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(enriched, sort_keys=True) + "\n")


@dataclass(frozen=True)
class MegatronBackendProfile:
    """One explicit, auditable set of Megatron wrapper optimizations."""

    name: str
    compile_architecture: bool
    use_mcore_bf16_master: bool
    finite_checks: bool = True
    compile_mode: str | None = None
    overlap_grad_reduce: bool = False
    ddp_num_buckets: int | None = None
    ddp_average_in_collective: bool = False
    compile_mode_overrides: tuple[tuple[str, str | None], ...] = ()

    def resolved_compile_mode(self, variant: str) -> str | None:
        overrides = dict(self.compile_mode_overrides)
        return overrides[variant] if variant in overrides else self.compile_mode


MEGATRON_BACKEND_PROFILES = {
    profile.name: profile
    for profile in (
        MegatronBackendProfile(
            name="legacy",
            compile_architecture=False,
            use_mcore_bf16_master=True,
        ),
        MegatronBackendProfile(
            name="compile",
            compile_architecture=True,
            use_mcore_bf16_master=True,
        ),
        MegatronBackendProfile(
            name="compile-reduce-overhead",
            compile_architecture=True,
            use_mcore_bf16_master=True,
            compile_mode="reduce-overhead",
        ),
        MegatronBackendProfile(
            name="compile-max-autotune",
            compile_architecture=True,
            use_mcore_bf16_master=True,
            compile_mode="max-autotune",
        ),
        MegatronBackendProfile(
            name="compile-safe-autotune",
            compile_architecture=True,
            use_mcore_bf16_master=True,
            compile_mode="max-autotune",
            compile_mode_overrides=(
                ("kda", None),
                ("kimi-k3-kda-update", None),
                ("qwen-gdn", None),
            ),
        ),
        MegatronBackendProfile(
            name="compile-dp-overlap",
            compile_architecture=True,
            use_mcore_bf16_master=True,
            overlap_grad_reduce=True,
            ddp_num_buckets=4,
        ),
        MegatronBackendProfile(
            name="compile-dp-overlap-average",
            compile_architecture=True,
            use_mcore_bf16_master=True,
            overlap_grad_reduce=True,
            ddp_num_buckets=4,
            ddp_average_in_collective=True,
        ),
        MegatronBackendProfile(
            name="native-master",
            compile_architecture=False,
            use_mcore_bf16_master=False,
        ),
        MegatronBackendProfile(
            name="speedrun",
            compile_architecture=True,
            use_mcore_bf16_master=False,
        ),
    )
}


def get_megatron_backend_profile(name: str) -> MegatronBackendProfile:
    try:
        return MEGATRON_BACKEND_PROFILES[name]
    except KeyError as error:
        choices = ", ".join(MEGATRON_BACKEND_PROFILES)
        raise ValueError(
            f"unknown Megatron backend profile {name!r}; choose one of: {choices}"
        ) from error


def _model_config_kwargs(
    dataset: str,
    scale: str,
    variant: CampaignVariant,
) -> dict[str, Any]:
    if _is_fineweb(dataset):
        return fineweb_model_config_kwargs(scale, variant)
    return campaign_model_config_kwargs(scale, variant)


def _variant_model_metrics(
    dataset: str,
    scale: str,
    template: CampaignVariant,
) -> tuple[int, float, float]:
    config = build_model_config(**_model_config_kwargs(dataset, scale, template))
    with torch.device("meta"):
        model = instantiate_model(config)
    parameter_count = int(model.num_scaling_params()["total"])
    algorithmic_flops = float(model.estimate_flops())
    executed_flops = float(
        model.estimate_executed_flops()
        if hasattr(model, "estimate_executed_flops")
        else algorithmic_flops
    )
    return parameter_count, algorithmic_flops, executed_flops


def resolve_variant_contract(
    dataset: str,
    scale: str,
    name: str,
    *,
    regime: ComparisonRegime | str | None = None,
    target_train_tokens: int | None = None,
    target_model_flops: float | None = None,
    tokens_per_parameter: float | None = None,
) -> tuple[CampaignVariant, BudgetResolution, float, float]:
    """Resolve exact model counts and one explicit comparison budget."""

    if _is_fineweb(dataset):
        template = get_fineweb_variant_template(scale, name)
    else:
        template = get_campaign_variant(scale, name)
    parameter_count, algorithmic_flops, executed_flops = _variant_model_metrics(
        dataset, scale, template
    )
    if not _is_fineweb(dataset) and parameter_count != template.parameter_count:
        raise RuntimeError(
            f"parameter count drift for {name}: {parameter_count} != {template.parameter_count}"
        )
    if regime is None:
        if _is_fineweb(dataset) and scale in FINEWEB_FIXED_TOKEN_TARGETS:
            regime = ComparisonRegime.CONTROLLED
            target_train_tokens = FINEWEB_FIXED_TOKEN_TARGETS[scale]
        elif _is_fineweb(dataset):
            regime = ComparisonRegime.SCALING
            tokens_per_parameter = FINEWEB_TOKENS_PER_PARAMETER
        else:
            # Preserve frozen ClimbMix step counts while labeling their actual
            # tokens/parameter semantics explicitly.
            regime = ComparisonRegime.SCALING
            tokens_per_parameter = template.training_tokens / parameter_count
    budget = resolve_training_budget(
        regime,
        batch_tokens=COMPARISON_BATCH_TOKENS,
        parameter_count=parameter_count,
        algorithmic_flops_per_token=algorithmic_flops,
        target_train_tokens=target_train_tokens,
        target_model_flops=target_model_flops,
        tokens_per_parameter=tokens_per_parameter,
    )
    warmup_steps = min(40, max(1, round(0.05 * budget.steps)))
    variant = replace(
        template,
        parameter_count=parameter_count,
        steps=budget.steps,
        warmup_steps=warmup_steps,
    )
    return variant, budget, algorithmic_flops, executed_flops


def resolve_fineweb_variant(scale: str, name: str) -> CampaignVariant:
    """Compatibility wrapper for the original FineWeb scaling contract."""

    variant, _budget, _algorithmic, _executed = resolve_variant_contract(
        "fineweb100b" if scale == "7b" else "fineweb10b",
        scale,
        name,
    )
    return variant


def _current_training_iteration(args) -> int:
    """Read Megatron's live loop counter, falling back to its resume counter."""
    if hasattr(args, "curr_iteration"):
        return int(args.curr_iteration)
    return int(args.iteration)


def _source_provenance(repository: Path) -> dict[str, Any]:
    return source_provenance(repository)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_checkpoint_artifact(checkpoint_dir: Path, expected_iteration: int) -> dict[str, Any]:
    """Require a complete final Megatron model/optimizer/RNG checkpoint."""

    tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise RuntimeError(f"final checkpoint tracker is missing: {tracker}")
    try:
        iteration = int(tracker.read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise RuntimeError(f"invalid final checkpoint tracker: {tracker}") from error
    if iteration != expected_iteration:
        raise RuntimeError(
            f"final checkpoint iteration mismatch: {iteration} != {expected_iteration}"
        )
    candidates = sorted(checkpoint_dir.glob(f"iter_{iteration:07d}/**/model_optim_rng.pt"))
    if not candidates:
        candidates = sorted(checkpoint_dir.glob(f"iter_{iteration:07d}/**/*.pt"))
    if not candidates:
        raise RuntimeError(f"final checkpoint payload is missing for iteration {iteration}")
    return {
        "iteration": iteration,
        "tracker": str(tracker),
        "payload_files": len(candidates),
        "payload_bytes": sum(path.stat().st_size for path in candidates),
        "payload_paths": [str(path) for path in candidates],
        "resume_contract": "optimizer-rng-plus-iteration-derived-loader-cursor-v2",
    }


def _global_rank() -> int:
    """Read torchrun's global rank before Megatron initializes process groups."""
    return int(os.environ.get("RANK", "0"))


def _reduce_validation_totals() -> tuple[float, int]:
    """Aggregate sharded validation numerators across data-parallel ranks."""
    totals = torch.tensor(
        (_EVAL_NATS, float(_EVAL_BYTES)),
        device=torch.cuda.current_device(),
        dtype=torch.float64,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    return float(totals[0].item()), int(totals[1].item())


def _claim_run_directory(
    run_dir: Path,
    variant: CampaignVariant,
    seed: int,
    *,
    attempt_id: str,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "RUNNING.json"
    descriptor = {
        "backend": "megatron",
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "variant": variant.name,
        "seed": seed,
        "attempt_id": attempt_id,
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


def _megatron_arguments(
    variant: CampaignVariant,
    profile: MegatronBackendProfile,
    recipe: OptimizationRecipe,
    *,
    dataset: str = "climbmix",
    scale: str = "10m",
    exact_global_batch_replay: bool = False,
    micro_batch_size_override: int | None = None,
    checkpoint_dir: Path | None = None,
    save_interval: int | None = None,
    resume: bool = False,
) -> list[str]:
    model_kwargs = _model_config_kwargs(dataset, scale, variant)
    head_dim = int(model_kwargs["head_dim"])
    hidden_size = (
        (int(model_kwargs["depth"]) * int(model_kwargs["aspect_ratio"]) + head_dim - 1)
        // head_dim
        * head_dim
    )
    num_attention_heads = hidden_size // head_dim
    if _is_fineweb(dataset):
        if COMPARISON_GLOBAL_BATCH_SIZE % _world_size():
            raise ValueError(
                "FineWeb DP requires the 192-sequence global batch to be divisible by world size"
            )
        rank_batch_size = COMPARISON_GLOBAL_BATCH_SIZE // _world_size()
        micro_batch_size = (
            micro_batch_size_override if micro_batch_size_override is not None else rank_batch_size
        )
        if micro_batch_size <= 0:
            raise ValueError("micro batch size must be positive")
        if COMPARISON_GLOBAL_BATCH_SIZE % (micro_batch_size * _world_size()):
            raise ValueError(
                "FineWeb global batch must be divisible by micro batch size times world size"
            )
        scheduled_global_batch_size = COMPARISON_GLOBAL_BATCH_SIZE
    elif micro_batch_size_override is not None:
        raise ValueError("micro batch override is currently supported only for FineWeb")
    elif exact_global_batch_replay:
        micro_batch_size = math.ceil(COMPARISON_GLOBAL_BATCH_SIZE / _world_size())
        scheduled_global_batch_size = micro_batch_size * _world_size()
        if profile.ddp_average_in_collective:
            raise ValueError(
                "exact per-token batch replay is incompatible with DDP average-in-collective"
            )
    else:
        micro_batch_size = HISTORICAL_MICRO_BATCH_SIZE
        scheduled_global_batch_size = COMPARISON_GLOBAL_BATCH_SIZE
    eval_iters = COMPARISON_EVAL_TOKENS // COMPARISON_BATCH_TOKENS
    eval_interval = (
        min(250, variant.steps)
        if _is_fineweb(dataset)
        else (250 if scale == "100m" else variant.steps)
    )
    arguments = [
        "next-gen-arch-megatron",
        "--use-mcore-models",
        "--num-layers",
        str(model_kwargs["depth"]),
        "--hidden-size",
        str(hidden_size),
        "--ffn-hidden-size",
        str(4 * hidden_size),
        "--num-attention-heads",
        str(num_attention_heads),
        "--seq-length",
        str(COMPARISON_SEQUENCE_LENGTH),
        "--max-position-embeddings",
        str(COMPARISON_SEQUENCE_LENGTH),
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
        str(micro_batch_size),
        "--global-batch-size",
        str(scheduled_global_batch_size),
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
        str(recipe.gradient_clip),
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        str(model_kwargs["vocab_size"]),
        "--dataloader-type",
        "external",
        "--num-workers",
        "0",
        "--eval-interval",
        str(eval_interval),
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
    if exact_global_batch_replay:
        arguments.append("--calculate-per-token-loss")
    if profile.use_mcore_bf16_master:
        arguments.append("--bf16")
    if profile.overlap_grad_reduce:
        arguments.append("--overlap-grad-reduce")
    if profile.ddp_num_buckets is not None:
        arguments.extend(("--ddp-num-buckets", str(profile.ddp_num_buckets)))
    if profile.ddp_average_in_collective:
        arguments.append("--ddp-average-in-collective")
    # This Megatron revision enables finite loss/gradient checks by default and
    # exposes only the negative CLI form.  Spell out the opt-out when a future
    # diagnostic profile needs it; never pass a nonexistent positive flag.
    if not profile.finite_checks:
        arguments.append("--no-check-for-nan-in-loss-and-grad")
    if checkpoint_dir is not None:
        if save_interval is None or save_interval < 1:
            raise ValueError("checkpointing requires a positive save interval")
        arguments.extend(
            (
                "--save",
                str(checkpoint_dir),
                "--save-interval",
                str(save_interval),
                "--ckpt-format",
                # Per-head NorMuon stacks same-shaped parameter states. MCore's
                # torch_dist optimizer serializer assumes one state tensor has
                # exactly one model-parameter shape; the legacy torch format
                # preserves the stacked optimizer state without that invalid
                # assumption. DP stores one full checkpoint copy.
                "torch",
            )
        )
        if resume:
            arguments.extend(("--load", str(checkpoint_dir)))
    elif save_interval is not None:
        raise ValueError("save interval requires a checkpoint directory")
    elif resume:
        raise ValueError("resume requires a checkpoint directory")
    return arguments


class SpeedrunSchedule:
    """The historical warmup/warmdown, momentum, and decay schedule."""

    def __init__(
        self,
        optimizer,
        variant: CampaignVariant,
        *,
        metrics_path: Path | None = None,
        metrics_every: int = 10,
        throughput_protocol: ThroughputProtocol | None = None,
    ):
        self.optimizer = optimizer
        self.variant = variant
        self.metrics_path = metrics_path
        self.metrics_every = metrics_every
        self.throughput_protocol = (
            ThroughputProtocol() if throughput_protocol is None else throughput_protocol
        )
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
        global _TRAIN_LOSS_SUM, _TRAIN_TOKENS
        del increment
        self.iteration += 1
        self.step_timestamps.append(time.perf_counter())
        self._apply(min(self.iteration, self.variant.steps))
        if self.metrics_path is not None:
            totals = torch.tensor(
                (_TRAIN_LOSS_SUM, float(_TRAIN_TOKENS)),
                device=torch.cuda.current_device(),
                dtype=torch.float64,
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
            if self.iteration % self.metrics_every == 0 and totals[1] > 0:
                _append_jsonl(
                    self.metrics_path,
                    {
                        "kind": "train",
                        "step": self.iteration,
                        "tokens": self.iteration * COMPARISON_BATCH_TOKENS,
                        "loss": float((totals[0] / totals[1]).item()),
                        "lr_multiplier": self._lr_multiplier(self.iteration),
                    },
                )
            _TRAIN_LOSS_SUM = 0.0
            _TRAIN_TOKENS = 0

    def state_dict(self) -> dict[str, int]:
        return {"iteration": self.iteration}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.iteration = int(state_dict["iteration"])
        self._apply(self.iteration)

    def measured_throughput(self) -> tuple[float, float]:
        summary = self.throughput_summary()
        return summary["measured_training_seconds"], summary["tokens_per_second"]

    def throughput_summary(self) -> dict[str, Any]:
        """Report a predeclared steady-state timing window."""

        return summarize_step_timestamps(
            self.step_timestamps,
            tokens_per_step=COMPARISON_BATCH_TOKENS,
            protocol=self.throughput_protocol,
        )


def _install_optimizer_adapter(
    variant: CampaignVariant,
    profile: MegatronBackendProfile,
    recipe: OptimizationRecipe,
    *,
    metrics_path: Path | None = None,
    metrics_every: int = 10,
    throughput_protocol: ThroughputProtocol | None = None,
) -> dict[str, SpeedrunSchedule]:
    import megatron.training.training as training_module
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer

    schedule_holder: dict[str, SpeedrunSchedule] = {}

    def setup_model_and_optimizer(model_provider_func, model_type, checkpointing_context=None):
        del checkpointing_context
        args = training_module.get_args()
        timers = training_module.get_timers()
        model = training_module.get_model(model_provider_func, model_type, wrap_with_ddp=True)
        unwrapped = training_module.unwrap_model(model)
        if len(unwrapped) != 1 or not hasattr(unwrapped[0], "architecture"):
            raise RuntimeError(
                "the comparison adapter requires one non-pipelined architecture model"
            )
        architecture = unwrapped[0].architecture
        optimizer_model = getattr(architecture, "_orig_mod", architecture)
        raw_optimizer = setup_model_optimizer(
            optimizer_model,
            unembedding_lr=recipe.unembedding_lr,
            embedding_lr=recipe.embedding_lr,
            matrix_lr=recipe.matrix_lr,
            scalar_lr=recipe.scalar_lr,
            weight_decay=0.28,
            distributed=False,
            matrix_optimizer=recipe.matrix_optimizer,
            adam_update_every=recipe.adam_update_every,
            cautious_adam_weight_decay=recipe.cautious_adam_weight_decay,
        )
        canonical_group_seen = False
        for group in raw_optimizer.param_groups:
            is_canonical = group["kind"] in {"muon", "adamh"} and not canonical_group_seen
            group["default_config"] = is_canonical
            canonical_group_seen = canonical_group_seen or is_canonical
        optimizer_config, _ = training_module.get_megatron_optimizer_config(args)
        optimizer_config.timers = timers
        optimizer_config.clip_grad = recipe.gradient_clip
        if profile.use_mcore_bf16_master:
            optimizer = Float16OptimizerWithFloat16Params(
                raw_optimizer,
                optimizer_config,
                grad_scaler=None,
                init_state_fn=lambda *_args, **_kwargs: None,
            )
        else:
            # Preserve the speedrun's mixed storage policy: projection masters stay
            # FP32 while embeddings may stay BF16, and Linear casts only for matmul.
            # MCore still owns DDP main-grad buffers and the optimizer lifecycle, but
            # there is no detached FP32 replica or per-step copy-back.
            optimizer = FP32Optimizer(
                raw_optimizer,
                optimizer_config,
                init_state_fn=lambda *_args, **_kwargs: None,
            )
        schedule = SpeedrunSchedule(
            optimizer,
            variant,
            metrics_path=metrics_path,
            metrics_every=metrics_every,
            throughput_protocol=throughput_protocol,
        )
        schedule_holder["schedule"] = schedule
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0
        return model, optimizer, schedule

    training_module.setup_model_and_optimizer = setup_model_and_optimizer
    return schedule_holder


def _external_batch_loader(
    tokenizer,
    split: str,
    *,
    dataset: str = "climbmix",
    data_root: Path | None = None,
    exact_global_batch_replay: bool = False,
    micro_batch_size_override: int | None = None,
):
    if _is_fineweb(dataset):
        if data_root is None:
            raise ValueError("FineWeb requires an explicit data root")
        if COMPARISON_GLOBAL_BATCH_SIZE % _world_size():
            raise ValueError("FineWeb global batch is not divisible by DP world size")
        rank_batch_size = COMPARISON_GLOBAL_BATCH_SIZE // _world_size()
        local_batch_size = (
            micro_batch_size_override if micro_batch_size_override is not None else rank_batch_size
        )
        if COMPARISON_GLOBAL_BATCH_SIZE % (local_batch_size * _world_size()):
            raise ValueError(
                "FineWeb global batch must be divisible by micro batch size times world size"
            )
        accumulation_microbatches = COMPARISON_GLOBAL_BATCH_SIZE // (
            local_batch_size * _world_size()
        )
        if split == "train":
            source = fineweb_distributed_data_loader(
                data_root,
                split,
                local_batch_size,
                COMPARISON_SEQUENCE_LENGTH,
                device="cuda",
                start_batch_index=_megatron_resume_iteration() * accumulation_microbatches,
            )
        else:
            evaluation_iterations = COMPARISON_EVAL_TOKENS // COMPARISON_BATCH_TOKENS
            source = fixed_fineweb_validation_loader(
                data_root,
                local_batch_size,
                COMPARISON_SEQUENCE_LENGTH,
                window_batches=evaluation_iterations * accumulation_microbatches,
                device="cuda",
            )
        for tokens, labels in source:
            yield {"tokens": tokens, "labels": labels}
        return

    if exact_global_batch_replay:
        if split == "train":
            resume_iteration = _megatron_resume_iteration()
            source = tokenizing_replicated_global_batch_loader_with_state_bos_bestfit(
                tokenizer,
                COMPARISON_GLOBAL_BATCH_SIZE,
                COMPARISON_SEQUENCE_LENGTH,
                split="train",
                device="cuda",
                resume_state_dict=({"batch_index": resume_iteration} if resume_iteration else None),
            )
            for tokens, labels, _state, _active in source:
                yield {"tokens": tokens, "labels": labels}
            return

        eval_batches = COMPARISON_EVAL_TOKENS // COMPARISON_BATCH_TOKENS
        while True:
            # Historical speedrun evaluation rebuilt the validation loader for
            # every checkpoint.  Recreate the same first eval window here.
            source = tokenizing_replicated_global_batch_loader_with_state_bos_bestfit(
                tokenizer,
                COMPARISON_GLOBAL_BATCH_SIZE,
                COMPARISON_SEQUENCE_LENGTH,
                split="val",
                device="cuda",
            )
            for _ in range(eval_batches):
                tokens, labels, _state, _active = next(source)
                yield {"tokens": tokens, "labels": labels}

    if split == "train":
        sequences_per_microbatch = HISTORICAL_MICRO_BATCH_SIZE * _world_size()
        if COMPARISON_GLOBAL_BATCH_SIZE % sequences_per_microbatch:
            raise ValueError(
                "ClimbMix global batch must be divisible by micro batch size times world size"
            )
        microbatches_per_step = COMPARISON_GLOBAL_BATCH_SIZE // sequences_per_microbatch
        resume_iteration = _megatron_resume_iteration()
        source = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer,
            HISTORICAL_MICRO_BATCH_SIZE,
            COMPARISON_SEQUENCE_LENGTH,
            split="train",
            device="cuda",
            resume_state_dict=(
                {"batch_index": resume_iteration * microbatches_per_step}
                if resume_iteration
                else None
            ),
        )
        for tokens, labels, _state in source:
            yield {"tokens": tokens, "labels": labels}
    else:
        source = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer,
            HISTORICAL_MICRO_BATCH_SIZE,
            COMPARISON_SEQUENCE_LENGTH,
            split="val",
            device="cuda",
        )
        for tokens, labels in source:
            yield {"tokens": tokens, "labels": labels}


def _loss_func(labels: torch.Tensor, training: bool, output_tensor: torch.Tensor):
    global _EVAL_BYTES, _EVAL_NATS, _TOKEN_BYTES, _TRAIN_LOSS_SUM, _TRAIN_TOKENS
    losses = output_tensor.reshape(-1).float()
    labels = labels.reshape(-1)
    valid = labels >= 0
    loss_sum = (losses * valid).sum()
    token_count = valid.sum(dtype=torch.int64)
    if _TOKEN_BYTES is None:
        _TOKEN_BYTES = get_token_bytes(device=labels.device)
    elif _TOKEN_BYTES.device != labels.device:
        _TOKEN_BYTES = _TOKEN_BYTES.to(labels.device)
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
    byte_counts = torch.where(valid, _TOKEN_BYTES[safe_labels], 0)
    nats = (losses * (byte_counts > 0)).sum()
    bytes_sum = byte_counts.sum()
    if training and _TRAIN_METRICS_ENABLED:
        _TRAIN_LOSS_SUM += float(loss_sum.detach())
        _TRAIN_TOKENS += int(token_count.detach())
    elif not training:
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


def _invoke_megatron_pretrain(
    training_module,
    datasets_provider,
    model_provider,
    model_type,
) -> None:
    """Call both the legacy CLI API and the config-container API from MCore 0.18+."""
    parameters = inspect.signature(training_module.pretrain).parameters
    if "cfg_container" in parameters:
        from megatron.training.argument_utils import pretrain_cfg_container_from_args
        from megatron.training.arguments import parse_and_validate_args

        args = parse_and_validate_args(args_defaults={"tokenizer_type": "NullTokenizer"})
        config = pretrain_cfg_container_from_args(args)
        training_module.pretrain(
            config,
            datasets_provider,
            model_provider,
            model_type,
            _forward_step,
        )
        return
    training_module.pretrain(
        datasets_provider,
        model_provider,
        model_type,
        _forward_step,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )


def _run_megatron(
    variant: CampaignVariant,
    seed: int,
    tokenizer,
    profile: MegatronBackendProfile,
    recipe: OptimizationRecipe,
    *,
    dataset: str,
    data_root: Path | None,
    scale: str,
    exact_global_batch_replay: bool,
    micro_batch_size_override: int | None,
    checkpoint_dir: Path | None,
    save_interval: int | None,
    resume: bool,
    metrics_path: Path | None,
    metrics_every: int,
    throughput_protocol: ThroughputProtocol,
    initialization_hash_mode: str,
):
    global _EVAL_BYTES, _EVAL_NATS, _TOKEN_BYTES, _TRAIN_LOSS_SUM
    global _TRAIN_METRICS_ENABLED, _TRAIN_TOKENS
    _EVAL_NATS = 0.0
    _EVAL_BYTES = 0
    _TRAIN_LOSS_SUM = 0.0
    _TRAIN_TOKENS = 0
    _TRAIN_METRICS_ENABLED = metrics_path is not None
    _TOKEN_BYTES = (
        token_bytes_for_tokenizer(tokenizer, FINEWEB_VOCAB_SIZE) if _is_fineweb(dataset) else None
    )
    sys.argv = _megatron_arguments(
        variant,
        profile,
        recipe,
        dataset=dataset,
        scale=scale,
        exact_global_batch_replay=exact_global_batch_replay,
        micro_batch_size_override=micro_batch_size_override,
        checkpoint_dir=checkpoint_dir,
        save_interval=save_interval,
        resume=resume,
    ) + ["--seed", str(seed)]

    import megatron.training.training as training_module
    from megatron.core.datasets import utils as dataset_utils
    from megatron.core.enums import ModelType
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.module import MegatronModule
    from megatron.training import get_args, print_rank_0
    from megatron.training.arguments import core_transformer_config_from_args

    def skip_unused_dataset_helper_build() -> None:
        print_rank_0(f"> external {dataset} loader: skipping unused dataset-index helper build")

    dataset_utils.compile_helpers = skip_unused_dataset_helper_build

    model_kwargs = _model_config_kwargs(dataset, scale, variant)
    model_kwargs.update(recipe.model_overrides)

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
            architecture = instantiate_model(model_config)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            architecture.init_weights()
            nonfinite_parameters = [
                name
                for name, parameter in architecture.named_parameters()
                if not torch.isfinite(parameter).all()
            ]
            if nonfinite_parameters:
                raise RuntimeError(
                    "non-finite parameters after initialization: "
                    + ", ".join(nonfinite_parameters[:5])
                )
            if variant.name == "engram":
                token_map, compressed_vocab_size = build_engram_token_map(
                    tokenizer, int(model_kwargs["vocab_size"])
                )
                architecture.configure_engram_token_map(token_map, tokenizer.get_bos_token_id())
                print_rank_0(f"Engram compressed vocabulary: {compressed_vocab_size}")
            actual_parameters = architecture.num_scaling_params()["total"]
            if actual_parameters != variant.parameter_count and not recipe.model_overrides:
                raise RuntimeError(
                    f"parameter count drift for {variant.name}: "
                    f"{actual_parameters} != {variant.parameter_count}"
                )
            schedule_holder["parameter_count"] = actual_parameters
            schedule_holder["algorithmic_flops_per_token"] = float(architecture.estimate_flops())
            schedule_holder["executed_flops_per_token"] = float(
                architecture.estimate_executed_flops()
                if hasattr(architecture, "estimate_executed_flops")
                else architecture.estimate_flops()
            )
            if initialization_hash_mode != "none" and _global_rank() == 0:
                include_names = None
                if initialization_hash_mode == "shared":
                    baseline_template = (
                        get_fineweb_variant_template(scale, "baseline")
                        if _is_fineweb(dataset)
                        else get_campaign_variant(scale, "baseline")
                    )
                    baseline_kwargs = _model_config_kwargs(dataset, scale, baseline_template)
                    baseline_kwargs.update(recipe.model_overrides)
                    with torch.device("meta"):
                        baseline_model = instantiate_model(build_model_config(**baseline_kwargs))
                    include_names = {name for name, _parameter in baseline_model.named_parameters()}
                    include_names &= {name for name, _parameter in architecture.named_parameters()}
                selected_count = sum(
                    1
                    for name, _parameter in architecture.named_parameters()
                    if include_names is None or name in include_names
                )
                schedule_holder["initialization"] = {
                    "mode": initialization_hash_mode,
                    "parameter_tensors": selected_count,
                    "sha256": hash_named_tensors(
                        architecture.named_parameters(), include_names=include_names
                    ),
                }
            compile_kwargs = {"dynamic": False}
            compile_mode = profile.resolved_compile_mode(variant.name)
            if compile_mode is not None:
                compile_kwargs["mode"] = compile_mode
            self.architecture = (
                torch.compile(architecture, **compile_kwargs)
                if profile.compile_architecture
                else architecture
            )

        def set_input_tensor(self, input_tensor) -> None:
            self.input_tensor = input_tensor
            is_empty_pipeline_input = isinstance(input_tensor, list) and all(
                tensor is None for tensor in input_tensor
            )
            if input_tensor is not None and not is_empty_pipeline_input:
                raise RuntimeError("the comparison adapter requires PP=1")

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
                raise RuntimeError("labels are required during the comparison")
            args = get_args()
            if hasattr(self.architecture, "set_training_step"):
                self.architecture.set_training_step(_current_training_iteration(args))
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
            _external_batch_loader(
                tokenizer,
                "train",
                dataset=dataset,
                data_root=data_root,
                exact_global_batch_replay=exact_global_batch_replay,
                micro_batch_size_override=micro_batch_size_override,
            ),
            _external_batch_loader(
                tokenizer,
                "val",
                dataset=dataset,
                data_root=data_root,
                exact_global_batch_replay=exact_global_batch_replay,
                micro_batch_size_override=micro_batch_size_override,
            ),
            None,
        )

    datasets_provider.is_distributed = True
    schedule_holder = _install_optimizer_adapter(
        variant,
        profile,
        recipe,
        metrics_path=metrics_path,
        metrics_every=metrics_every,
        throughput_protocol=throughput_protocol,
    )
    validation_holder: dict[str, float] = {}
    original_evaluate_and_print = training_module.evaluate_and_print_results

    def record_evaluate_and_print(
        prefix,
        forward_step_func,
        data_iterator,
        model,
        iteration,
        process_non_loss_data_func,
        config,
        **kwargs,
    ):
        global _EVAL_BYTES, _EVAL_NATS
        _EVAL_NATS = 0.0
        _EVAL_BYTES = 0
        result = original_evaluate_and_print(
            prefix,
            forward_step_func,
            data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            **kwargs,
        )
        validation_nats, validation_bytes = _reduce_validation_totals()
        if validation_bytes <= 0:
            raise RuntimeError("Megatron validation produced no represented bytes")
        bpb = validation_nats / (math.log(2.0) * validation_bytes)
        validation_holder["final_bpb"] = bpb
        _append_jsonl(
            metrics_path,
            {
                "kind": "validation",
                "step": int(iteration),
                "tokens": int(iteration) * COMPARISON_BATCH_TOKENS,
                "bpb": bpb,
                "eval_tokens": COMPARISON_EVAL_TOKENS,
            },
        )
        return result

    training_module.evaluate_and_print_results = record_evaluate_and_print
    try:
        _invoke_megatron_pretrain(
            training_module,
            datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
        )
    finally:
        training_module.evaluate_and_print_results = original_evaluate_and_print
        _TRAIN_METRICS_ENABLED = False
    if "final_bpb" not in validation_holder:
        raise RuntimeError("Megatron completed without a recorded validation point")
    return (
        schedule_holder["schedule"],
        int(schedule_holder["parameter_count"]),
        validation_holder["final_bpb"],
        {
            "initialization": schedule_holder.get(
                "initialization", {"mode": initialization_hash_mode, "sha256": None}
            ),
            "algorithmic_flops_per_token": schedule_holder["algorithmic_flops_per_token"],
            "executed_flops_per_token": schedule_holder["executed_flops_per_token"],
        },
    )


def _environment(
    variant: CampaignVariant,
    seed: int,
    mode: str,
    profile: MegatronBackendProfile,
    recipe: OptimizationRecipe,
    *,
    dataset: str,
    data_root: Path | None,
    scale: str,
    exact_global_batch_replay: bool,
    micro_batch_size_override: int | None,
    checkpoint_dir: Path | None,
    save_interval: int | None,
    resume: bool,
    throughput_protocol: ThroughputProtocol,
    budget: BudgetResolution,
    artifact_policy: str,
    data_identity: dict[str, Any] | None,
    algorithmic_flops_per_token: float,
    executed_flops_per_token: float,
    initialization_hash_mode: str,
) -> dict[str, Any]:
    repository = _repository_root()
    profile_payload = asdict(profile)
    profile_payload["resolved_compile_mode"] = profile.resolved_compile_mode(variant.name)
    if micro_batch_size_override is not None:
        micro_batch_size = micro_batch_size_override
    elif _is_fineweb(dataset):
        micro_batch_size = COMPARISON_GLOBAL_BATCH_SIZE // _world_size()
    elif exact_global_batch_replay:
        micro_batch_size = math.ceil(COMPARISON_GLOBAL_BATCH_SIZE / _world_size())
    else:
        micro_batch_size = HISTORICAL_MICRO_BATCH_SIZE
    scheduled_global_batch_size = (
        math.ceil(COMPARISON_GLOBAL_BATCH_SIZE / _world_size()) * _world_size()
        if exact_global_batch_replay
        else COMPARISON_GLOBAL_BATCH_SIZE
    )
    return {
        "backend": "megatron",
        "support_tier": "mcore_training_wrapper",
        "variant": asdict(variant),
        "scale": scale,
        "seed": seed,
        "mode": mode,
        "dataset": dataset,
        "comparison_regime": budget.regime.value,
        "budget": asdict(budget),
        "artifact_policy": artifact_policy,
        "data_identity": data_identity,
        "dataset_manifest_sha256": (
            data_identity["manifest_identity_sha256"] if data_identity is not None else None
        ),
        "data_order_id": stable_json_sha256(
            {
                "dataset_manifest_sha256": (
                    data_identity["manifest_identity_sha256"]
                    if data_identity is not None
                    else "unverified"
                ),
                "seed": seed,
                "loader": (
                    "fineweb-distributed-microbatch-cursor-v1"
                    if _is_fineweb(dataset)
                    else "historical-climbmix-loader"
                ),
            }
        ),
        "initialization_hash_mode": initialization_hash_mode,
        "algorithmic_flops_per_token": algorithmic_flops_per_token,
        "executed_flops_per_token": executed_flops_per_token,
        "exact_global_batch_replay": exact_global_batch_replay,
        "effective_global_batch_sequences": COMPARISON_GLOBAL_BATCH_SIZE,
        "micro_batch_sequences": micro_batch_size,
        "gradient_accumulation_microbatches": (
            scheduled_global_batch_size // (micro_batch_size * _world_size())
        ),
        "scheduled_global_batch_sequences": scheduled_global_batch_size,
        "world_size": _world_size(),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "save_interval": save_interval,
        "resume": resume,
        "throughput_protocol": throughput_protocol.to_dict(),
        "dataloader_resume_contract": (
            "fineweb-distributed-microbatch-cursor-v1"
            if _is_fineweb(dataset)
            else "climbmix-exact-microbatch-cursor-v2"
        ),
        "validation_window_contract": (
            "fixed-first-window-v1" if _is_fineweb(dataset) else "historical-climbmix-window"
        ),
        "backend_profile": profile_payload,
        "optimization_recipe": asdict(recipe),
        "optimizer_contract_sha256": stable_json_sha256(asdict(recipe)),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        **_source_provenance(repository),
        "data_root": (
            str(data_root) if data_root is not None else os.environ.get("NANOCHAT_BASE_DIR")
        ),
        "triton_ptxas_path": os.environ.get("TRITON_PTXAS_PATH"),
        "semantic_equivalence": (
            f"same architecture, {dataset} data order, tokenizer, seed, batch budget, and "
            "mixed matrix/Adam schedule; the named optimization recipe is the only experimental "
            "delta; compilation changes execution only; native-master "
            "profiles preserve the speedrun mixed-storage policy without detached FP32 "
            "replicas; MCore owns initialization, DDP accumulation, finite checks, and "
            "evaluation scheduling; bitwise equivalence is not claimed"
        ),
    }


def main() -> None:
    global _ATTEMPT_ID, _RUN_ID

    parser = argparse.ArgumentParser(description=__doc__)
    scales = tuple(dict.fromkeys((*CAMPAIGN_VARIANTS, *FINEWEB_CAMPAIGN_VARIANTS)))
    parser.add_argument("--dataset", choices=DATASETS, default="climbmix")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--scale", choices=scales, default="10m")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=COMPARISON_SEEDS)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument(
        "--comparison-regime",
        choices=tuple(item.value for item in ComparisonRegime),
        help="controlled=fixed tokens, fixed_compute=fixed model FLOPs, scaling=fixed tokens/parameter",
    )
    parser.add_argument("--target-train-tokens", type=int)
    parser.add_argument("--target-model-flops", type=float)
    parser.add_argument("--tokens-per-parameter", type=float)
    parser.add_argument(
        "--artifact-policy",
        choices=("metrics_only", "research"),
        default="metrics_only",
        help="research requires content provenance, initialization hash, and final checkpoint",
    )
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument(
        "--data-verification",
        choices=("metadata", "full"),
        default="metadata",
    )
    parser.add_argument(
        "--initialization-hash",
        choices=("none", "shared", "full"),
        default="none",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        help="FineWeb rank-local micro batch; Megatron accumulates to the fixed global batch",
    )
    parser.add_argument(
        "--exact-global-batch-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="preserve the 192-sequence canonical stream over non-divisible DP worlds",
    )
    parser.add_argument("--metrics-every", type=int, default=10)
    parser.add_argument(
        "--throughput-warmup-steps",
        type=int,
        default=10,
        help="optimizer-step intervals excluded from steady-state throughput",
    )
    parser.add_argument(
        "--throughput-measurement-steps",
        type=int,
        default=0,
        help="steady-state intervals to measure (0 uses every post-warmup interval)",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="load model/optimizer state and derive the FineWeb cursor from checkpoint iteration",
    )
    parser.add_argument(
        "--backend-profile",
        choices=tuple(MEGATRON_BACKEND_PROFILES),
        default="legacy",
        help="explicit Megatron wrapper optimization profile",
    )
    parser.add_argument(
        "--optimization-recipe",
        choices=tuple(OPTIMIZATION_RECIPES),
        default="baseline",
        help="isolated speedrun/Marin optimization recipe",
    )
    args = parser.parse_args()
    if args.probe_steps < 0:
        parser.error("--probe-steps must be non-negative")
    if args.metrics_every < 1:
        parser.error("--metrics-every must be positive")
    if args.throughput_warmup_steps < 0 or args.throughput_measurement_steps < 0:
        parser.error("throughput step counts must be non-negative")
    if args.micro_batch_size is not None and args.micro_batch_size < 1:
        parser.error("--micro-batch-size must be positive")
    if args.checkpoint_dir is not None and (args.save_interval is None or args.save_interval < 1):
        parser.error("--checkpoint-dir requires a positive --save-interval")
    if args.checkpoint_dir is None and args.save_interval is not None:
        parser.error("--save-interval requires --checkpoint-dir")
    if args.resume and args.checkpoint_dir is None:
        parser.error("--resume requires --checkpoint-dir")
    if args.artifact_policy == "research":
        if args.checkpoint_dir is None:
            parser.error("research artifact policy requires --checkpoint-dir")
        if args.data_manifest is None:
            parser.error("research artifact policy requires --data-manifest")
        if args.initialization_hash == "none":
            parser.error("research artifact policy requires --initialization-hash=shared or full")
    try:
        contract_variant, budget, probed_algorithmic_flops, probed_executed_flops = (
            resolve_variant_contract(
                args.dataset,
                args.scale,
                args.variant,
                regime=args.comparison_regime,
                target_train_tokens=args.target_train_tokens,
                target_model_flops=args.target_model_flops,
                tokens_per_parameter=args.tokens_per_parameter,
            )
        )
    except (ContractError, KeyError, ValueError) as error:
        parser.error(str(error))
    if _is_fineweb(args.dataset):
        if args.scale not in FINEWEB_CAMPAIGN_VARIANTS:
            parser.error(f"--scale={args.scale} is not available for FineWeb")
        if args.data_root is None:
            parser.error("--data-root is required for FineWeb")
        if args.micro_batch_size is not None and COMPARISON_GLOBAL_BATCH_SIZE % (
            args.micro_batch_size * _world_size()
        ):
            parser.error(
                "FineWeb global batch must be divisible by micro batch size times world size"
            )
        data_root = args.data_root.expanduser().resolve()
        data_summary = inspect_fineweb_dataset(
            data_root,
            expected_train_shards=FINEWEB_EXPECTED_TRAIN_SHARDS[args.dataset],
            required_train_tokens=contract_variant.training_tokens + 1,
        )
    else:
        if args.scale not in CAMPAIGN_VARIANTS:
            parser.error(f"--scale={args.scale} is not available for ClimbMix")
        if args.data_root is not None:
            parser.error("--data-root is only valid with a FineWeb dataset")
        data_root = None
        data_summary = None
    climbmix_root = os.environ.get("NANOCHAT_BASE_DIR")
    dataset_root_for_manifest = (
        data_root
        if data_root is not None
        else (resolve_climbmix_data_dir(climbmix_root) if climbmix_root else None)
    )
    data_identity = None
    if args.data_manifest is not None:
        if dataset_root_for_manifest is None:
            parser.error("a dataset root is required to verify --data-manifest")
        data_identity = verify_dataset_manifest(
            dataset_root_for_manifest,
            args.data_manifest,
            mode=args.data_verification,
        )
    variant = (
        replace(contract_variant, steps=args.probe_steps) if args.probe_steps else contract_variant
    )
    mode = "probe" if args.probe_steps else "full"
    profile = get_megatron_backend_profile(args.backend_profile)
    recipe = get_optimization_recipe(args.optimization_recipe)
    throughput_protocol = ThroughputProtocol(
        warmup_steps=args.throughput_warmup_steps,
        measurement_steps=args.throughput_measurement_steps,
    )
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_dir = (
        args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir is not None else None
    )
    primary = _global_rank() == 0
    attempt_id = str(uuid.uuid4())
    _ATTEMPT_ID = attempt_id
    marker = (
        _claim_run_directory(run_dir, variant, args.seed, attempt_id=attempt_id)
        if primary
        else run_dir / "RUNNING.json"
    )
    metrics_path = run_dir / "metrics.jsonl"
    try:
        if primary:
            try:
                metrics_path.touch(exist_ok=False)
            except FileExistsError as error:
                raise RuntimeError(
                    f"refusing to overwrite existing metrics file: {metrics_path}"
                ) from error
        runtime = validate_runtime(require_pretrain=False)
        environment = _environment(
            variant,
            args.seed,
            mode,
            profile,
            recipe,
            dataset=args.dataset,
            data_root=data_root,
            scale=args.scale,
            exact_global_batch_replay=args.exact_global_batch_replay,
            micro_batch_size_override=args.micro_batch_size,
            checkpoint_dir=checkpoint_dir,
            save_interval=args.save_interval,
            resume=args.resume,
            throughput_protocol=throughput_protocol,
            budget=budget,
            artifact_policy=args.artifact_policy,
            data_identity=data_identity,
            algorithmic_flops_per_token=probed_algorithmic_flops,
            executed_flops_per_token=probed_executed_flops,
            initialization_hash_mode=args.initialization_hash,
        )
        environment["runtime"] = runtime
        if data_summary is not None:
            environment["data"] = data_summary
        tokenizer = (
            get_pretrained_tokenizer("gpt2") if _is_fineweb(args.dataset) else get_tokenizer()
        )
        padded_vocab_size = int(
            _model_config_kwargs(args.dataset, args.scale, contract_variant)["vocab_size"]
        )
        environment["tokenizer_sha256"] = hash_tokenizer_vocabulary(
            tokenizer, padded_vocab_size=padded_vocab_size
        )
        environment["metrics_path"] = str(metrics_path)
        environment["attempt_id"] = attempt_id
        identity_payload = {
            key: environment[key]
            for key in (
                "backend",
                "variant",
                "scale",
                "seed",
                "dataset",
                "comparison_regime",
                "budget",
                "dataset_manifest_sha256",
                "data_order_id",
                "tokenizer_sha256",
                "optimizer_contract_sha256",
                "source_commit",
                "source_worktree_sha256",
                "world_size",
                "effective_global_batch_sequences",
            )
        }
        _RUN_ID = stable_json_sha256(identity_payload)
        environment["run_id"] = _RUN_ID
        environment["run_identity"] = identity_payload
        if args.artifact_policy == "research" and environment["source_dirty"]:
            raise RuntimeError("research runs require a clean source worktree")
        if primary:
            _write_json(run_dir / "resolved_run.json", environment)
        started = time.perf_counter()
        schedule, parameter_count, final_bpb, model_audit = _run_megatron(
            variant,
            args.seed,
            tokenizer,
            profile,
            recipe,
            dataset=args.dataset,
            data_root=data_root,
            scale=args.scale,
            exact_global_batch_replay=args.exact_global_batch_replay,
            micro_batch_size_override=args.micro_batch_size,
            checkpoint_dir=checkpoint_dir,
            save_interval=args.save_interval,
            resume=args.resume,
            metrics_path=metrics_path,
            metrics_every=args.metrics_every,
            throughput_protocol=throughput_protocol,
            initialization_hash_mode=args.initialization_hash,
        )
        wall_seconds = time.perf_counter() - started
        throughput = schedule.throughput_summary()
        if not math.isfinite(final_bpb):
            raise RuntimeError("validation produced a non-finite final BPB")
        if not math.isclose(
            model_audit["algorithmic_flops_per_token"],
            probed_algorithmic_flops,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError("algorithmic FLOP accounting drifted after model construction")
        checkpoint_artifact = None
        if args.artifact_policy == "research":
            assert checkpoint_dir is not None
            checkpoint_artifact = _validate_checkpoint_artifact(checkpoint_dir, variant.steps)
        result = {
            **environment,
            "status": "complete",
            "parameter_count": parameter_count,
            "initialization": model_audit["initialization"],
            "shared_initialization_sha256": model_audit["initialization"].get("sha256"),
            "training_steps": variant.steps,
            "training_tokens": variant.training_tokens,
            "global_batch_tokens": COMPARISON_BATCH_TOKENS,
            "sequence_length": COMPARISON_SEQUENCE_LENGTH,
            "contract_training_steps": contract_variant.steps,
            "contract_training_tokens": contract_variant.training_tokens,
            "validation_tokens": COMPARISON_EVAL_TOKENS,
            "final_bpb": final_bpb,
            "wall_seconds": wall_seconds,
            "metrics_sha256": sha256_file(metrics_path),
            "checkpoint_artifact": checkpoint_artifact,
            "wandb_run_url": os.environ.get("WANDB_RUN_URL"),
            **throughput,
            "completed_at_unix": time.time(),
        }
        if primary:
            _write_json(run_dir / "result.json", result)
            marker.replace(run_dir / "COMPLETE.json")
    except BaseException as error:
        if primary:
            failure = classify_failure(error)
            _write_json(
                run_dir / "FAILED.json",
                {
                    "status": "failed",
                    "run_id": _RUN_ID or None,
                    "attempt_id": attempt_id,
                    "failure": failure.to_dict(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "failed_at_unix": time.time(),
                },
            )
            marker.unlink(missing_ok=True)
        raise
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
