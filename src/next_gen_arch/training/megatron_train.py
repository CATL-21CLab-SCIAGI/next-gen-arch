"""Megatron pretrain-loop adapter for controlled small-scale architecture runs.

This adapter deliberately keeps the architecture math and historical mixed
Muon/Adam policy in this package while delegating distributed initialization,
DDP gradient accumulation, pipeline scheduling, finite checks, and reporting to
the pinned Megatron-LM submodule. It is a comparison adapter, not a claim that
every mechanism has a tensor-parallel-native MCore layer implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
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
from next_gen_arch.training.optimization_recipes import (
    OPTIMIZATION_RECIPES,
    OptimizationRecipe,
    get_optimization_recipe,
)
from next_gen_arch.training.tokenizer import get_token_bytes, get_tokenizer

_EVAL_NATS = 0.0
_EVAL_BYTES = 0
_TOKEN_BYTES: torch.Tensor | None = None


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


def _current_training_iteration(args) -> int:
    """Read Megatron's live loop counter, falling back to its resume counter."""
    if hasattr(args, "curr_iteration"):
        return int(args.curr_iteration)
    return int(args.iteration)


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={root}", *args],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def _source_provenance(repository: Path) -> dict[str, Any]:
    status = _git_output(repository, "status", "--porcelain=v1", "--untracked-files=all")
    diff = subprocess.run(
        ["git", "-c", f"safe.directory={repository}", "diff", "--binary", "HEAD"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    untracked_output = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    untracked = sorted(Path(os.fsdecode(path)) for path in untracked_output.split(b"\0") if path)
    untracked_digest = hashlib.sha256()
    worktree_digest = hashlib.sha256(b"tracked-diff\0" + diff)
    for relative in untracked:
        path = repository / relative
        if path.is_symlink():
            content = b"symlink\0" + os.fsencode(os.readlink(path))
        elif path.is_file():
            content = b"file\0" + path.read_bytes()
        else:
            content = b"other\0"
        framed = (
            len(os.fsencode(relative)).to_bytes(8, "big")
            + os.fsencode(relative)
            + len(content).to_bytes(8, "big")
            + content
        )
        untracked_digest.update(framed)
        worktree_digest.update(framed)
    return {
        "source_commit": _git_output(repository, "rev-parse", "HEAD"),
        "source_dirty": bool(status),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_untracked_files": [str(path) for path in untracked],
        "source_untracked_sha256": untracked_digest.hexdigest(),
        "source_worktree_sha256": worktree_digest.hexdigest(),
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _megatron_arguments(
    variant: TenMVariant,
    profile: MegatronBackendProfile,
    recipe: OptimizationRecipe,
) -> list[str]:
    eval_iters = TEN_M_EVAL_TOKENS // TEN_M_BATCH_TOKENS
    arguments = [
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
        str(recipe.gradient_clip),
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
    return arguments


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


def _install_optimizer_adapter(
    variant: TenMVariant,
    profile: MegatronBackendProfile,
    recipe: OptimizationRecipe,
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
            raise RuntimeError("the 10M adapter requires one non-pipelined architecture model")
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


def _run_megatron(
    variant: TenMVariant,
    seed: int,
    tokenizer,
    profile: MegatronBackendProfile,
    recipe: OptimizationRecipe,
):
    sys.path.insert(0, str(MEGATRON_ROOT))
    sys.argv = _megatron_arguments(variant, profile, recipe) + ["--seed", str(seed)]

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
                token_map, compressed_vocab_size = build_engram_token_map(tokenizer, 32_768)
                architecture.configure_engram_token_map(token_map, tokenizer.get_bos_token_id())
                print_rank_0(f"Engram compressed vocabulary: {compressed_vocab_size}")
            actual_parameters = architecture.num_scaling_params()["total"]
            if actual_parameters != variant.parameter_count and not recipe.model_overrides:
                raise RuntimeError(
                    f"parameter count drift for {variant.name}: "
                    f"{actual_parameters} != {variant.parameter_count}"
                )
            schedule_holder["parameter_count"] = actual_parameters
            compile_kwargs = {"dynamic": False}
            if profile.compile_mode is not None:
                compile_kwargs["mode"] = profile.compile_mode
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
            _external_batch_loader(tokenizer, "train"),
            _external_batch_loader(tokenizer, "val"),
            None,
        )

    datasets_provider.is_distributed = True
    schedule_holder = _install_optimizer_adapter(variant, profile, recipe)
    pretrain(
        datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        _forward_step,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    return schedule_holder["schedule"], int(schedule_holder["parameter_count"])


def _environment(
    variant: TenMVariant,
    seed: int,
    mode: str,
    profile: MegatronBackendProfile,
    recipe: OptimizationRecipe,
) -> dict[str, Any]:
    repository = _repository_root()
    return {
        "backend": "megatron",
        "support_tier": "mcore_training_wrapper",
        "variant": asdict(variant),
        "seed": seed,
        "mode": mode,
        "backend_profile": asdict(profile),
        "optimization_recipe": asdict(recipe),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        **_source_provenance(repository),
        "megatron_commit": MEGATRON_COMMIT,
        "data_root": os.environ.get("NANOCHAT_BASE_DIR"),
        "triton_ptxas_path": os.environ.get("TRITON_PTXAS_PATH"),
        "semantic_equivalence": (
            "same architecture, ClimbMix packing, tokenizer, seed, batch budget, and "
            "mixed matrix/Adam schedule; the named optimization recipe is the only experimental "
            "delta; compilation changes execution only; native-master "
            "profiles preserve the speedrun mixed-storage policy without detached FP32 "
            "replicas; MCore owns initialization, DDP accumulation, finite checks, and "
            "evaluation scheduling; bitwise equivalence is not claimed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=TEN_M_SEEDS)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--probe-steps", type=int, default=0)
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
    contract_variant = get_ten_m_variant(args.variant)
    variant = (
        replace(contract_variant, steps=args.probe_steps) if args.probe_steps else contract_variant
    )
    mode = "probe" if args.probe_steps else "full"
    profile = get_megatron_backend_profile(args.backend_profile)
    recipe = get_optimization_recipe(args.optimization_recipe)
    run_dir = args.run_dir.expanduser().resolve()
    primary = _global_rank() == 0
    marker = (
        _claim_run_directory(run_dir, variant, args.seed) if primary else run_dir / "RUNNING.json"
    )
    try:
        submodule = validate_submodule()
        environment = _environment(variant, args.seed, mode, profile, recipe)
        environment["submodule"] = submodule
        if primary:
            _write_json(run_dir / "resolved_run.json", environment)
        tokenizer = get_tokenizer()
        started = time.perf_counter()
        schedule, parameter_count = _run_megatron(variant, args.seed, tokenizer, profile, recipe)
        wall_seconds = time.perf_counter() - started
        measured_seconds, tokens_per_second = schedule.measured_throughput()
        validation_nats, validation_bytes = _reduce_validation_totals()
        if validation_bytes <= 0:
            raise RuntimeError("Megatron completed without a validation BPB denominator")
        final_bpb = validation_nats / (math.log(2.0) * validation_bytes)
        if not math.isfinite(final_bpb):
            raise RuntimeError("validation produced a non-finite final BPB")
        result = {
            **environment,
            "status": "complete",
            "parameter_count": parameter_count,
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
        if primary:
            _write_json(run_dir / "result.json", result)
            marker.replace(run_dir / "COMPLETE.json")
    except BaseException as error:
        if primary:
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
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
