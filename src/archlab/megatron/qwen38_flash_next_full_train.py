"""Megatron-native trainer for the full Qwen3.8-Flash-Next text variant."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

from archlab.architectures.qwen38_flash_next_full import (
    SOURCE_CONFIG_SHA256,
    SOURCE_MODEL,
    SOURCE_REVISION,
    TOKENIZER_SHA256,
    Qwen38FlashNextFullConfig,
    parameter_count_contract,
)
from archlab.artifacts import atomic_write_json as _atomic_json
from archlab.artifacts import sha256_file as _sha256
from archlab.megatron.backend import validate_runtime
from archlab.megatron.checkpoint_staging import (  # noqa: F401 - historical private import compatibility
    _completed_checkpoint_iteration,
    _execute_checkpoint_request_by_local_rank,
    _install_bounded_torch_dist_staging,
)
from archlab.megatron.indexed_data import (  # noqa: F401 - historical private import compatibility
    data_prefixes as _data_prefixes,
)
from archlab.megatron.indexed_data import validated_data_prefixes as _validated_data_prefixes
from archlab.megatron.lifecycle import current_iteration as _current_iteration
from archlab.megatron.lifecycle import invoke_pretrain
from archlab.megatron.losses import (
    masked_token_loss as _loss_func,  # noqa: F401 - historical private import compatibility
)
from archlab.megatron.losses import native_token_forward_step as _forward_step
from archlab.megatron.qwen38_flash_next_checks import (
    _effective_probe_gradient,
    _probe_parameter_counts,
    _probe_ple_replica_equality,
    _probe_restored_replica_equality,
)
from archlab.megatron.qwen38_flash_next_config import (  # noqa: F401 - historical private import compatibility
    ATTENTION_GROUPING,
    BILLION_DEPTH48_NO_MTP_MODEL_VARIANT,
    CHECKPOINT_INTERVAL_STEPS,
    CHECKPOINT_WRITER_THREADS,
    DISTRIBUTED_TIMEOUT_MINUTES,
    EFFECTIVE_TOKENS,
    FULL_MODEL_VARIANT,
    LOSS_NORMALIZATION,
    NATIVE_MUON_FP32_MATMUL_PRECISION,
    QUARTER_DEPTH48_NO_MTP_MODEL_VARIANT,
    TOKENS_PER_STEP,
    TRAIN_STEPS,
    WIDTH320_E32_MODEL_VARIANT,
    _megatron_argv,
    _native_muon_contract,
    _parser,
)
from archlab.megatron.qwen38_flash_next_model import (  # noqa: F401 - historical private import compatibility
    _assert_dp_only_groups,
    _bind_native_moe_layer_number,
    _build_model_classes,
    _resolve_qwen_layer_number,
    _tag_native_optimizer_fallbacks,
    build_model,
)
from archlab.megatron.simplicial_production import (
    attention_variant_contract,
    validate_attention_resume,
)
from archlab.megatron.token_batches import DPRankTokenBatches, partition_prefixes_for_dp_rank


def shifted_mtp_targets(labels: torch.Tensor, depths: int = 3) -> tuple[torch.Tensor, ...]:
    """CPU-testable statement of the three native-MTP target shifts."""
    targets = []
    for depth in range(1, depths + 1):
        target = torch.full_like(labels, -1)
        if depth < labels.size(1):
            target[:, :-depth] = labels[:, depth:]
        targets.append(target)
    return tuple(targets)


def mtp_weighted_mean(losses: torch.Tensor, scaling: float = 0.1) -> torch.Tensor:
    if losses.size(0) != 3:
        raise ValueError("the supported MTP objective has exactly three depths")
    return scaling * losses.mean(dim=0)


def _assert_pipeline_data_rank_layout(
    *,
    global_rank: int,
    data_parallel_rank: int,
    data_parallel_world_size: int,
    pipeline_global_ranks: tuple[int, ...],
    pipeline_world_size: int = 4,
) -> None:
    """Validate PP sample ownership without a collective inside the pipeline schedule."""
    if data_parallel_world_size < 1:
        raise RuntimeError("data-parallel world size must be positive")
    if (
        pipeline_world_size < 1
        or len(pipeline_global_ranks) != pipeline_world_size
        or global_rank not in pipeline_global_ranks
    ):
        raise RuntimeError("pipeline group does not match the configured layer layout")
    projected_data_ranks = {rank % data_parallel_world_size for rank in pipeline_global_ranks}
    if projected_data_ranks != {data_parallel_rank}:
        raise RuntimeError("pipeline stages do not share one deterministic data rank")


def _write_contract(args, config) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    tokenizer_hash = _sha256(args.tokenizer / "tokenizer.json")
    config_hash = _sha256(args.tokenizer / "config.json")
    if tokenizer_hash != TOKENIZER_SHA256 or config_hash != SOURCE_CONFIG_SHA256:
        raise RuntimeError("pinned Qwen source/tokenizer hash drift")
    runtime = validate_runtime(require_pretrain=False)
    attention_variant = attention_variant_contract(args)
    if args.model_variant == FULL_MODEL_VARIANT:
        model_name = "Qwen3.8-Flash-Next dense-attention owner-sharded-PLE variant"
        variant_differences = [
            "dense global attention at 2K instead of QSA",
            "three MTP depths sharing one physical layer",
            "Megatron-native Muon instead of private Canzona",
            "GPU-owner-sharded PLE instead of unpublished host prefetch",
        ]
    elif args.model_variant == QUARTER_DEPTH48_NO_MTP_MODEL_VARIANT:
        model_name = "Qwen3.8-Flash-Next quarter-shape depth-48 no-MTP variant"
        variant_differences = [
            "divisible width, head, expert, GR, and PLE shapes quartered",
            "all 48 backbone layers retained with an even PP4 split",
            "MTP module and auxiliary objective disabled",
            "dense global attention at 2K instead of QSA",
            "quarter-shape router stability coefficients use auxiliary 0.01 and z-loss 0.001",
            "Megatron-native Muon instead of private Canzona",
            "GPU-owner-sharded PLE instead of unpublished host prefetch",
        ]
    elif args.model_variant == WIDTH320_E32_MODEL_VARIANT:
        model_name = "Qwen3.8-Flash-Next width-320 E32 source-ratio no-MTP variant"
        variant_differences = [
            "48 layers; width 320; 32 routed experts, top-10 plus shared, width 80",
            "four residual streams, rank 40; 16 PLE hash heads; 512H table base",
            "sigmoid attention output gate and per-head QK RMSNorm restored",
            "source zero-centered GR/PLE/QK RMSNorm; direct-gamma GDN output norm",
            "native TE Q/gate/K/V are separate Muon matrices (no ungated QKV splitter)",
            "DP-only; all experts and all PLE partitions local on every GPU",
            "MTP disabled; dense global attention at 2K instead of QSA; text only",
            "source router auxiliary coefficient 0.001 and no z-loss",
        ]
    else:
        model_name = "Qwen3.8-Flash-Next 1B depth-48 no-MTP variant"
        variant_differences = [
            "approximately 1B total parameters, including every expert and PLE table",
            "all 48 backbone layers retained; width 384, 64 experts, expert width 112",
            "PLE base vocabulary 1M per hash head; embedding width 384",
            (
                "DP-only: complete experts and PLE tables replicated on every GPU"
                if args.parallelism == "dp-only"
                else "PP1 with node-local EP8 and four expert-data-parallel replicas"
            ),
            "MTP module and auxiliary objective disabled",
            "dense global attention at 2K instead of QSA",
            "router auxiliary coefficient 0.01 and z-loss coefficient 0.001",
            "Megatron-native Muon instead of private Canzona",
            "GPU-owner-sharded PLE instead of unpublished host prefetch",
        ]
    payload = {
        "model": model_name,
        "source": {
            "model": SOURCE_MODEL,
            "revision": SOURCE_REVISION,
            "config_sha256": SOURCE_CONFIG_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "weights_loaded": False,
            "scope": "from-scratch text-only pretraining",
        },
        "model_config": config.to_dict(),
        "parameter_count": parameter_count_contract(config),
        "variant_differences": variant_differences,
        "parallelism": {
            "tensor": 1,
            "pipeline": len(config.pipeline_layers),
            "expert": 1 if args.parallelism == "dp-only" else 8,
            "expert_tensor": 1,
            "context": 1,
            "mode": args.parallelism,
        },
        "execution": {
            "cuda_device_max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
            "overlap_grad_reduce": True,
            "overlap_param_gather": args.parallelism != "dp-only",
            "moe_permute_fusion": args.fused_moe,
            "moe_router_fusion": args.fused_moe,
            "cross_entropy_loss_fusion": args.fused_cross_entropy,
            "cross_entropy_fusion_impl": "native",
            "load_dir": str(args.load_dir) if args.load_dir else None,
        },
        "pipeline_layers": list(config.pipeline_layers),
        "optimizer": {
            **_native_muon_contract(),
            **(
                {"qkv_split": "separate native TE Q/gate/K/V parameters; each a Muon matrix"}
                if config.attention_output_gate
                else {}
            ),
            "peak_lr": args.learning_rate,
            "minimum_lr": args.minimum_learning_rate,
            "warmup_fraction": args.warmup_fraction,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.clip_grad,
            "fc1_layout": "distinct native TE gate/up parameters",
            "ple_tables": "Adam, zero weight decay",
        },
        "checkpointing": {
            "format": "Megatron torch_dist",
            "serialization": "container-owned distributed checkpointing",
            "host_staging": "one local GPU rank per node at a time",
            "files_per_rank": CHECKPOINT_WRITER_THREADS,
            "reason": "bound owner-sharded PLE optimizer staging to host memory",
        },
        "training": {
            "loss_normalization": LOSS_NORMALIZATION,
            "attention_grouping": ATTENTION_GROUPING,
            "seed": args.seed,
            "sequence_length": config.sequence_len,
            "micro_batch_sequences": args.micro_batch_size,
            "global_batch_sequences": args.global_batch_size,
            "tokens_per_step": args.global_batch_size * config.sequence_len,
            "train_steps": args.probe_steps or TRAIN_STEPS,
            "target_tokens": args.target_train_tokens,
            "effective_tokens": (
                args.probe_steps * args.global_batch_size * config.sequence_len
                if args.probe_steps
                else EFFECTIVE_TOKENS
            ),
            "checkpoint_interval_steps": (
                args.probe_save_interval or args.probe_steps
                if args.probe_steps
                else CHECKPOINT_INTERVAL_STEPS
            ),
        },
        "data": {
            "root": str(args.data_root),
            "manifest_sha256": _sha256(args.data_root / "DATA_READY.json"),
            "sharding": "data-parallel rank; identical across TP/PP/EP/CP ranks",
        },
        "precision": "BF16 model/compute; FP32 optimizer and Muon orthogonalization",
        "source_commit": os.environ.get("NGA_EXPECTED_COMMIT"),
        "launch_recipe": {
            "path": os.environ.get("NGA_LAUNCH_RECIPE"),
            "sha256": os.environ.get("NGA_LAUNCH_RECIPE_SHA256"),
        },
        "implementation_sha256": {
            "architecture": _sha256(Path(inspect.getfile(Qwen38FlashNextFullConfig)).resolve()),
            "trainer": _sha256(Path(__file__).resolve()),
        },
        "runtime": runtime,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "created_at_unix": time.time(),
    }
    source_root = Path(__file__).parents[1]
    for relative in (
        "artifacts.py", "megatron/indexed_data.py", "megatron/token_batches.py",
        "megatron/lifecycle.py", "megatron/losses.py", "megatron/checkpoint_staging.py",
        "megatron/qwen38_flash_next_config.py", "megatron/qwen38_flash_next_model.py",
        "megatron/qwen38_flash_next_checks.py",
    ):
        payload["implementation_sha256"][relative] = _sha256(source_root / relative)
    if attention_variant is not None:
        import importlib.metadata

        payload["attention_variant"] = attention_variant
        payload["model"] += " / " + attention_variant["name"]
        payload["parameter_count"]["simplicial_extra"] = attention_variant["extra_parameters"]
        payload["parameter_count"]["total"] += attention_variant["extra_parameters"]
        payload["variant_differences"].append(
            "six alternate full-attention cores replaced with 16x128 gated simplicial attention"
        )
        source_root = Path(__file__).parents[1]
        for relative in ("megatron/simplicial_production.py", "megatron/simplicial_attention.py",
                         "architectures/simplicial_attention.py", "architectures/simplicial_kernels.py"):
            payload["implementation_sha256"][relative] = _sha256(source_root / relative)
        payload["runtime"]["simplicial_packages"] = {
            name: importlib.metadata.version(name) for name in ("fla-core", "triton", "emerging-optimizers")
        }
    contract = args.run_dir / "RUN_CONTRACT.json"
    if contract.exists() and not args.resume:
        raise RuntimeError("run directory already contains a contract")
    resume_contract = args.load_dir.parent / "RUN_CONTRACT.json" if args.load_dir else contract
    if args.load_dir and not resume_contract.is_file():
        raise RuntimeError("explicit checkpoint source lacks a RUN_CONTRACT.json")
    for previous_contract in dict.fromkeys((contract, resume_contract)):
        if not previous_contract.exists():
            continue
        previous = json.loads(previous_contract.read_text())
        validate_attention_resume(previous, payload)
        if previous.get("training", {}).get("loss_normalization") != LOSS_NORMALIZATION:
            raise RuntimeError("loss normalization changed; use a fresh run directory and weights")
        if previous.get("model_config") != payload["model_config"]:
            raise RuntimeError("model geometry changed; use a fresh run directory and weights")
        if previous.get("training", {}).get("attention_grouping") != ATTENTION_GROUPING:
            raise RuntimeError("native attention grouping changed; use a fresh run directory")
    if not contract.exists():
        _atomic_json(contract, payload)
    _atomic_json(args.run_dir / "contracts" / f"attempt-{time.time_ns()}.json", payload)
    _atomic_json(args.run_dir / "LATEST_CONTRACT.json", payload)


def _run(args: argparse.Namespace) -> None:
    if args.model_variant == FULL_MODEL_VARIANT:
        config = Qwen38FlashNextFullConfig(sequence_len=args.sequence_length)
    elif args.model_variant == BILLION_DEPTH48_NO_MTP_MODEL_VARIANT:
        config = Qwen38FlashNextFullConfig.billion_depth48_no_mtp()
    elif args.model_variant == WIDTH320_E32_MODEL_VARIANT:
        config = Qwen38FlashNextFullConfig.width320_e32_depth48_no_mtp()
    else:
        config = Qwen38FlashNextFullConfig.quarter_depth48_no_mtp()
    train_prefixes, validation_prefixes = _validated_data_prefixes(args.data_root)
    attention_variant = attention_variant_contract(args)
    _write_contract(args, config)
    sys.argv = _megatron_argv(args, config)

    import megatron.training.training as training_module
    from megatron.core import parallel_state
    from megatron.core.enums import ModelType
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.utils import get_pg_rank
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    probe_gradient_state = {"expected": False, "seen": False, "nonfinite": False}
    probe_models = []
    restored_gradients = {}

    def model_provider(
        pre_process=True,
        post_process=True,
        vp_stage=None,
        config=None,
        pg_collection=None,
    ):
        transformer_config = config or core_transformer_config_from_args(get_args())
        groups = pg_collection or ProcessGroupCollection.use_mpu_process_groups()
        pp_rank = get_pg_rank(groups.pp)
        if args.parallelism == "dp-only":
            sizes = _assert_dp_only_groups(groups)
            if torch.distributed.get_rank() == 0:
                _atomic_json(args.run_dir / "PARALLELISM.json", sizes)
        model = build_model(
            config_outer,
            transformer_config,
            groups,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        if (attention_variant is not None or args.initialization_reference is not None
                or (args.probe_steps and config_outer.attention_output_gate)):
            from archlab.megatron.simplicial_production import install_production_attention

            initialization = install_production_attention(model, args)
            if torch.distributed.get_rank() == 0:
                _atomic_json(args.run_dir / "INITIALIZATION.json", initialization)
        partition = _tag_native_optimizer_fallbacks(model)
        if config_outer.attention_output_gate and torch.distributed.get_rank() == 0:
            _atomic_json(
                args.run_dir / "MODEL_SHAPES.json",
                {
                    name: {
                        "shape": list(p.shape),
                        "parameters": p.numel(),
                        "optimizer": getattr(p, "archlab_optimizer", "adamw"),
                    }
                    for name, p in model.named_parameters()
                },
            )
        if args.probe_steps and config_outer.attention_output_gate:
            for suffix in (
                "attention.linear_qkv.gate.weight",
                "attention.q_layernorm.weight",
                "attention.k_layernorm.weight",
                "attention_residual.norm.weight",
                "ple.norm_query.weight",
            ):
                parameter = next(p for name, p in model.named_parameters() if name.endswith(suffix))
                restored_gradients[suffix] = {"seen": False, "nonfinite": False}

                def record(gradient, key=suffix, weight=parameter):
                    state = restored_gradients[key]
                    effective = _effective_probe_gradient(weight, gradient)
                    state["nonfinite"] |= not bool(torch.isfinite(effective).all())
                    state["seen"] |= bool(torch.count_nonzero(effective))
                    return gradient

                parameter.register_hook(record)
        if args.probe_steps and attention_variant is not None:
            from archlab.megatron.simplicial_attention import EXTRA_MARKERS

            for name, parameter in model.named_parameters():
                if not any(marker in name for marker in EXTRA_MARKERS):
                    continue
                restored_gradients[name] = {"seen": False, "nonfinite": False}

                def record_extra(gradient, key=name, weight=parameter):
                    state = restored_gradients[key]
                    effective = _effective_probe_gradient(weight, gradient)
                    state["nonfinite"] |= not bool(torch.isfinite(effective).all())
                    state["seen"] |= bool(torch.count_nonzero(effective))
                    return gradient

                parameter.register_hook(record_extra)
        if args.probe_steps:
            probe_models.append(model)
        if args.probe_steps or args.parallelism == "dp-only":
            counts = _probe_parameter_counts(
                model,
                data_replicas=groups.dp.size(),
                expert_replicas=groups.expt_dp.size(),
            )
            expected_count = parameter_count_contract(config_outer)["total"] + (
                attention_variant["extra_parameters"] if attention_variant else 0
            )
            if counts["total"] != expected_count:
                raise RuntimeError(f"native model parameter count differs from recipe: {counts}")
            if torch.distributed.get_rank() == 0:
                _atomic_json(
                    args.run_dir
                    / ("PROBE_PARAMETERS.json" if args.probe_steps else "MODEL_PARAMETERS.json"),
                    {**counts, "local_parameters": sum(p.numel() for p in model.parameters())},
                )
        if args.probe_steps and pp_rank == 0:
            early_parameter = next(
                (
                    parameter
                    for name, parameter in model.named_parameters()
                    if name.endswith("attention.in_proj_qkv.weight")
                ),
                None,
            )
            if early_parameter is None:
                raise RuntimeError("the Flash-Next probe could not identify an early GDN parameter")
            probe_gradient_state["expected"] = True

            def record_early_gradient(gradient):
                if not torch.isfinite(gradient).all():
                    probe_gradient_state["nonfinite"] = True
                elif torch.count_nonzero(gradient):
                    probe_gradient_state["seen"] = True
                return gradient

            early_parameter.register_hook(record_early_gradient)
        if torch.distributed.get_rank() == 0:
            _atomic_json(args.run_dir / "OPTIMIZER_PARTITION.json", partition)
        return model

    config_outer = config

    def datasets_provider(_sample_counts):
        dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
        dp_world = parallel_state.get_data_parallel_world_size(with_context_parallel=True)
        _assert_pipeline_data_rank_layout(
            global_rank=torch.distributed.get_rank(),
            data_parallel_rank=dp_rank,
            data_parallel_world_size=dp_world,
            pipeline_global_ranks=tuple(
                torch.distributed.get_process_group_ranks(
                    parallel_state.get_pipeline_model_parallel_group()
                )
            ),
            pipeline_world_size=len(config.pipeline_layers),
        )
        train = partition_prefixes_for_dp_rank(train_prefixes, dp_rank, dp_world)
        validation = partition_prefixes_for_dp_rank(
            validation_prefixes, dp_rank, dp_world, require_distinct=False
        )
        accumulation = args.global_batch_size // (dp_world * args.micro_batch_size)

        def train_batches():
            yield from DPRankTokenBatches(
                train,
                batch_size=args.micro_batch_size,
                sequence_len=config.sequence_len,
                start_batch=_current_iteration() * accumulation,
                device=torch.device("cuda", torch.cuda.current_device()),
            )

        def validation_batches():
            window = args.eval_iters * accumulation
            yield from DPRankTokenBatches(
                validation,
                batch_size=args.micro_batch_size,
                sequence_len=config.sequence_len,
                start_batch=dp_rank * window,
                repeat_window_batches=window,
                device=torch.device("cuda", torch.cuda.current_device()),
            )

        return train_batches(), validation_batches(), None

    datasets_provider.is_distributed = True
    _install_bounded_torch_dist_staging(training_module)
    invoke_pretrain(
        training_module,
        datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step=_forward_step,
    )
    if args.probe_steps:
        if config_outer.attention_output_gate:
            evidence = _probe_restored_replica_equality(
                probe_models[0], parallel_state.get_data_parallel_group()
            )
            bad = torch.tensor(
                sum(not s["seen"] or s["nonfinite"] for s in restored_gradients.values()),
                device=torch.cuda.current_device(),
                dtype=torch.int64,
            )
            torch.distributed.all_reduce(bad)
            if bad.item():
                raise RuntimeError(f"restored-component gradient gate failed: {restored_gradients}")
            if torch.distributed.get_rank() == 0:
                _atomic_json(
                    args.run_dir / "PROBE_RESTORED_COMPONENTS.json",
                    {
                        **evidence,
                        "gradients": restored_gradients,
                        "gradient_verified_ranks": torch.distributed.get_world_size(),
                    },
                )
        replica_evidence = _probe_ple_replica_equality(
            probe_models[0], parallel_state.get_expert_data_parallel_group()
        )
        if torch.distributed.get_rank() == 0:
            _atomic_json(args.run_dir / "PROBE_PLE_REPLICAS.json", replica_evidence)
        gradient_evidence = torch.tensor(
            [
                int(probe_gradient_state["expected"]),
                int(probe_gradient_state["seen"]),
                int(probe_gradient_state["nonfinite"]),
            ],
            device=torch.device("cuda", torch.cuda.current_device()),
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(gradient_evidence)
        expected, seen, nonfinite = gradient_evidence.tolist()
        expected_owners = torch.distributed.get_world_size() // len(config.pipeline_layers)
        if expected != expected_owners or seen != expected or nonfinite:
            raise RuntimeError(
                "Flash-Next probe early-backbone gradient rejection: "
                f"expected={expected}, seen={seen}, nonfinite={nonfinite}"
            )
        if torch.distributed.get_rank() == 0:
            _atomic_json(
                args.run_dir / "PROBE_GRADIENTS.json",
                {
                    "status": "passed",
                    "early_gdn_owner_ranks": expected,
                    "nonzero_gradient_ranks": seen,
                    "nonfinite_gradient_ranks": nonfinite,
                },
            )
    if torch.distributed.get_rank() == 0:
        _atomic_json(
            args.run_dir
            / ("PROBE_COMPLETE.json" if args.probe_steps else "TRAINING_COMPLETE.json"),
            {
                "iteration": _completed_checkpoint_iteration(args.run_dir),
                "completed_at_unix": time.time(),
            },
        )


def main() -> None:
    args = _parser().parse_args()
    if (
        min(
            args.sequence_length,
            args.micro_batch_size,
            args.global_batch_size,
            args.target_train_tokens,
            args.eval_interval,
            args.eval_iters,
            args.log_interval,
        )
        < 1
    ):
        raise SystemExit("all integer training controls must be positive")
    if args.sequence_length != 2_048:
        raise SystemExit("the supported training recipe is fixed at 2,048 tokens")
    if args.probe_steps < 0 or args.probe_save_interval < 0:
        raise SystemExit("probe controls must be non-negative")
    for path in (
        args.data_root / "DATA_READY.json",
        args.tokenizer / "tokenizer.json",
        args.tokenizer / "config.json",
    ):
        if not path.is_file():
            raise SystemExit(f"required artifact is missing: {path}")
    ready = json.loads((args.data_root / "DATA_READY.json").read_text())
    if ready.get("tokenizer_sha256") != TOKENIZER_SHA256:
        raise SystemExit("FineWeb-Edu data tokenizer hash does not match Qwen3.8")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "checkpoints").mkdir(exist_ok=True)
    _run(args)


if __name__ == "__main__":
    main()
