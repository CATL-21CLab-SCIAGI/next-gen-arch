# ruff: noqa: I001  # Preserve the checkpoint-qualified import order.
"""Controlled full-text-backbone continuation from matched 50M adapters."""

from __future__ import annotations
import argparse
import datetime
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback


def all_errors(errors):
    import torch.distributed as dist

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, errors)
    if any(gathered):
        raise RuntimeError(f"full-training admission failed: {gathered}")


def gradient_step(
    model,
    optimizer,
    indexers,
    batches,
    *,
    rate,
    audit=False,
    expected_loss=None,
    attention_masks=None,
    router_auxiliary=False,
):
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from archlab.optimizers.sharded_adafactor import local_tensor
    from archlab.automodel.deepseek_v41_full_indexer import set_indexer_loss_scale
    from archlab.automodel.deepseek_v41_training import emit

    rank, world = dist.get_rank(), dist.get_world_size()  # noqa: F841 — preserve checkpoint-qualified executable AST
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = rate
    set_indexer_loss_scale(indexers, coefficient=0.01, world_size=world, accumulation=len(batches))
    totals = torch.tensor(
        [sum(b[2] for b in batches), sum(b[0].numel() for b in batches)],
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(totals)
    target_count = int(totals[0])
    if target_count <= 0:
        raise ValueError("empty global target batch")
    loss_sum = torch.zeros((), device="cuda", dtype=torch.float64)
    start = time.perf_counter()
    for micro, (inputs, labels, local_targets) in enumerate(batches):
        if router_auxiliary:
            from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
            from archlab.optimizers.router_balance import auxiliary_scale

            MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
                auxiliary_scale(local_targets, target_count),
                device=inputs.device,
                dtype=torch.float32,
            )
        if audit:
            emit("full_forward_start", micro=micro, context=inputs.shape[1])
        forward_kwargs = (
            {} if attention_masks is None else {"attention_mask": attention_masks[micro]}
        )
        hidden = model(input_ids=inputs, return_hidden_states=True, **forward_kwargs).hidden_states
        loss = model.lm_head.loss(hidden, labels)
        loss_sum += loss.detach().double()
        if audit:
            emit(
                "full_backward_start",
                micro=micro,
                loss=float(loss.detach()),
                memory_gib=torch.cuda.memory_allocated() / 2**30,
            )
        (loss / target_count).backward()
        if audit:
            emit(
                "full_backward_complete",
                micro=micro,
                memory_gib=torch.cuda.memory_allocated() / 2**30,
            )
        del hidden, loss
    errors, coverage = [], []
    squares = torch.zeros((), device="cuda", dtype=torch.float64)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            errors.append(f"frozen parameter: {name}")
        if p.grad is None:
            errors.append(f"missing gradient: {name}")
            continue
        grad = local_tensor(p.grad)
        replicated = not isinstance(p, DTensor)
        if replicated:
            dist.all_reduce(grad)
        finite = torch.ones((), device="cuda", dtype=torch.bool)
        magnitude = torch.zeros((), device="cuda", dtype=torch.float64)
        nonzero = torch.zeros((), device="cuda", dtype=torch.int64)
        for chunk in grad.reshape(-1).split(16 * 1024 * 1024):
            finite &= torch.isfinite(chunk).all()
            magnitude += torch.linalg.vector_norm(chunk, dtype=torch.float32).double().square()
            if audit:
                nonzero += chunk.count_nonzero()
        if not bool(finite):
            errors.append(f"nonfinite gradient: {name}")
        squares += magnitude / (world if replicated else 1)
        if audit:
            coverage.append(
                {
                    "name": name,
                    "global_shape": list(p.shape),
                    "local_shape": list(grad.shape),
                    "dtype": str(p.dtype),
                    "gradient_dtype": str(grad.dtype),
                    "nonzero_local_elements": int(nonzero),
                    "gradient_l2_local": float(magnitude.sqrt()),
                    "placements": str(p.placements) if isinstance(p, DTensor) else "replicated",
                }
            )
    all_errors(errors)
    dist.all_reduce(squares)
    norm = float(squares.sqrt())
    if not math.isfinite(norm) or norm == 0:
        raise FloatingPointError(f"invalid full gradient norm: {norm}")
    clip = min(1.0, 1.0 / norm)
    if clip < 1:
        for p in model.parameters():
            local_tensor(p.grad).mul_(clip)
    dist.all_reduce(loss_sum)
    measured_loss = float(loss_sum / target_count)
    if expected_loss is not None and abs(measured_loss - expected_loss) > 1e-4:
        raise AssertionError(
            f"50M checkpoint forward equivalence failed: {measured_loss} vs {expected_loss}"
        )
    optimizer.step()
    torch.cuda.synchronize()
    maxima = torch.tensor(
        [time.perf_counter() - start, torch.cuda.max_memory_allocated() / 2**30],
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    metric = {
        "loss": float(loss_sum / target_count),
        "supervised_tokens": target_count,
        "input_tokens": int(totals[1]),
        "seconds": float(maxima[0]),
        "learning_rate": rate,
        "gradient_norm_before_clip": norm,
        "max_memory_allocated_gib": float(maxima[1]),
        "indexer_kl_local": [x._archlab_last_kl for x in indexers],
        **optimizer.last_metrics,
    }
    if not math.isfinite(metric["loss"]) or optimizer.last_metrics["changed_local_elements"] == 0:
        raise FloatingPointError(f"unhealthy full-model update: {metric}")
    if audit:
        metric["coverage"] = coverage
    return metric


def _fingerprint(model, optimizer):
    import torch
    from archlab.optimizers.sharded_adafactor import local_tensor

    h = hashlib.sha256()
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        h.update(name.encode())
        for part in local_tensor(p.detach()).reshape(-1).split(8 * 1024 * 1024):
            h.update(part.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    for p in optimizer.param_groups[0]["params"]:
        for key, value in sorted(optimizer.state[p].items()):
            h.update(key.encode())
            h.update(
                value.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                if isinstance(value, torch.Tensor)
                else str(value).encode()
            )
    h.update(torch.get_rng_state().numpy().tobytes())
    h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("simplicial", "normal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume-full", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--expected-initial-loss", type=float)
    parser.add_argument("--eval-interval-tokens", type=int, default=10000000)
    parser.add_argument("--eval-targets", type=int, default=64000)
    parser.add_argument("--control-policy", type=Path)
    parser.add_argument("--initial-window", action="store_true")
    args = parser.parse_args()
    faulthandler.enable(all_threads=True)
    faulthandler.register(signal.SIGUSR2, all_threads=True)
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(Path(os.environ["ARCHLAB_CONTAINER_KERNEL_PACKAGES"]))
    import torch
    import torch.distributed as dist
    from archlab.artifacts import atomic_write_json
    from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
    from archlab.automodel.deepseek_v41_official_execution import build_official_base
    from archlab.automodel.deepseek_v41_official_adapter import install_official_adapters
    from archlab.automodel.deepseek_v41_full_boundaries import install_full_training_boundaries
    from archlab.automodel.deepseek_v41_full_indexer import install_trainable_indexers
    from archlab.automodel.deepseek_v41_full_checkpoint import (
        save_full_checkpoint,
        restore_full_checkpoint,
    )
    from archlab.optimizers.sharded_adafactor import ShardedAdafactor, local_tensor
    from archlab.automodel.deepseek_v41_training import emit, append_metric
    from archlab.automodel.deepseek_v41_official_execution import official_core_hashes
    from archlab.automodel.deepseek_v41_full_validation import (
        make_plan,
        evaluate_pilot,
        evaluate_batches,
    )
    from archlab.automodel.deepseek_v41_control import (
        distributed_policy,
        pending_evaluation,
        checkpoint_due,
        evaluation_due,
        pin_checkpoint,
        publish_evaluation_request,
        admit_control_upgrade,
    )
    import subprocess

    source_root = Path(__file__).resolve().parents[3]
    source_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if os.environ.get("ARCHLAB_FULL_ALLOW_DEVELOPMENT") != "1":
        if (
            source_commit != os.environ["NGA_EXPECTED_COMMIT"]
            or subprocess.check_output(
                ["git", "-C", str(source_root), "status", "--porcelain"], text=True
            ).strip()
        ):
            raise ValueError("launch from the recorded clean immutable full-training source")
    elif not args.tiny:
        raise ValueError("production cannot use a mutable development source")
    implementation = official_core_hashes()
    for path in sorted((source_root / "src/archlab/automodel").glob("deepseek_v41_full*.py")):
        implementation[str(path.relative_to(source_root / "src/archlab"))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for relative in (
        "optimizers/sharded_adafactor.py",
        "architectures/deepseek_v41_indexer_objective.py",
    ):
        implementation[relative] = hashlib.sha256(
            (source_root / "src/archlab" / relative).read_bytes()
        ).hexdigest()
    for relative in (
        "automodel/deepseek_v41_control.py",
        "automodel/deepseek_v41_live_window.py",
        "serving/openai_chat.py",
    ):
        implementation[relative] = hashlib.sha256(
            (source_root / "src/archlab" / relative).read_bytes()
        ).hexdigest()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=90),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy = distributed_policy(args.control_policy) if args.control_policy else None
    front = None
    startup_activity = None
    try:
        if world != 16:
            raise ValueError("the controlled full-training contract requires 16 GPUs per variant")
        if args.eval_interval_tokens <= 0 or args.eval_targets <= 0:
            raise ValueError("periodic validation requires positive cadence and target budget")
        if rank == 0 and not args.tiny:
            from archlab.automodel.deepseek_v41_live_window import Activity

            recorded = (
                json.loads((args.resume_full / "COMPLETE.json").read_text())["cursor"]
                if args.resume_full
                else {"step": 307}
            )
            startup_activity = Activity(output, recorded)
            startup_activity.phase = "restoring"
            startup_activity.__enter__()
            if policy and args.variant == "normal":
                from archlab.serving.openai_chat import ChatFront

                front = ChatFront(
                    policy["chat_host"],
                    policy["chat_port"],
                    policy["chat_token_file"],
                    max_tokens=policy["chat_max_tokens"],
                )
                front.set_state("loading")
        validation = plan = None
        if not args.tiny:
            from archlab.automodel.deepseek_v41_data import MathPilot

            validation = MathPilot(
                Path(os.environ["ARCHLAB_DEEPSEEK_V41_VALIDATION_PILOT"]),
                expected_split="validation",
                expected_budget=1000000,
            )
            plan = make_plan(validation, args.eval_targets)
            if rank == 0:
                atomic_write_json(output / "VALIDATION_PLAN.json", plan)
        if args.tiny:
            from archlab.automodel.deepseek_v41_full_optimizer_probe import (
                qualify_distributed_optimizer,
            )

            atomic_write_json(
                output / f"rank-{rank:02d}-optimizer-oracle.json", qualify_distributed_optimizer()
            )
            from archlab.automodel.deepseek_v41_full_sparse_probe import qualify_sparse_sink

            atomic_write_json(
                output / f"rank-{rank:02d}-sparse-sink-oracle.json", qualify_sparse_sink()
            )
            from archlab.automodel.deepseek_v41_full_indexer_memory_probe import (
                qualify_indexer_memory,
            )
            from archlab.automodel.deepseek_v41_official_execution import tiny_official_config

            config = tiny_official_config(
                Path(os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"])
            ).text_config
            atomic_write_json(
                output / f"rank-{rank:02d}-indexer-memory-oracle.json",
                qualify_indexer_memory(config, sequence=128),
            )
        model, setup, loading = build_official_base(
            weights=Path(os.environ["ARCHLAB_DEEPSEEK_V41_CHECKPOINT"]),
            assets=Path(os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"]),
            ep_size=8,
            tiny=args.tiny,
        )
        adapters = install_official_adapters(
            model,
            V41AdapterConfig(width=256) if args.tiny else V41AdapterConfig(),
            layer_indices=(1, 3, 5) if args.tiny else (4, 9, 14, 19, 24, 29, 34, 39),
            device="cuda",
            backend="deterministic" if args.variant == "simplicial" else "flash-attn-deterministic",
            variant=args.variant,
        )
        cursor = {"step": 0, "supervised_tokens": 0, "phase_step": 0}
        if not args.tiny:
            if args.checkpoint is None:
                raise ValueError("full comparison must start at a verified 50M adapter checkpoint")
            state = torch.load(
                args.checkpoint / "adapter-state.pt", map_location="cpu", weights_only=False
            )
            if state["cursor"]["step"] != 307 or state["cursor"]["supervised_tokens"] != 50119869:
                raise ValueError("expected matched step307 / 50,119,869 target checkpoint")
            for layer, adapter in adapters.items():
                adapter.load_state_dict(state["adapters"][str(layer)], strict=True)
            cursor.update({k: state["cursor"][k] for k in ("step", "supervised_tokens")})
            del state
        boundaries = install_full_training_boundaries(model)
        indexers = install_trainable_indexers(model)
        optimizer = ShardedAdafactor(model.parameters(), lr=1e-4)
        logical = torch.tensor(
            sum(
                local_tensor(p).numel() / (world if not hasattr(p, "placements") else 1)
                for p in model.parameters()
            ),
            device="cuda",
            dtype=torch.float64,
        )
        dist.all_reduce(logical)
        contract = {
            "format": "archlab-v41-full-training-v1",
            "project_commit": source_commit,
            "implementation_sha256": implementation,
            "variant": args.variant,
            "world_size": world,
            "ep_size": 8,
            "expert_fsdp_size": 2,
            "engram_owners": world,
            "microbatch": 1,
            "accumulation": 2,
            "global_windows": 32,
            "full_text_parameters": int(logical),
            "all_parameters_unfrozen": True,
            "optimizer": "FP32-factored-Adafactor-stochastic-BF16",
            "peak_relative_lr": 1e-4,
            "warmup_phase_steps": 20,
            "weight_decay": 0,
            "indexer_kl_coefficient": 0.01,
            "indexer_sampled_queries": 64,
            "gradient_clip": 1.0,
            "cpu_offload": False,
            "start_checkpoint": None if args.checkpoint is None else str(args.checkpoint.resolve()),
            "indexer_selector_storage": "inplace-score-relu-and-multiply-byte-exact",
            "cuda_allocator": os.environ.get("PYTORCH_ALLOC_CONF"),
            "tiny": args.tiny,
            "runtime": {k: v for k, v in loading.items() if k != "local_parameter_gib"},
            "boundaries": boundaries,
        }
        contract["periodic_evaluation"] = {
            "interval_tokens": args.eval_interval_tokens,
            "targets": args.eval_targets,
            "plan_sha256": None if plan is None else plan["sha256"],
            "at_resume": True,
            "preserve_rng": True,
        }
        contract["control_protocol"] = "optimizer-boundary-eval-windows-v1"
        atomic_write_json(output / f"rank-{rank:02d}-loading.json", contract)
        if rank == 0:
            if (output / "RUN_CONTRACT.json").exists() and args.resume_full:
                previous = json.loads((output / "RUN_CONTRACT.json").read_text())
                atomic_write_json(
                    output / "contract-history" / f"{previous['project_commit']}.json", previous
                )
            else:
                atomic_write_json(output / "RUN_CONTRACT.json", contract)
        model.train()
        if args.tiny:
            from archlab.automodel.deepseek_v41_official_mesh_probe import _batch

            inputs, labels = _batch(rank, args.context)
            other_inputs, other_labels = _batch(rank + world, args.context)
            batches = [
                (inputs, labels, int((labels != -100).sum())),
                (other_inputs, other_labels, int((other_labels != -100).sum())),
            ]
            updates = []
            for phase_step in range(3):
                metric = gradient_step(model, optimizer, indexers, batches, rate=1e-3, audit=True)
                atomic_write_json(output / f"rank-{rank:02d}-update-{phase_step + 1}.json", metric)
                emit(
                    "full_tiny_update",
                    step=phase_step + 1,
                    **{k: v for k, v in metric.items() if k != "coverage"},
                )
                updates.append(metric)
            checkpoint = output / "checkpoint"
            before = _fingerprint(model, optimizer)
            save_full_checkpoint(checkpoint, model, optimizer, cursor, contract)
            torch.manual_seed(987)
            with torch.no_grad():
                for p in model.parameters():
                    local_tensor(p).zero_()
            restore_full_checkpoint(checkpoint, model, optimizer, contract)
            after = _fingerprint(model, optimizer)
            all_errors([] if before == after else ["full checkpoint state/RNG is not exact"])
            metric = gradient_step(model, optimizer, indexers, batches, rate=1e-3, audit=True)
            atomic_write_json(output / f"rank-{rank:02d}-restored-update.json", metric)
            saved_replay = {
                name: local_tensor(p).detach().cpu().clone() for name, p in model.named_parameters()
            }
            restore_full_checkpoint(checkpoint, model, optimizer, contract)
            before_eval = _fingerprint(model, optimizer)
            evaluation = evaluate_batches(model, batches)
            after_eval = _fingerprint(model, optimizer)
            all_errors(
                []
                if before_eval == after_eval and abs(evaluation["loss"] - metric["loss"]) < 1e-4
                else ["validation changed state/RNG or differs from training forward"]
            )
            atomic_write_json(
                output / f"rank-{rank:02d}-validation-oracle.json",
                {
                    "passed": True,
                    "state_rng_exact": True,
                    "ce_error": evaluation["loss"] - metric["loss"],
                    **evaluation,
                },
            )
            from archlab.automodel.deepseek_v41_live_window import qualify_window

            window_qualification = qualify_window(
                model,
                optimizer,
                checkpoint,
                batches,
                variant=args.variant,
                assets=Path(os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"]),
                weights=Path(os.environ["ARCHLAB_DEEPSEEK_V41_CHECKPOINT"]),
                source=source_root,
            )
            atomic_write_json(
                output / f"rank-{rank:02d}-eval-chat-window-oracle.json", window_qualification
            )
            repeat = gradient_step(model, optimizer, indexers, batches, rate=1e-3, audit=True)
            error_square, reference_square = 0.0, 0.0
            for name, p in model.named_parameters():
                current = local_tensor(p).detach().cpu().float()
                error_square += float((current - saved_replay[name].float()).square().sum())
                reference_square += float(saved_replay[name].float().square().sum())
            relative = math.sqrt(error_square / max(reference_square, 1e-30))
            all_errors(
                []
                if relative < 1e-4 and abs(repeat["loss"] - metric["loss"]) < 1e-4
                else [f"full checkpoint continuation differs: {relative}"]
            )
            atomic_write_json(
                output / f"rank-{rank:02d}-checkpoint-replay.json",
                {
                    "passed": True,
                    "relative_parameter_l2": relative,
                    "loss_difference": repeat["loss"] - metric["loss"],
                },
            )
            atomic_write_json(
                output / f"rank-{rank:02d}-qualified.json",
                {"passed": True, "checkpoint_exact": True, "full_parameter_count": int(logical)},
            )
            dist.barrier()
            if rank == 0:
                atomic_write_json(
                    output / "COMPLETE.json",
                    {
                        "passed": True,
                        "world_size": world,
                        "variant": args.variant,
                        "full_parameter_count": int(logical),
                        "checkpoint_exact": True,
                        "updates": 5,
                        "accumulation": 2,
                        "implementation_sha256": implementation,
                        "project_commit": source_commit,
                    },
                )
            return
        qualification = (
            {}
            if args.qualification is None
            else json.loads((args.qualification / "COMPLETE.json").read_text())
        )
        if (
            not qualification.get("passed")
            or qualification.get("variant") != args.variant
            or qualification.get("implementation_sha256") != implementation
            or qualification.get("accumulation") != 2
        ):
            raise ValueError(
                "the identical full-training source must pass tiny16 qualification with two accumulation steps"
            )
        from archlab.automodel.deepseek_v41_data import MathPilot

        data = MathPilot(
            Path(os.environ["ARCHLAB_DEEPSEEK_V41_TRAIN_PILOT"]),
            expected_split="train",
            expected_budget=1000000000,
        )
        if args.resume_full:
            saved_contract = json.loads((args.resume_full / "COMPLETE.json").read_text())[
                "contract"
            ]
            admit_control_upgrade(saved_contract, contract, gradient_step)
            cursor = restore_full_checkpoint(args.resume_full, model, optimizer, saved_contract)
            if rank == 0:
                atomic_write_json(
                    output / "RESUME.json",
                    {
                        "path": str(args.resume_full),
                        "cursor": cursor,
                        "saved_contract": saved_contract,
                        "eval_only_upgrade": saved_contract != contract,
                    },
                )
                previous_path = output / "RUN_CONTRACT.json"
                previous = (
                    json.loads(previous_path.read_text())
                    if previous_path.exists()
                    else saved_contract
                )
                atomic_write_json(
                    output / "SOURCE_TRANSITION.json",
                    {
                        "from_commit": previous["project_commit"],
                        "to_commit": source_commit,
                        "cursor": cursor,
                        "training_math_unchanged": True,
                        "change": "optimizer-boundary checkpoint/evaluation/chat scheduling",
                    },
                )
                atomic_write_json(previous_path, contract)
        if (
            sum(w["targets"] for w in data.windows[: cursor["step"] * 32])
            != cursor["supervised_tokens"]
        ):
            raise ValueError("checkpoint cursor disagrees with the sealed training order")
        last_save = cursor["supervised_tokens"]

        def window(checkpoint):
            from archlab.automodel.deepseek_v41_live_window import run_window

            complete = cursor["step"] * 32 >= len(data)
            verification = [
                validation.batch(micro * world + rank, device="cuda")
                if complete
                else data.batch(cursor["step"] * 32 + micro * world + rank, device="cuda")
                for micro in range(2)
            ]
            return run_window(
                model,
                optimizer,
                cursor,
                checkpoint,
                policy,
                assets=Path(os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"]),
                weights=Path(os.environ["ARCHLAB_DEEPSEEK_V41_CHECKPOINT"]),
                front=front,
                output=output,
                verification_batches=verification,
                final=complete,
            )

        def validate():
            metric = evaluate_pilot(model, validation, plan)
            metric.update(
                {"step": cursor["step"], "consumed_supervised_tokens": cursor["supervised_tokens"]}
            )
            emit("full_validation", **metric)
            if rank == 0:
                append_metric(output / "validation.jsonl", metric)
            return metric

        validate()
        if startup_activity:
            startup_activity.__exit__(None, None, None)
            startup_activity = None
        if args.initial_window:
            if args.variant != "normal" or not policy or not args.resume_full:
                raise ValueError(
                    "initial evaluation window requires the baseline policy and restored checkpoint"
                )
            window(args.resume_full)
        next_eval = (
            cursor["supervised_tokens"] // args.eval_interval_tokens + 1
        ) * args.eval_interval_tokens
        while cursor["step"] * 32 < len(data):
            batches = [
                data.batch(cursor["step"] * 32 + micro * world + rank, device="cuda")
                for micro in range(2)
            ]
            rate = 1e-4 * min(1.0, (cursor["phase_step"] + 1) / 20)
            metric = gradient_step(
                model,
                optimizer,
                indexers,
                batches,
                rate=rate,
                audit=cursor["phase_step"] < 2,
                expected_loss=args.expected_initial_loss if cursor["phase_step"] == 0 else None,
            )
            cursor["phase_step"] += 1
            cursor["step"] += 1
            cursor["supervised_tokens"] += metric["supervised_tokens"]
            metric.update(
                {
                    "step": cursor["step"],
                    "phase_step": cursor["phase_step"],
                    "consumed_supervised_tokens": cursor["supervised_tokens"],
                }
            )
            if "coverage" in metric:
                atomic_write_json(
                    output / f"rank-{rank:02d}-gradient-coverage-{cursor['phase_step']}.json",
                    metric.pop("coverage"),
                )
            emit("full_train_update", **metric)
            if rank == 0:
                append_metric(output / "train-metrics.jsonl", metric)
            if cursor["supervised_tokens"] >= next_eval:
                validate()
                next_eval = (
                    cursor["supervised_tokens"] // args.eval_interval_tokens + 1
                ) * args.eval_interval_tokens
            stop = torch.tensor(
                int(
                    (output / "STOP_REQUEST").exists()
                    or (args.steps > 0 and cursor["phase_step"] >= args.steps)
                ),
                device="cuda",
            )
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if policy:
                policy = distributed_policy(args.control_policy)
            waiting = torch.tensor(
                int(
                    bool(policy)
                    and args.variant == "normal"
                    and rank == 0
                    and pending_evaluation(policy, "simplicial") is not None
                ),
                device="cuda",
            )
            dist.all_reduce(waiting, op=dist.ReduceOp.MAX)
            scheduled = (
                checkpoint_due(cursor["step"], policy)
                if policy
                else cursor["supervised_tokens"] - last_save >= 20000000
            )
            if (
                cursor["phase_step"] == 5
                or scheduled
                or bool(waiting)
                or bool(stop)
                or cursor["step"] * 32 >= len(data)
            ):
                checkpoint = output / "checkpoints" / f"step-{cursor['step']:06d}"
                save_full_checkpoint(checkpoint, model, optimizer, cursor, contract)
                last_save = cursor["supervised_tokens"]
                emit("full_checkpoint_complete", **cursor)
                if policy and rank == 0:
                    pin_checkpoint(checkpoint, cursor, policy)
                    if evaluation_due(cursor["step"], policy) or cursor["step"] * 32 >= len(data):
                        publish_evaluation_request(
                            policy["evaluation_requests"], checkpoint, cursor, args.variant
                        )
                if (
                    policy
                    and args.variant == "normal"
                    and (evaluation_due(cursor["step"], policy) or bool(waiting))
                    and not bool(stop)
                ):
                    window(checkpoint)
            if cursor["phase_step"] >= 5 and rank == 0:
                atomic_write_json(
                    output / "TRAINING_HEALTHY.json", {"passed": True, **cursor, "latest": metric}
                )
            if bool(stop):
                emit("full_training_paused", **cursor)
                break
        if policy and args.variant == "normal" and cursor["step"] * 32 >= len(data):
            checkpoint = output / "checkpoints" / f"step-{cursor['step']:06d}"
            while True:
                end = torch.tensor(int((output / "STOP_SERVING").exists()), device="cuda")
                dist.all_reduce(end, op=dist.ReduceOp.MAX)
                if bool(end):
                    break
                window(checkpoint)
    except BaseException:
        atomic_write_json(
            output / f"rank-{rank:02d}-failure.json", {"traceback": traceback.format_exc()}
        )
        raise
    finally:
        if startup_activity:
            startup_activity.__exit__(None, None, None)
        if front:
            front.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
