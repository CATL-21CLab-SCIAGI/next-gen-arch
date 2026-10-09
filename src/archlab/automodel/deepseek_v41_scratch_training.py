"""Qualified scratch comparisons with exact full-state checkpoints."""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import time
import traceback
from pathlib import Path


def canonical_contract(value):
    """Use the exact JSON representation compared by full checkpoint restoration."""
    return json.loads(json.dumps(value, allow_nan=False))


def training_runtime(loading):
    return {key: value for key, value in loading.items() if key != "local_parameter_gib"}


def learning_rate(step, tokens, *, warmup=100, budget=10_000_000_000, peak=0.01, floor=0.001):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, max(0.0, tokens / budget))
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def training_budgets(matched_contract, prefix_tokens=None, cell=None):
    """Shorten a matched production prefix without compressing its historical LR schedule."""
    schedule = (matched_contract["training"]["supervised_tokens"] if matched_contract else
                10_000_000_000 if cell is None else cell["unique_targets"] * cell["epochs"])
    if prefix_tokens is None:
        return schedule, schedule
    if (matched_contract is None or cell is not None or type(prefix_tokens) is not int
            or not 0 < prefix_tokens <= schedule or prefix_tokens % 5):
        raise ValueError("training prefix requires a matched corpus prefix and five equal milestones")
    return prefix_tokens, schedule


def fingerprint(model, *, common_only=False):
    import torch

    from archlab.optimizers.sharded_adafactor import local_tensor

    h = hashlib.sha256()
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        if common_only and ".simplicial_adapter." in name:
            continue
        t = local_tensor(p.detach())
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        for part in t.reshape(-1).split(4 * 1024 * 1024):
            h.update(part.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_fingerprint(model, optimizer):
    from archlab.automodel.deepseek_v41_full_training import _fingerprint

    return _fingerprint(model, optimizer)


def build_batches(data, window_cursor, rank, *, microbatch=8, accumulation=1, world_size=8, trim_alignment=None):
    return [
        data.batch(
            [
                window_cursor + (micro * world_size + rank) * microbatch + i
                for i in range(microbatch)
            ],
            device="cuda",
            **({"trim_alignment": trim_alignment} if trim_alignment is not None else {}),
        )
        for micro in range(accumulation)
    ]


def recovery_checkpoint_due(step, *, start_step, validation_due, interval=100, startup_updates=10):
    """Protect trained state before validation and between evaluation milestones."""
    if interval < 1 or startup_updates < 1 or step < start_step:
        raise ValueError("invalid recovery checkpoint schedule")
    return validation_due or step == start_step + startup_updates or step % interval == 0


def checkpoint_cursor(cursor, *, next_validation_tokens):
    """Remember an unfinished evaluation without changing the live data cursor."""
    return {**cursor, "pending_validation_tokens": (
        next_validation_tokens if cursor["supervised_tokens"] >= next_validation_tokens else None
    )}


def consume_pending_validation(cursor):
    """Retry a saved pre-validation boundary before another training update."""
    pending = cursor.get("pending_validation_tokens")
    if pending is not None and (
        type(pending) is not int or pending <= 0 or pending % 100_000_000
        or pending > cursor["supervised_tokens"]
    ):
        raise ValueError("checkpoint has an invalid pending validation boundary")
    cursor.pop("pending_validation_tokens", None)
    return pending


def record_recovery_checkpoint(output, checkpoint, cursor, contract, *, keep=2):
    """Publish a completed recovery save before pruning this run's older saves."""
    from archlab.artifacts import atomic_write_json

    if keep < 1:
        raise ValueError("recovery retention requires at least one full state")
    root = Path(output).resolve() / "recovery"
    checkpoint = Path(checkpoint)
    catalog_path = Path(output) / "RECOVERY_CHECKPOINTS.json"
    catalog = {"checkpoints": []}
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text())
        if catalog.get("format") != "archlab-v41-scratch-recovery-v1":
            raise ValueError("unrecognized recovery catalog format")
    records = list(catalog["checkpoints"])
    if records and cursor["step"] <= records[-1]["cursor"]["step"]:
        raise ValueError("recovery checkpoints must advance the training cursor")
    row = {"path": str(checkpoint), "cursor": dict(cursor)}
    for item in [*records, row]:
        path = Path(item["path"])
        if (path.is_symlink() or path.resolve().parent != root
                or path.name != f"step-{item['cursor']['step']:06d}"):
            raise ValueError("recovery catalog points outside this run's owned checkpoints")
        marker_bytes = (path / "COMPLETE.json").read_bytes()
        digest = hashlib.sha256(marker_bytes).hexdigest()
        if item is not row and item.get("complete_marker_sha256") != digest:
            raise ValueError("recorded recovery checkpoint completion marker changed")
        marker = json.loads(marker_bytes)
        if (marker.get("format") != "archlab-v41-full-sharded-v1"
                or marker["cursor"] != item["cursor"] or marker["contract"] != contract):
            raise ValueError("recovery checkpoint is incomplete or differs from this run")
        if item is row:
            row["complete_marker_sha256"] = digest
    records.append(row)
    retained, obsolete = records[-keep:], records[:-keep]
    atomic_write_json(catalog_path, {"format": "archlab-v41-scratch-recovery-v1",
                                   "keep": keep, "checkpoints": retained}, allow_nan=False)
    for item in obsolete:
        shutil.rmtree(item["path"])


def summarize_training_timings(records, *, warmup_updates):
    """Use measured real targets and wall time; cold updates never set the ETA."""
    measured = records[warmup_updates:]
    if warmup_updates < 0 or not measured:
        raise ValueError("training timing requires measured updates after warmup")
    if any(row["wall_seconds"] <= 0 or row["supervised_tokens"] <= 0 for row in measured):
        raise ValueError("training timing requires positive time and supervised targets")
    seconds = sum(row["wall_seconds"] for row in measured)
    targets = sum(row["supervised_tokens"] for row in measured)
    rate = targets / seconds
    return dict(
        scope="qualification-real-data-updates; not a production learning curve",
        warmup_updates=warmup_updates, measured_updates=len(measured),
        measured_supervised_tokens=targets, measured_wall_seconds=seconds,
        median_step_seconds=statistics.median(row["wall_seconds"] for row in measured),
        valid_tokens_per_second=rate,
        projected_10B_training_seconds=10_000_000_000 / rate,
        excludes="validation and checkpoint IO; data loading and router updates included",
        records=records,
    )


def measure_real_data_updates(model, indexers, gates, optimizer, contract, output):
    """Benchmark the same five sealed batches in every matched arm, then discard."""
    import torch
    import torch.distributed as dist

    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_full_training import gradient_step
    from archlab.automodel.deepseek_v41_scratch_data import ScratchData
    from archlab.optimizers.router_balance import balance_routers

    data = ScratchData(os.environ["ARCHLAB_DEEPSEEK_V41_SCRATCH_DATA"])
    execution = contract.get("execution", {})
    optimized = execution.get("optimized_synchronization", False)
    rank, world = dist.get_rank(), dist.get_world_size()
    window_cursor = 0
    records = []
    for step in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        batches = build_batches(
            data, window_cursor, rank, microbatch=contract["microbatch"],
            accumulation=contract["accumulation"], world_size=world,
            trim_alignment=execution.get("trim_alignment"),
        )
        metric = gradient_step(
            model, optimizer, indexers, batches, rate=0.001,
            router_auxiliary=True, optimized=optimized,
            attention_masks=[batch[1] != -100 for batch in batches],
        )
        metric.update(balance_routers(gates, contract["router_bias_update"], proportional=True,
                                     batched_metrics=optimized))
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter() - start, device="cuda", dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        window_cursor += contract["global_windows"]
        if data.targets_before(window_cursor) != sum(r["supervised_tokens"] for r in records) + metric["supervised_tokens"]:
            raise ValueError("timing smoke lost or repeated sealed data targets")
        records.append(dict(step=step + 1, window_cursor=window_cursor,
                            supervised_tokens=metric["supervised_tokens"],
                            input_tokens=metric["input_tokens"],
                            wall_seconds=float(elapsed), loss=metric["loss"],
                            peak_memory_gib=metric["max_memory_allocated_gib"]))
        atomic_write_json(output / f"rank-{rank:02d}-real-data-timing.json",
                          dict(status="running", records=records))
    summary = summarize_training_timings(records, warmup_updates=2)
    atomic_write_json(output / f"rank-{rank:02d}-real-data-timing.json", dict(status="complete", **summary))
    return summary


def evaluate(model, data, *, targets, step):
    import torch
    import torch.distributed as dist

    was_training = model.training
    model.eval()
    total = torch.zeros(2, device="cuda", dtype=torch.float64)
    index = 0
    start = time.perf_counter()
    try:
        with torch.no_grad():
            while int(total[1]) < targets:
                ids, labels, count = data.batch(
                    [index * dist.get_world_size() + dist.get_rank()], device="cuda"
                )
                hidden = model(
                    input_ids=ids, attention_mask=labels != -100, return_hidden_states=True
                ).hidden_states
                loss = model.lm_head.loss(hidden, labels)
                packet = torch.tensor([float(loss), count], device="cuda", dtype=torch.float64)
                dist.all_reduce(packet)
                total += packet
                index += 1
                if count == 0 and float(packet[1]) == 0:
                    raise ValueError("validation data ended early")
            if int(total[1]) != targets:
                raise ValueError("validation target budget differs")
            loss = float(total[0] / total[1])
            if not math.isfinite(loss):
                raise FloatingPointError("nonfinite scratch validation loss")
            return {
                "step": step,
                "loss": loss,
                "supervised_tokens": int(total[1]),
                "seconds": time.perf_counter() - start,
            }
    finally:
        model.train(was_training)


def run_qualification(model, indexers, gates, optimizer, contract, output, *, context):
    import torch
    import torch.distributed as dist

    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_full_checkpoint import (
        restore_full_checkpoint,
        save_full_checkpoint,
    )
    from archlab.automodel.deepseek_v41_full_optimizer_probe import qualify_distributed_optimizer
    from archlab.automodel.deepseek_v41_full_training import gradient_step
    from archlab.automodel.deepseek_v41_official_mesh_probe import _batch
    from archlab.optimizers.router_balance import balance_routers

    performance = contract.get("performance", contract.get("execution", {}))
    optimized = performance.get("optimized_synchronization", False)
    router_rate = performance.get("router_bias_rate", 0.01)
    sparse = contract["runtime"].get("sparse_precision", {})
    allow_atomic = (
        sparse.get("kv_gradient_reduction") == "atomic-fp32"
        and sparse.get("bitwise_next_update") is False
    )
    sparse_oracle = None
    if allow_atomic:
        from archlab.automodel.deepseek_v41_sparse_qualification import qualify_batched_sparse

        sparse_oracle = qualify_batched_sparse(
            head_dim=contract["runtime"]["geometry"]["text_config"]["head_dim"]
        )
        atomic_write_json(output / f"rank-{dist.get_rank():02d}-sparse-oracle.json", sparse_oracle)
    rank, world = dist.get_rank(), dist.get_world_size()
    atomic_write_json(
        output / f"rank-{rank:02d}-optimizer-oracle.json", qualify_distributed_optimizer()
    )
    batches = []
    for micro in range(2):
        ids, labels = _batch(rank + micro * world, 128)
        batches.append((ids, labels, int((labels != -100).sum())))
    masks = [torch.ones_like(b[1], dtype=torch.bool) for b in batches]
    common = fingerprint(model, common_only=True)
    atomic_write_json(
        output / f"rank-{rank:02d}-initial.json",
        {"common_sha256": common, "initialization_seed": 42},
    )
    metrics = []
    for step in range(2):
        metric = gradient_step(
            model,
            optimizer,
            indexers,
            batches,
            rate=0.001,
            router_auxiliary=True,
            optimized=optimized,
            audit=step == 0,
            attention_masks=masks,
        )
        metric.update(balance_routers(gates, router_rate, proportional=True, batched_metrics=optimized))
        metric.pop("coverage", None)
        metrics.append(metric)
        atomic_write_json(output / f"rank-{rank:02d}-qualification-update-{step + 1}.json", metric)
    cursor = {
        "step": 2,
        "phase_step": 2,
        "supervised_tokens": sum(m["supervised_tokens"] for m in metrics),
        "window_cursor": 4 * world,
    }
    before = state_fingerprint(model, optimizer)
    checkpoint = output / "checkpoint"
    if contract.get("scaling_study") or contract.get("matched_mixer_study"):
        from archlab.storage.oss_checkpoint import prepare_distributed_checkpoint

        checkpoint = prepare_distributed_checkpoint(
            checkpoint,
            Path(os.environ["ARCHLAB_SCALING_CHECKPOINT_ROOT"])
            / (f"{contract['variant']}-qualification" if contract.get("matched_mixer_study") else output.name)
            / "checkpoint",
        )
    save_full_checkpoint(checkpoint, model, optimizer, cursor, contract)
    first = gradient_step(
        model,
        optimizer,
        indexers,
        batches,
        rate=0.001,
        router_auxiliary=True,
        optimized=optimized,
        attention_masks=masks,
    )
    first.update(balance_routers(gates, router_rate, proportional=True, batched_metrics=optimized))
    expected = state_fingerprint(model, optimizer)
    restored = restore_full_checkpoint(checkpoint, model, optimizer, contract)
    if restored != cursor or state_fingerprint(model, optimizer) != before:
        raise AssertionError("full scratch checkpoint restore is not exact")
    repeated = gradient_step(
        model,
        optimizer,
        indexers,
        batches,
        rate=0.001,
        router_auxiliary=True,
        optimized=optimized,
        attention_masks=masks,
    )
    repeated.update(balance_routers(gates, router_rate, proportional=True, batched_metrics=optimized))
    actual = state_fingerprint(model, optimizer)
    if (expected != actual and not allow_atomic) or first["loss"] != repeated["loss"]:
        raise AssertionError(
            f"next-update replay differs: {first['loss']} vs {repeated['loss']}; state_equal={expected == actual}"
        )
    # Exercise the real pretraining context and production microbatch size.
    generator = torch.Generator(device="cpu").manual_seed(702 + rank)
    ids = torch.randint(3, 1000, (2, context), generator=generator).cuda()
    labels = ids.roll(-1, dims=1)
    labels[:, -1] = 1
    tail = context // 3
    ids[1, tail:] = 2
    labels[1, tail:] = -100
    ids = ids.repeat(contract["microbatch"] // 2, 1)
    labels = labels.repeat(contract["microbatch"] // 2, 1)
    peak = gradient_step(
        model,
        optimizer,
        indexers,
        [(ids, labels, int((labels != -100).sum()))] * contract["accumulation"],
        rate=0.001,
        router_auxiliary=True,
        optimized=optimized,
        attention_masks=[labels != -100] * contract["accumulation"],
    )
    peak.update(balance_routers(gates, router_rate, proportional=True, batched_metrics=optimized))
    empty_labels = labels.clone()
    if rank:
        empty_labels.fill_(-100)
    empty = gradient_step(
        model,
        optimizer,
        indexers,
        [(ids, empty_labels, int((empty_labels != -100).sum()))],
        rate=0.001,
        router_auxiliary=True,
        optimized=optimized,
        attention_masks=[empty_labels != -100],
    )
    empty.update(balance_routers(gates, router_rate, proportional=True, batched_metrics=optimized))
    real_data_timing = (
        measure_real_data_updates(model, indexers, gates, optimizer, contract, output)
        if contract.get("matched_mixer_study") else None
    )
    post_training_validation = None
    post_validation_update = None
    if contract.get("matched_mixer_study"):
        from archlab.automodel.deepseek_v41_scratch_data import ScratchData

        validation = ScratchData(os.environ["ARCHLAB_DEEPSEEK_V41_SCRATCH_DATA"], split="validation")
        post_training_validation = evaluate(model, validation, targets=1_000_000, step=0)
        atomic_write_json(output / f"rank-{rank:02d}-post-training-validation.json", post_training_validation)
        # Validation uses singleton batches after variable trimmed training
        # batches. Qualify the transition back to training as well: finite
        # gradients and a nonempty update remain enforced by gradient_step.
        post_validation_update = gradient_step(
            model, optimizer, indexers, batches, rate=0.001, router_auxiliary=True,
            optimized=optimized, attention_masks=masks,
        )
        post_validation_update.update(balance_routers(
            gates, router_rate, proportional=True, batched_metrics=optimized
        ))
        atomic_write_json(output / f"rank-{rank:02d}-post-validation-update.json", post_validation_update)
    receipt = {
        "passed": True,
        "exact_checkpoint_restore": True,
        "identical_next_update": expected == actual,
        "identical_next_update_loss": True,
        "identical_next_update_state": expected == actual,
        "atomic_gradient_replay": allow_atomic,
        "sparse_oracle": sparse_oracle,
        "common_initial_sha256": common,
        "production_context": context,
        "production_microbatch": contract["microbatch"],
        "production_accumulation": contract["accumulation"],
        "metrics": metrics,
        "production_shape_metric": peak,
        "empty_rank_metric": empty,
        "real_data_timing": real_data_timing,
        "post_training_validation": post_training_validation,
        "post_validation_update": post_validation_update,
    }
    atomic_write_json(output / f"rank-{rank:02d}-qualified.json", receipt)
    dist.barrier()
    if rank == 0:
        atomic_write_json(
            output / "QUALIFIED.json",
            {
                "passed": True,
                "contract": contract,
                "world_size": world,
                "context": context,
                "ranks": [f"rank-{i:02d}-qualified.json" for i in range(world)],
                "post_training_validation_passed": post_training_validation is not None,
                "post_validation_update_passed": post_validation_update is not None,
            },
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant", choices=["normal", "simplicial", "linear", "linsimp", "gdn", "triadic"], required=True
    )
    p.add_argument("--mode", choices=["qualify", "train"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--scaling-study", action="store_true")
    p.add_argument("--performance-contract", type=Path)
    p.add_argument("--sweep-cell", type=Path)
    p.add_argument("--matched-mixer-contract", type=Path)
    p.add_argument("--training-prefix-tokens", type=int)
    p.add_argument("--microbatch", type=int)
    p.add_argument("--accumulation", type=int)
    p.add_argument(
        "--sparse-backend", choices=["deterministic", "batched"], default="deterministic"
    )
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--steps", type=int, default=0)
    a = p.parse_args()
    matched_contract = matched_mixer = None
    if a.matched_mixer_contract is not None:
        from archlab.automodel.triadic_scratch_qualification import (
            mixer_specification,
            read_matched_contract,
        )

        matched_contract = read_matched_contract(a.matched_mixer_contract)
        matched_mixer = mixer_specification(matched_contract, a.variant)
        spec_training = matched_contract["training"]
        if (a.tiny or a.scaling_study or a.sweep_cell is not None or a.performance_contract is not None
                or a.width != matched_mixer["backbone"]["width"]
                or a.context != spec_training["sequence_length"]
                or a.microbatch != spec_training["micro_batch_size"]
                or a.accumulation != spec_training["accumulation"]):
            p.error("matched scratch arguments differ from the registered study")
    elif a.microbatch is not None or a.accumulation is not None or a.variant in ("gdn", "triadic"):
        p.error("new mixer/batch geometry requires an explicit matched scratch contract")
    from archlab.automodel.deepseek_v41_performance import (
        configure_performance,
        read_performance_contract,
    )

    performance = read_performance_contract(a.performance_contract)
    execution = performance or (matched_contract["training"]["execution"] if matched_contract else {})
    if performance and not a.scaling_study:
        p.error("the performance contract requires the 16-rank scaling study")
    cell = None if a.sweep_cell is None else json.loads(a.sweep_cell.read_text())
    if cell is not None:
        from archlab.automodel.loop_regularization import validate_cell

        validate_cell(cell)
    budget, schedule_budget = training_budgets(matched_contract, a.training_prefix_tokens, cell)
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    resolved_kernel_packages = select_container_kernel_packages(
        Path(os.environ["ARCHLAB_CONTAINER_KERNEL_PACKAGES"])
    )
    import torch
    import torch.distributed as dist

    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_full_checkpoint import (
        restore_full_checkpoint,
        save_full_checkpoint,
    )
    from archlab.automodel.deepseek_v41_full_training import gradient_step
    from archlab.automodel.deepseek_v41_official_execution import official_core_hashes
    from archlab.automodel.deepseek_v41_scratch_construct import construct_scratch
    from archlab.automodel.deepseek_v41_scratch_data import ScratchData
    from archlab.automodel.deepseek_v41_training import append_metric, emit
    from archlab.optimizers.router_balance import balance_routers
    from archlab.optimizers.sharded_adafactor import ShardedAdafactor

    faulthandler.enable(all_threads=True)
    faulthandler.register(signal.SIGUSR2, all_threads=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=90),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    rank = dist.get_rank()
    world = dist.get_world_size()
    microbatch = a.microbatch if matched_contract else performance["microbatch"] if performance else (4 if a.scaling_study else 8)
    accumulation = a.accumulation if matched_contract else 1
    evaluation_checkpoints = bool(a.scaling_study or matched_contract)
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        source = Path(__file__).resolve().parents[3]
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if (
            commit != os.environ["NGA_EXPECTED_COMMIT"]
            or subprocess.check_output(
                ["git", "-C", str(source), "status", "--porcelain"], text=True
            ).strip()
        ):
            raise ValueError("scratch training requires its recorded clean immutable source")
        code = official_core_hashes()
        paths = [
            *list((source / "src/archlab/automodel").glob("deepseek_v41_full*.py")),
            *list((source / "src/archlab/automodel").glob("deepseek_v41_scratch*.py")),
            source / "src/archlab/architectures/deepseek_v41_scratch.py",
            source / "src/archlab/optimizers/router_balance.py",
            source / "src/archlab/optimizers/sharded_adafactor.py",
            source / "src/archlab/optimizers/sinkhorn.py",
            source / "src/archlab/architectures/engram_scaling.py",
            source / "src/archlab/automodel/deepseek_v41_performance.py",
        ]
        paths += [
            source / "src/archlab/architectures/linsimp_attention.py",
            source / "src/archlab/architectures/deepseek_v41_linsimp_adapter.py",
            source / "src/archlab/automodel/deepseek_v41_official_adapter.py",
            source / "src/archlab/automodel/deepseek_v41_sparse_qualification.py",
        ]
        if a.scaling_study:
            paths.append(source / "src/archlab/storage/oss_checkpoint.py")
        if matched_contract is not None:
            paths += [
                source / "src/archlab/architectures/triadic_attention.py",
                source / "src/archlab/architectures/deepseek_v41_triadic_adapter.py",
                source / "src/archlab/architectures/deepseek_v41_matched_mixer.py",
                source / "src/archlab/automodel/triadic_scratch_qualification.py",
                source / "src/archlab/storage/oss_checkpoint.py",
            ]
        if cell is not None:
            paths += [
                source / "src/archlab/architectures/prelude_loop_coda.py",
                source / "src/archlab/automodel/deepseek_v41_loop.py",
                source / "src/archlab/automodel/loop_regularization.py",
                source / "src/archlab/automodel/repeated_scratch_data.py",
            ]
        for path in paths:
            code[str(path.relative_to(source / "src/archlab"))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        model, indexers, gates, loading = construct_scratch(
            base_config=os.environ["ARCHLAB_DEEPSEEK_V41_BASE_CONFIG"],
            assets=os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"],
            variant=a.variant,
            tiny=a.tiny,
            width=a.width,
            sparse_backend=a.sparse_backend,
            scaling_study=a.scaling_study,
            sweep_cell=cell,
            matched_mixer=matched_mixer,
            **({"grouped_experts": performance["grouped_experts"],
                "scale_engram": performance["scale_engram"],
                "engram_anchor_width": performance.get("engram_anchor_width"),
                "retain_activations": performance["retain_activations"],
                "expert_dispatcher": performance["expert_dispatcher"],
                "simplicial_backend": performance.get("simplicial_backend", "deterministic")} if performance else {}),
        )
        loading["resolved_kernel_packages"] = resolved_kernel_packages
        configure_performance(model, indexers, performance, a.context)
        if matched_contract is not None:
            # Cropping only masked suffixes keeps the original sampled query
            # positions, rather than choosing a different indexer objective.
            for indexer in indexers:
                indexer._archlab_sample_context = a.context
                indexer._archlab_fixed_sample_grid = True
        weight_decay = 0.0 if cell is None else cell["weight_decay"]
        if performance:
            from archlab.optimizers.sinkhorn import EngramSinkhornAdafactor

            optimizer = EngramSinkhornAdafactor(
                model.named_parameters(), lr=0.01, weight_decay=weight_decay,
                table_lr_scale=performance["table_lr_scale"], batched_updates=performance["batched_updates"],
            )
        else:
            optimizer = ShardedAdafactor(model.parameters(), lr=0.01, weight_decay=weight_decay)
        data_root = Path(os.environ["ARCHLAB_DEEPSEEK_V41_SCRATCH_DATA"])
        data_contract = json.loads((data_root / "CONTRACT.json").read_text())
        if (
            data_contract["training_targets"] != 10_000_000_000
            or data_contract["validation_targets"] != 1_000_000
            or data_contract["sequence"] != a.context
        ):
            raise ValueError("sealed data must match the 10B train / 1M validation token contract")
        contract = {
            "format": "archlab-v41-scratch-comparison-v1",
            "project_commit": commit,
            "implementation_sha256": code,
            "variant": a.variant,
            "tiny": a.tiny,
            "world_size": world,
            "runtime": training_runtime(loading),
            "data_contract": data_contract,
            "context": a.context,
            "microbatch": microbatch,
            "accumulation": accumulation,
            "global_windows": world * microbatch * accumulation,
            "optimizer": "FP32-factored-Adafactor-stochastic-BF16",
            "peak_relative_learning_rate": 0.01,
            "minimum_relative_learning_rate": 0.001,
            "warmup_updates": 100,
            "target_supervised_tokens": budget,
            "gradient_clip": 1.0,
            "router_auxiliary_loss_coefficient": 0.01,
            "router_bias_update": 0.01,
            "router_bias_rule": "centered proportional load error clipped [-5,1]",
            "indexer_kl_coefficient": 0.01,
            "cpu_offload": False,
            "checkpoint_interval_tokens": budget // 5 if evaluation_checkpoints else 500_000_000,
            "validation_interval_tokens": 100_000_000,
        }
        if performance:
            contract["performance"] = performance
            contract["optimizer"] = "Sinkhorn-Engram-FP32-momentum; Adafactor-other-parameters"
            contract["engram_peak_base_learning_rate"] = .01 * performance["table_lr_scale"]
            contract["engram_learning_rate_multiplier"] = 5
            contract["engram_weight_decay"] = 0
            contract["router_bias_update"] = performance["router_bias_rate"]
        if a.scaling_study:
            contract.update(
                scaling_study=True,
                evaluation_checkpoints=5,
                checkpoint_storage="OSS with NAS symlinks",
                active_moe_width_ratio="3/1",
            )
        if matched_contract is not None:
            contract.update(
                matched_mixer_study=True, matched_study=matched_contract,
                execution=execution,
                evaluation_checkpoints=5, checkpoint_storage="OSS with NAS symlinks",
                recovery_checkpoint_interval_updates=100,
                recovery_checkpoint_startup_updates=10,
                recovery_checkpoint_before_validation=True,
                recovery_checkpoint_keep=2,
            )
            if contract["global_windows"] != matched_contract["training"]["global_windows_per_update"]:
                raise ValueError("matched study changed effective batch or data-order geometry")
        if a.training_prefix_tokens is not None:
            contract.update(training_prefix_tokens=budget, learning_rate_schedule_tokens=schedule_budget,
                            historical_curve_comparison=True,
                            comparison_note="Historical controls require their separate provenance audit; no fresh four-arm match is claimed.")
        if cell is not None:
            contract["repeated_data_sweep"] = cell
            contract["token_allowance"] = 10_000_000_000
            contract["matrix_weight_decay"] = cell["weight_decay"]
            contract["weight_decay_rule"] = (
                "decoupled: p -= min(relative_lr, step**-0.5) * wd * p; matrices only"
            )
            contract["compute_axis"] = "training GPU-seconds, excluding evaluation/checkpoint IO"
        contract = canonical_contract(contract)
        atomic_write_json(output / f"rank-{rank:02d}-constructed.json", contract)
        atomic_write_json(
            output / f"rank-{rank:02d}-storage.json",
            {"local_parameter_gib": loading["local_parameter_gib"]},
        )
        signatures = [None] * world
        dist.all_gather_object(
            signatures, hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        )
        if len(set(signatures)) != 1:
            raise ValueError("training contract must be identical on all ranks")
        if rank == 0:
            atomic_write_json(output / "RUN_CONTRACT.json", contract)
        model.train()
        if a.mode == "qualify":
            run_qualification(
                model, indexers, gates, optimizer, contract, output, context=a.context
            )
            return
        if a.tiny:
            raise ValueError("tiny geometry is qualification-only")
        if a.qualification is None:
            raise ValueError("production requires qualification on its full model and mesh")
        q = json.loads((a.qualification / "QUALIFIED.json").read_text())
        if not q["passed"] or q["contract"] != contract:
            raise ValueError("production differs from the qualified contract")
        if matched_contract and (q.get("post_training_validation_passed") is not True
                                 or q.get("post_validation_update_passed") is not True):
            raise ValueError("matched production requires the qualified train/eval/train transition")
        if cell is None:
            data = ScratchData(data_root, target_budget=budget if a.training_prefix_tokens is not None else None)
        else:
            from archlab.automodel.repeated_scratch_data import RepeatedScratchData

            data = RepeatedScratchData(
                data_root, unique_targets=cell["unique_targets"], epochs=cell["epochs"]
            )
        validation = ScratchData(data_root, split="validation")
        cursor = {"step": 0, "phase_step": 0, "supervised_tokens": 0, "window_cursor": 0}
        if a.resume:
            parent_marker = json.loads((a.resume / "COMPLETE.json").read_text())
            parent = parent_marker["contract"]
            if parent.get("performance") != contract.get("performance"):
                raise ValueError("performance geometry/optimizer changes require a fresh experiment")
            from archlab.automodel.deepseek_v41_scratch_resume import maintenance_resume_changes

            controlled_changes = maintenance_resume_changes(parent, contract)
            if controlled_changes is None:
                allowed_parents = {
                    "cc57146c39020c01476f3fa536fa429fef6adf8b": (10, 577519, 640),
                    "96878552c466119996967721ea5da163be129917": (20, 1175800, 1280),
                }
                position = (
                    parent_marker["cursor"]["step"],
                    parent_marker["cursor"]["supervised_tokens"],
                    parent_marker["cursor"]["window_cursor"],
                )
                if allowed_parents.get(parent.get("project_commit")) != position:
                    raise ValueError("controlled correction must use a recorded matched checkpoint")
                if parent.get("variant") != a.variant or parent.get("world_size") != 8:
                    raise ValueError("unrecognized controlled continuation parent")
                for field in (
                    "geometry",
                    "parameters",
                    "packages",
                    "cuda",
                    "nccl",
                    "container_image",
                ):
                    if parent["runtime"][field] != contract["runtime"][field]:
                        raise ValueError(f"parent runtime/model differs: {field}")
                if (
                    parent["data_contract"] != contract["data_contract"]
                    or parent["global_windows"] != contract["global_windows"]
                ):
                    raise ValueError(
                        "controlled continuation changed data order or effective batch"
                    )
                controlled_changes = ["legacy router/batch correction from recorded parent"]
            cursor = restore_full_checkpoint(a.resume, model, optimizer, parent)
            atomic_write_json(
                output / f"rank-{rank:02d}-resume.json",
                {
                    "parent_checkpoint": str(a.resume),
                    "parent_contract_sha256": hashlib.sha256(
                        json.dumps(parent, sort_keys=True).encode()
                    ).hexdigest(),
                    "cursor": cursor,
                    "controlled_changes": controlled_changes,
                },
            )
        if data.targets_before(cursor["window_cursor"]) != cursor["supervised_tokens"]:
            raise ValueError("checkpoint cursor disagrees with sealed data")
        pending_validation = consume_pending_validation(cursor)
        initial = fingerprint(model, common_only=True)
        atomic_write_json(
            output / f"rank-{rank:02d}-initial.json",
            {"common_sha256": initial, "random_initialization": a.resume is None},
        )
        if a.resume is None:
            init_eval = evaluate(model, validation, targets=1_000_000, step=0)
            if rank == 0:
                append_metric(output / "validation.jsonl", init_eval)
        elif pending_validation is not None:
            resumed_eval = evaluate(model, validation, targets=1_000_000, step=cursor["step"])
            resumed_eval["resumed_pending_validation_tokens"] = pending_validation
            if rank == 0:
                append_metric(output / "validation.jsonl", resumed_eval)
        start_step = cursor["step"]
        recent_router = []
        last_save = cursor["supervised_tokens"]
        save_interval = contract["checkpoint_interval_tokens"]
        next_save = (last_save // save_interval + 1) * save_interval
        next_eval = (cursor["supervised_tokens"] // 100_000_000 + 1) * 100_000_000
        while cursor["supervised_tokens"] < budget:
            wall = time.perf_counter()
            batches = build_batches(
                data, cursor["window_cursor"], rank, microbatch=microbatch, world_size=world,
                accumulation=accumulation,
                trim_alignment=execution.get("trim_alignment"),
            )
            rate = learning_rate(cursor["step"], cursor["supervised_tokens"], budget=schedule_budget)
            metric = gradient_step(
                model,
                optimizer,
                indexers,
                batches,
                rate=rate,
                optimized=bool(execution.get("optimized_synchronization")),
                router_auxiliary=True,
                audit=cursor["step"] < 2,
                attention_masks=[b[1] != -100 for b in batches],
            )
            metric.update(balance_routers(gates, contract["router_bias_update"], proportional=True,
                                         batched_metrics=bool(execution.get("optimized_synchronization"))))
            metric.pop("coverage", None)
            if cell is not None:
                depth = len(model.model.layers.layout.execution(cell["recursions"]))
                cursor["training_gpu_seconds"] = (
                    cursor.get("training_gpu_seconds", 0) + world * metric["seconds"]
                )
                cursor["executed_block_tokens"] = (
                    cursor.get("executed_block_tokens", 0) + depth * metric["supervised_tokens"]
                )
                metric.update(
                    recursions=cell["recursions"],
                    executed_depth=depth,
                    stored_layers=cell["stored_layers"],
                    matrix_weight_decay=cell["weight_decay"],
                    training_gpu_seconds=cursor["training_gpu_seconds"],
                    executed_block_tokens=cursor["executed_block_tokens"],
                )
            cursor["step"] += 1
            cursor["phase_step"] += 1
            cursor["window_cursor"] += contract["global_windows"]
            cursor["supervised_tokens"] += metric["supervised_tokens"]
            if data.targets_before(cursor["window_cursor"]) != cursor["supervised_tokens"]:
                raise ValueError("training lost or repeated data targets")
            metric.update(
                step=cursor["step"],
                phase_step=cursor["phase_step"],
                consumed_supervised_tokens=cursor["supervised_tokens"],
                window_cursor=cursor["window_cursor"],
                wall_seconds=time.perf_counter() - wall,
            )
            if rank == 0:
                append_metric(output / "train-metrics.jsonl", metric)
            emit("scratch_train_update", **metric)
            stop = torch.tensor(
                int((output / "STOP_REQUEST").exists() or (a.steps and cursor["step"] >= a.steps)),
                device="cuda",
            )
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            validation_due = cursor["supervised_tokens"] >= next_eval
            recovery_due = bool(matched_contract) and recovery_checkpoint_due(
                cursor["step"], start_step=start_step, validation_due=validation_due,
                interval=contract["recovery_checkpoint_interval_updates"],
                startup_updates=contract["recovery_checkpoint_startup_updates"],
            )
            if (
                (not evaluation_checkpoints and cursor["step"] == start_step + 10)
                or (
                    cursor["supervised_tokens"] >= next_save
                    if evaluation_checkpoints
                    else cursor["supervised_tokens"] - last_save >= save_interval
                )
                or bool(stop)
                or cursor["supervised_tokens"] >= budget
                or recovery_due
            ):
                name = f"step-{cursor['step']:06d}"
                checkpoint = output / "checkpoints" / name
                milestone = evaluation_checkpoints and cursor["supervised_tokens"] >= next_save
                if milestone:
                    from archlab.storage.oss_checkpoint import prepare_distributed_checkpoint

                    checkpoint = prepare_distributed_checkpoint(
                        checkpoint,
                        Path(os.environ["ARCHLAB_SCALING_CHECKPOINT_ROOT"])
                        / (contract["variant"] if contract.get("matched_mixer_study") else output.name)
                        / name,
                    )
                elif evaluation_checkpoints:
                    checkpoint = output / "recovery" / name
                saved_cursor = (checkpoint_cursor(cursor, next_validation_tokens=next_eval)
                                if matched_contract else cursor)
                save_full_checkpoint(
                    checkpoint,
                    model,
                    optimizer,
                    saved_cursor,
                    contract,
                )
                if matched_contract and not milestone:
                    from archlab.storage.oss_checkpoint import on_rank_zero

                    on_rank_zero(lambda checkpoint=checkpoint, saved_cursor=saved_cursor: record_recovery_checkpoint(
                        output, checkpoint, saved_cursor, contract, keep=contract["recovery_checkpoint_keep"]
                    ))
                last_save = cursor["supervised_tokens"]
                if milestone:
                    from archlab.storage.oss_checkpoint import record_evaluation_checkpoint

                    record_evaluation_checkpoint(
                        output, checkpoint, milestone=next_save, budget=budget
                    )
                    next_save += save_interval
                emit("scratch_checkpoint_complete", **cursor)
            if validation_due:
                val = evaluate(model, validation, targets=1_000_000, step=cursor["step"])
                if cell is not None:
                    val.update(
                        consumed_supervised_tokens=cursor["supervised_tokens"],
                        training_gpu_seconds=cursor["training_gpu_seconds"],
                        epoch=cursor["supervised_tokens"] / cell["unique_targets"],
                    )
                if rank == 0:
                    append_metric(output / "validation.jsonl", val)
                next_eval += 100_000_000
            recent_router.append(metric)
            recent_router = recent_router[-20:] if a.scaling_study else recent_router[-10:]
            router_ok = (
                len(recent_router) == (20 if a.scaling_study else 10)
                and sum(x["router_load_cv_mean"] for x in recent_router) / len(recent_router) < 2.5
                and metric["router_usage_window_updates"] == 20
                and metric["router_unused_fraction_window"] < 0.01
                and metric["router_worst_unused_fraction_window"] < 0.05
            )
            stable_loss = not a.scaling_study or (
                len(recent_router) == 20
                and cursor["step"] >= 200
                and max(x["loss"] for x in recent_router)
                < 1.2 * math.log(contract["runtime"]["geometry"]["text_config"]["vocab_size"])
                and sum(x["loss"] for x in recent_router[-10:])
                <= 1.1 * sum(x["loss"] for x in recent_router[:10])
            )
            if cursor["step"] >= start_step + 10 and router_ok and stable_loss and rank == 0:
                atomic_write_json(
                    output / "TRAINING_HEALTHY.json",
                    {
                        "passed": True,
                        **cursor,
                        "latest": metric,
                        "router_checks_passed": True,
                        "loss_stability_checks_passed": stable_loss,
                        "post_warmup_updates": max(0, cursor["step"] - contract["warmup_updates"]),
                    },
                )
            if bool(stop):
                break
        if cursor["supervised_tokens"] >= budget and rank == 0:
            atomic_write_json(output / "COMPLETE.json", {"passed": True, **cursor})
    except BaseException:
        atomic_write_json(
            output / f"rank-{rank:02d}-failure.json", {"traceback": traceback.format_exc()}
        )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
