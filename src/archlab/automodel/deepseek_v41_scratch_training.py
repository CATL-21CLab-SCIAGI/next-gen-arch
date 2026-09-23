"""Qualified eight-GPU scratch comparison with exact full-state checkpoints."""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import hashlib
import json
import math
import os
import signal
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


def build_batches(data, window_cursor, rank, *, microbatch=8, accumulation=1):
    return [
        data.batch(
            [window_cursor + (micro * 8 + rank) * microbatch + i for i in range(microbatch)],
            device="cuda",
        )
        for micro in range(accumulation)
    ]


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
                ids, labels, count = data.batch([index * 8 + dist.get_rank()], device="cuda")
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
            return {
                "step": step,
                "loss": float(total[0] / total[1]),
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

    sparse = contract["runtime"].get("sparse_precision", {})
    allow_atomic = (
        sparse.get("kv_gradient_reduction") == "atomic-fp32"
        and sparse.get("bitwise_next_update") is False
    )
    sparse_oracle = None
    if allow_atomic:
        from archlab.automodel.deepseek_v41_sparse_qualification import qualify_batched_sparse

        sparse_oracle = qualify_batched_sparse()
        atomic_write_json(output / f"rank-{dist.get_rank():02d}-sparse-oracle.json", sparse_oracle)
    rank = dist.get_rank()
    atomic_write_json(
        output / f"rank-{rank:02d}-optimizer-oracle.json", qualify_distributed_optimizer()
    )
    batches = []
    for micro in range(2):
        ids, labels = _batch(rank + micro * 8, 128)
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
            audit=step == 0,
            attention_masks=masks,
        )
        metric.update(balance_routers(gates, 0.01, proportional=True))
        metric.pop("coverage", None)
        metrics.append(metric)
        atomic_write_json(output / f"rank-{rank:02d}-qualification-update-{step + 1}.json", metric)
    cursor = {
        "step": 2,
        "phase_step": 2,
        "supervised_tokens": sum(m["supervised_tokens"] for m in metrics),
        "window_cursor": 32,
    }
    before = state_fingerprint(model, optimizer)
    checkpoint = output / "checkpoint"
    save_full_checkpoint(checkpoint, model, optimizer, cursor, contract)
    first = gradient_step(
        model,
        optimizer,
        indexers,
        batches,
        rate=0.001,
        router_auxiliary=True,
        attention_masks=masks,
    )
    first.update(balance_routers(gates, 0.01, proportional=True))
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
        attention_masks=masks,
    )
    repeated.update(balance_routers(gates, 0.01, proportional=True))
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
    ids = ids.repeat(4, 1)
    labels = labels.repeat(4, 1)
    peak = gradient_step(
        model,
        optimizer,
        indexers,
        [(ids, labels, int((labels != -100).sum()))],
        rate=0.001,
        router_auxiliary=True,
        attention_masks=[labels != -100],
    )
    peak.update(balance_routers(gates, 0.01, proportional=True))
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
        attention_masks=[empty_labels != -100],
    )
    empty.update(balance_routers(gates, 0.01, proportional=True))
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
        "production_microbatch": 8,
        "production_accumulation": 1,
        "metrics": metrics,
        "production_shape_metric": peak,
        "empty_rank_metric": empty,
    }
    atomic_write_json(output / f"rank-{rank:02d}-qualified.json", receipt)
    dist.barrier()
    if rank == 0:
        atomic_write_json(
            output / "QUALIFIED.json",
            {
                "passed": True,
                "contract": contract,
                "world_size": 8,
                "context": context,
                "ranks": [f"rank-{i:02d}-qualified.json" for i in range(8)],
            },
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant", choices=["normal", "simplicial", "linear", "linsimp"], required=True
    )
    p.add_argument("--mode", choices=["qualify", "train"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--steps", type=int, default=0)
    a = p.parse_args()
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
        ]
        paths += [
            source / "src/archlab/architectures/linsimp_attention.py",
            source / "src/archlab/architectures/deepseek_v41_linsimp_adapter.py",
            source / "src/archlab/automodel/deepseek_v41_official_adapter.py",
            source / "src/archlab/automodel/deepseek_v41_sparse_qualification.py",
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
        )
        loading["resolved_kernel_packages"] = resolved_kernel_packages
        optimizer = ShardedAdafactor(model.parameters(), lr=0.01)
        data_root = Path(os.environ["ARCHLAB_DEEPSEEK_V41_SCRATCH_DATA"])
        data_contract = json.loads((data_root / "CONTRACT.json").read_text())
        contract = {
            "format": "archlab-v41-scratch-comparison-v1",
            "project_commit": commit,
            "implementation_sha256": code,
            "variant": a.variant,
            "tiny": a.tiny,
            "world_size": 8,
            "runtime": training_runtime(loading),
            "data_contract": data_contract,
            "context": a.context,
            "microbatch": 8,
            "accumulation": 1,
            "global_windows": 64,
            "optimizer": "FP32-factored-Adafactor-stochastic-BF16",
            "peak_relative_learning_rate": 0.01,
            "minimum_relative_learning_rate": 0.001,
            "warmup_updates": 100,
            "target_supervised_tokens": 10_000_000_000,
            "gradient_clip": 1.0,
            "router_auxiliary_loss_coefficient": 0.01,
            "router_bias_update": 0.01,
            "router_bias_rule": "centered proportional load error clipped [-5,1]",
            "indexer_kl_coefficient": 0.01,
            "cpu_offload": False,
            "checkpoint_interval_tokens": 500_000_000,
            "validation_interval_tokens": 100_000_000,
        }
        contract = canonical_contract(contract)
        atomic_write_json(output / f"rank-{rank:02d}-constructed.json", contract)
        atomic_write_json(
            output / f"rank-{rank:02d}-storage.json",
            {"local_parameter_gib": loading["local_parameter_gib"]},
        )
        signatures = [None] * 8
        dist.all_gather_object(
            signatures, hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        )
        if len(set(signatures)) != 1:
            raise ValueError("training contract must be identical on all eight ranks")
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
            raise ValueError("production requires full-size8-GPU qualification")
        q = json.loads((a.qualification / "QUALIFIED.json").read_text())
        if not q["passed"] or q["contract"] != contract:
            raise ValueError("production differs from the qualified contract")
        data = ScratchData(data_root)
        validation = ScratchData(data_root, split="validation")
        cursor = {"step": 0, "phase_step": 0, "supervised_tokens": 0, "window_cursor": 0}
        if a.resume:
            parent_marker = json.loads((a.resume / "COMPLETE.json").read_text())
            parent = parent_marker["contract"]
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
        initial = fingerprint(model, common_only=True)
        atomic_write_json(
            output / f"rank-{rank:02d}-initial.json",
            {"common_sha256": initial, "random_initialization": a.resume is None},
        )
        if a.resume is None:
            init_eval = evaluate(model, validation, targets=1_000_000, step=0)
            if rank == 0:
                append_metric(output / "validation.jsonl", init_eval)
        start_step = cursor["step"]
        recent_router = []
        last_save = cursor["supervised_tokens"]
        next_eval = (cursor["supervised_tokens"] // 100_000_000 + 1) * 100_000_000
        while cursor["supervised_tokens"] < 10_000_000_000:
            wall = time.perf_counter()
            batches = build_batches(data, cursor["window_cursor"], rank)
            rate = learning_rate(cursor["step"], cursor["supervised_tokens"])
            metric = gradient_step(
                model,
                optimizer,
                indexers,
                batches,
                rate=rate,
                router_auxiliary=True,
                audit=cursor["step"] < 2,
                attention_masks=[b[1] != -100 for b in batches],
            )
            metric.update(balance_routers(gates, 0.01, proportional=True))
            metric.pop("coverage", None)
            cursor["step"] += 1
            cursor["phase_step"] += 1
            cursor["window_cursor"] += 64
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
            if (
                cursor["step"] == start_step + 10
                or cursor["supervised_tokens"] - last_save >= 500_000_000
                or bool(stop)
                or cursor["supervised_tokens"] >= 10_000_000_000
            ):
                save_full_checkpoint(
                    output / "checkpoints" / f"step-{cursor['step']:06d}",
                    model,
                    optimizer,
                    cursor,
                    contract,
                )
                last_save = cursor["supervised_tokens"]
                emit("scratch_checkpoint_complete", **cursor)
            if cursor["supervised_tokens"] >= next_eval:
                val = evaluate(model, validation, targets=1_000_000, step=cursor["step"])
                if rank == 0:
                    append_metric(output / "validation.jsonl", val)
                next_eval += 100_000_000
            recent_router.append(metric)
            recent_router = recent_router[-10:]
            router_ok = (
                len(recent_router) == 10
                and sum(x["router_load_cv_mean"] for x in recent_router) / 10 < 2.5
                and metric["router_usage_window_updates"] == 20
                and metric["router_unused_fraction_window"] < 0.01
                and metric["router_worst_unused_fraction_window"] < 0.05
            )
            if cursor["step"] >= start_step + 10 and router_ok and rank == 0:
                atomic_write_json(
                    output / "TRAINING_HEALTHY.json",
                    {"passed": True, **cursor, "latest": metric, "router_checks_passed": True},
                )
            if bool(stop):
                break
        if cursor["supervised_tokens"] >= 10_000_000_000 and rank == 0:
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
