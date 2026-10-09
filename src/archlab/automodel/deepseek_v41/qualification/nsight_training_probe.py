"""Bounded training timings and optional Nsight capture; never saves training state."""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--variant", choices=("normal", "simplicial"), default="normal")
    parser.add_argument(
        "--stage",
        choices=(
            "original",
            "grouped",
            "scaled",
            "sinkhorn",
            "trimmed",
            "synchronized",
            "head",
            "resident",
            "hc",
            "deepep",
            "selected",
        ),
        default="original",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--performance-contract", type=Path)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--repeat-batches", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--table-lr-scale", type=float, default=1.0)
    parser.add_argument("--router-rate", type=float, default=0.01)
    parser.add_argument("--schedule", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or args.microbatch < 1:
        parser.error("steps/microbatch must be positive and warmup nonnegative")
    if args.checkpoint and args.stage != "original":
        parser.error("geometry and optimizer ablations require fresh initialization")
    # Load training code from the production snapshot, not the profiling checkout.
    sys.path.insert(0, str(args.source / "src"))
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    packages = select_container_kernel_packages(
        Path(os.environ["ARCHLAB_CONTAINER_KERNEL_PACKAGES"])
    )
    import torch
    import torch.distributed as dist

    from archlab.automodel import deepseek_v41_official_moe as moe
    from archlab.automodel.deepseek_v41_full_checkpoint import restore_full_checkpoint
    from archlab.automodel.deepseek_v41_full_training import gradient_step
    from archlab.automodel.deepseek_v41_performance import read_performance_contract
    from archlab.automodel.deepseek_v41_scratch_construct import construct_scratch
    from archlab.automodel.deepseek_v41_scratch_data import ScratchData
    from archlab.automodel.deepseek_v41_scratch_training import build_batches, learning_rate
    from archlab.optimizers.router_balance import balance_routers
    from archlab.optimizers.sharded_adafactor import ShardedAdafactor

    if args.stage == "selected":
        performance = read_performance_contract(args.performance_contract)
        if performance is None:
            parser.error("selected stage requires --performance-contract")
        if args.microbatch != performance["microbatch"]:
            parser.error("microbatch must match the selected performance contract")
        args.table_lr_scale = performance["table_lr_scale"]
        args.router_rate = performance["router_bias_rate"]
    else:
        if args.performance_contract is not None:
            parser.error("--performance-contract requires the selected stage")
        level = (
            "original",
            "grouped",
            "scaled",
            "sinkhorn",
            "trimmed",
            "synchronized",
            "head",
            "resident",
            "hc",
            "deepep",
        ).index(args.stage)
        performance = dict(
            grouped_experts=level >= 1,
            scale_engram=level >= 2,
            engram_optimizer="sinkhorn-algorithm-1" if level >= 3 else "adafactor",
            trim_alignment=128 if level >= 4 else None,
            optimized_synchronization=level >= 5,
            batched_updates=level >= 5,
            head_loss_chunk=1024 if level >= 6 else 128,
            retain_activations=level >= 7,
            compile_hc_backward=level >= 8,
            expert_dispatcher="deepep" if level >= 9 else "torch",
        )
    optimized = performance["optimized_synchronization"]
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=15),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    rank = dist.get_rank()
    contract = (
        None
        if args.checkpoint is None
        else json.loads((args.checkpoint / "COMPLETE.json").read_text())["contract"]
    )
    options = {
        key: performance[key]
        for key in ("grouped_experts", "scale_engram", "retain_activations", "expert_dispatcher")
    }
    options["simplicial_backend"] = performance.get("simplicial_backend", "deterministic")
    model, indexers, gates, runtime = construct_scratch(
        base_config=os.environ["ARCHLAB_DEEPSEEK_V41_BASE_CONFIG"],
        assets=os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"],
        variant=args.variant if contract is None else contract["variant"],
        width=args.width,
        sparse_backend="batched",
        scaling_study=True,
        **options,
    )
    if performance["compile_hc_backward"]:
        from archlab.automodel.deepseek_v41_performance import compile_hc_backward

        compile_hc_backward(model)
    model.lm_head._archlab_loss_chunk = performance["head_loss_chunk"]
    if performance["engram_optimizer"] == "sinkhorn-algorithm-1":
        from archlab.optimizers.sinkhorn import EngramSinkhornAdafactor

        optimizer = EngramSinkhornAdafactor(
            model.named_parameters(),
            lr=0.01,
            table_lr_scale=args.table_lr_scale,
            batched_updates=performance["batched_updates"],
        )
    else:
        optimizer = ShardedAdafactor(
            model.parameters(), lr=0.01, batched_updates=performance["batched_updates"]
        )
    if performance["trim_alignment"] is not None:
        for indexer in indexers:
            indexer._archlab_sample_context = 2048
    data = ScratchData(os.environ["ARCHLAB_DEEPSEEK_V41_SCRATCH_DATA"])
    first_window = 0 if args.stage == "selected" else 207 * 64
    cursor = (
        restore_full_checkpoint(args.checkpoint, model, optimizer, contract)
        if args.checkpoint
        else {
            "step": 0 if args.stage == "selected" else 207,
            "window_cursor": first_window,
            "supervised_tokens": data.targets_before(first_window),
        }
    )

    def ranged(name, fn):
        if not args.profile:
            return fn

        @functools.wraps(fn)
        def call(*a, **kw):
            with torch.cuda.nvtx.range(name):
                return fn(*a, **kw)

        return call

    # Diagnostic process only: wrappers add ranges without changing tensor math.
    moe._native_up_grouped_down = ranged("moe/local_experts", moe._native_up_grouped_down)
    parameter_names = {id(p): name for name, p in model.named_parameters()}
    original_geometry, original_step = optimizer._geometry, optimizer.step
    parameter_range_open = False

    def geometry(p):
        nonlocal parameter_range_open
        if parameter_range_open:
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("optimizer/parameter/" + parameter_names[id(p)])
        parameter_range_open = True
        return original_geometry(p)

    def optimizer_step(*a, **kw):
        nonlocal parameter_range_open
        with torch.cuda.nvtx.range("optimizer/adafactor"):
            try:
                return original_step(*a, **kw)
            finally:
                if parameter_range_open:
                    torch.cuda.nvtx.range_pop()
                    parameter_range_open = False

    if args.profile:
        optimizer._geometry, optimizer.step = geometry, optimizer_step
    model.forward = ranged("training/forward", model.forward)
    torch.autograd.backward = ranged("training/backward", torch.autograd.backward)
    for name, module in model.named_modules():
        kind = type(module).__name__
        if any(
            word in kind.lower() for word in ("attention", "moe", "engram", "indexer", "adapter")
        ):
            module.forward = ranged(f"module/{kind}/{name}", module.forward)
    model.lm_head.loss = ranged("loss/lm_head", model.lm_head.loss)
    from nemo_automodel.components.models.deepseek_v4.kernels import sparse_attention

    for module, function, label in (
        (sparse_attention.sparse_mla_fwd, "sparse_mqa_fwd_interface", "sparse/forward"),
        (sparse_attention.sparse_mla_bwd, "sparse_mqa_bwd_interface", "sparse/backward"),
    ):
        setattr(module, function, ranged(label, getattr(module, function)))
    rows = []
    first_window = cursor["window_cursor"]
    try:
        for iteration in range(args.warmup + args.steps):
            if iteration == args.warmup:
                dist.barrier()
                torch.cuda.synchronize()
                if rank == 0 and args.profile:
                    torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push("profile")
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            with torch.cuda.nvtx.range(f"step/{iteration}"):
                with torch.cuda.nvtx.range("data/batch"):
                    if args.repeat_batches:
                        cursor["window_cursor"] = (
                            first_window + (iteration % args.repeat_batches) * 16 * args.microbatch
                        )
                    batches = build_batches(
                        data,
                        cursor["window_cursor"],
                        rank,
                        microbatch=args.microbatch,
                        world_size=16,
                        trim_alignment=performance["trim_alignment"],
                    )
                with torch.cuda.nvtx.range("training/gradient_step"):
                    metric = gradient_step(
                        model,
                        optimizer,
                        indexers,
                        batches,
                        rate=(
                            learning_rate(cursor["step"], cursor["supervised_tokens"])
                            if args.checkpoint
                            else (
                                learning_rate(iteration, cursor["supervised_tokens"])
                                if args.schedule
                                else args.learning_rate
                            )
                        ),
                        router_auxiliary=True,
                        audit=False,
                        timing=True,
                        **({"optimized": True} if optimized else {}),
                        attention_masks=[b[1] != -100 for b in batches],
                    )
                with torch.cuda.nvtx.range("router/balance"):
                    metric.update(
                        balance_routers(
                            gates,
                            args.router_rate,
                            proportional=True,
                            **({"batched_metrics": True} if optimized else {}),
                        )
                    )
            torch.cuda.synchronize()
            elapsed = torch.tensor(
                time.perf_counter() - wall_start, device="cuda", dtype=torch.float64
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            metric["wall_seconds"] = float(elapsed)
            metric["valid_tokens_per_second"] = metric["supervised_tokens"] / float(elapsed)
            cursor["step"] += 1
            cursor["window_cursor"] += 16 * args.microbatch
            cursor["supervised_tokens"] += metric["supervised_tokens"]
            rows.append({"iteration": iteration, "captured": iteration >= args.warmup, **metric})
            if rank == 0:
                print(json.dumps(rows[-1]), flush=True)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        if rank == 0 and args.profile:
            torch.cuda.cudart().cudaProfilerStop()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank-{rank:02d}.json").write_text(
            json.dumps(
                {
                    "arguments": {
                        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                    },
                    "performance": performance,
                    "microbatch": args.microbatch,
                    "stage": args.stage,
                    "variant": args.variant,
                    "width": args.width,
                    "source_sha256": {
                        str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (args.source / "src/archlab").rglob("*.py")
                    },
                    "checkpoint": str(args.checkpoint),
                    "source": str(args.source),
                    "training_state_saved": False,
                    "runtime": runtime,
                    "kernel_packages": packages,
                    "steps": rows,
                },
                indent=2,
            )
            + "\n"
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
