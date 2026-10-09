"""Distributed Limite warmup on one sealed Math-v2 order."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from archlab.automodel.limite_adapter_common import (
    attention_kernel_name,
    build_model,
    frozen_fingerprint,
    loss_sum,
    runtime_contract,
    save_adapter,
)
from archlab.automodel.limite_adapter_communication import (
    check_warmup_resume,
    communication_contract,
    fp32_mean_hook,
    synchronize_gradients,
)
from archlab.automodel.limite_adapter_compilation import configure_compilation
from archlab.automodel.limite_adapter_data import MathWindows, WindowPrefetcher
from archlab.automodel.limite_adapter_graph import ManualTrainingGraph, graph_contract
from archlab.automodel.limite_performance import baseline_flops
from archlab.optimizers.limite_warmup import build_warmup_optimizer, warmup_learning_rates


class Objective(nn.Module):
    def __init__(self, model, chunk, compile_mode="none", *, checkpoint_head=True):
        super().__init__()
        self.policy = model
        self.chunk = chunk
        self.checkpoint_head = checkpoint_head
        self.head_part, self.compilation = configure_compilation(model, compile_mode)

    def forward(self, ids, targets):
        return loss_sum(
            self.policy,
            ids,
            targets,
            self.chunk,
            head_part=self.head_part,
            checkpoint_head=self.checkpoint_head,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--attention-backend", choices=("native", "tilelang"), default=None)
    p.add_argument("--normal-kernel", choices=("shared", "gqa"), default=None)
    p.add_argument("--normal-backward", choices=("tilelang", "fa4"), default=None)
    p.add_argument("--trainable-mode", choices=("adapter", "full"), default=None)
    p.add_argument("--allow-backend-migration", action="store_true")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--credentials", type=Path)
    p.add_argument("--oss")
    p.add_argument("--tokens", type=int, default=10_000_000_000)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=2)
    p.add_argument("--head-chunk", type=int, default=512)
    p.add_argument("--no-head-checkpoint", action="store_true")
    p.add_argument("--cuda-graph", choices=("none", "manual"), default="none")
    p.add_argument("--static-gradient-sync", action="store_true")
    p.add_argument(
        "--compile-mode", choices=("none", "strict-regional", "strict-blocks"), default="none"
    )
    p.add_argument("--prefetch-windows", action="store_true")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--resume", type=Path)
    p.add_argument("--ddp-bucket-cap-mb", type=float, default=None)
    p.add_argument("--metric-segment", default=None)
    p.add_argument(
        "--communication-schedule", choices=("bucketed", "bucketed-fp32", "deferred"), default=None
    )
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--backbone-warmup-steps", type=int, default=100)
    a = p.parse_args()
    if a.static_gradient_sync and a.cuda_graph != "manual":
        raise ValueError("static gradient synchronization requires qualified manual graphs")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    prior = json.loads((a.resume / "COMPLETE.json").read_text()) if a.resume else {}
    scope_hint = a.trainable_mode or prior.get("trainable_mode", "adapter")
    torch.cuda.set_device(local)
    execution_stream = None
    if a.cuda_graph == "manual":
        # AccumulateGrad must belong to the same nondefault stream used by
        # capture, validation, the optimizer and ordinary training backward.
        execution_stream = torch.cuda.Stream()
        torch.cuda.set_stream(execution_stream)
    torch.set_num_threads(4)
    if world > 1:
        # Full optimizer payloads exceed 13 GiB. Permit verified OSS checkpoint
        # publication without treating its IO barrier as a ten-minute hang.
        dist.init_process_group(
            "nccl",
            device_id=torch.device("cuda", local),
            **({"timeout": timedelta(minutes=30)} if scope_hint == "full" else {}),
        )
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    a.output.mkdir(parents=True, exist_ok=True)
    data = MathWindows(a.data)
    valid = MathWindows(a.data, "validation")
    seq = data.spec["context"]
    batch = world * a.microbatch * a.accumulation
    model = build_model(
        a.model,
        a.variant,
        f"cuda:{local}",
        a.resume,
        attention_backend=a.attention_backend,
        trainable_mode=a.trainable_mode,
        allow_backend_migration=a.allow_backend_migration,
        normal_kernel=a.normal_kernel,
        normal_backward=a.normal_backward,
    )
    attention_backend = model.model.adapter_config["attention_backend"]
    normal_kernel = getattr(model.model, "normal_kernel", "shared")
    normal_backward = getattr(model.model, "normal_backward", "tilelang")
    trainable_mode = model.model.trainable_mode
    schedule = a.communication_schedule or ("deferred" if trainable_mode == "full" else "bucketed")
    train = [p for p in model.parameters() if p.requires_grad]
    communication = communication_contract(
        train,
        a.ddp_bucket_cap_mb,
        schedule=schedule,
        static_gradient_sync=a.static_gradient_sync,
    )
    communication["process_group_timeout_seconds"] = (
        1800 if trainable_mode == "full" else "container_default"
    )
    base_hash = model.archlab_base_snapshot_sha256
    if not math.isfinite(a.backbone_lr) or a.backbone_lr <= 0 or a.backbone_warmup_steps < 1:
        raise ValueError("invalid full-weight backbone learning-rate schedule")
    optimizer = build_warmup_optimizer(model, backbone_lr=a.backbone_lr)
    step = tokens = 0
    step = prior.get("step", 0)
    tokens = prior.get("tokens", 0)
    full_weight_start_step = prior.get("full_weight_start_step", step)
    full_weight_start_tokens = prior.get("full_weight_start_tokens", tokens)
    source_2b_checkpoint = prior.get("source_2b_checkpoint", str(a.resume) if a.resume else None)
    source_2b_attention_backend = prior.get(
        "source_2b_attention_backend", prior.get("adapter", {}).get("attention_backend", "native")
    )
    warmup_schedule = (
        dict(
            target_tokens=a.tokens,
            adapter_peak_lr=1e-4,
            adapter_warmup_steps=100,
            backbone_peak_lr=a.backbone_lr,
            backbone_warmup_steps=a.backbone_warmup_steps,
            full_weight_start_step=full_weight_start_step,
            full_weight_start_tokens=full_weight_start_tokens,
            source_2b_checkpoint=source_2b_checkpoint,
        )
        if trainable_mode == "full"
        else None
    )
    if trainable_mode == "full" and a.tokens <= full_weight_start_tokens:
        raise ValueError("full-weight target must exceed its original phase start")
    if a.resume:
        check_warmup_resume(
            prior,
            global_batch=batch,
            context=seq,
            data_contract=data.contract,
            warmup_schedule=warmup_schedule,
        )
        optimizer.load_state_dict(
            torch.load(a.resume / "optimizer.pt", map_location="cpu", weights_only=True)
        )
        rng = torch.load(a.resume / "rng.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"], device=local)
    objective = Objective(
        model, a.head_chunk, a.compile_mode, checkpoint_head=not a.no_head_checkpoint
    )
    if a.cuda_graph == "manual" and (a.accumulation != 1 or schedule != "deferred"):
        raise ValueError("manual graphs require one accumulation and deferred FP32 communication")
    training_graph = (
        ManualTrainingGraph(
            objective,
            optimizer,
            world_size=world,
            global_batch=batch,
            context=seq,
            static_gradient_sync=a.static_gradient_sync,
        )
        if a.cuda_graph == "manual"
        else None
    )
    ddp = (
        DDP(
            objective,
            device_ids=[local],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            **({"bucket_cap_mb": a.ddp_bucket_cap_mb} if a.ddp_bucket_cap_mb is not None else {}),
        )
        if world > 1 and schedule in ("bucketed", "bucketed-fp32")
        else objective
    )
    if world > 1 and schedule == "bucketed-fp32":
        ddp.register_comm_hook(None, fp32_mean_hook)
    useful_flops = (
        baseline_flops(model.config, seq, trainable_mode) if a.variant == "normal" else None
    )
    contract = dict(
        variant=a.variant,
        attention_backend=attention_backend,
        normal_kernel=normal_kernel,
        normal_backward=normal_backward,
        normal_backward_runtime=model.archlab_normal_attention_backward_contract,
        trainable_mode=trainable_mode,
        model=a.model,
        adapter=model.model.adapter_config,
        world_size=world,
        microbatch=a.microbatch,
        accumulation=a.accumulation,
        target_tokens=a.tokens,
        warmup_schedule=warmup_schedule,
        context=seq,
        trainable_parameters=sum(p.numel() for p in train),
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        base_snapshot_sha256=base_hash,
        data_contract=data.contract,
        objective=data.spec["objective"],
        head_chunk=a.head_chunk,
        checkpoint_head=objective.checkpoint_head,
        prefetch_windows=a.prefetch_windows,
        compilation=objective.compilation,
        execution_graph=graph_contract(a.cuda_graph, static_gradient_sync=a.static_gradient_sync),
        runtime=runtime_contract(),
        model_flops=(
            dict(
                definition="useful forward/backward matmuls; valid causal/window attention",
                linear_per_token=useful_flops.linear_per_token,
                attention_per_token=useful_flops.attention_per_token,
                dense_bf16_peak_per_gpu=2.25e15,
                excludes="recomputation, optimizer, lookup tables, communication, padded tiles",
            )
            if useful_flops is not None
            else None
        ),
        optimizer=dict(
            name="torch.optim.AdamW",
            lr=1e-4,
            betas=[0.9, 0.95],
            weight_decay=0,
            fused=True,
            gradient_clip=1.0,
        ),
        attention_kernel=attention_kernel_name(
            a.variant,
            attention_backend,
            normal_kernel=normal_kernel,
            normal_backward=normal_backward,
        ),
        communication=communication,
        tracking={"metric_segment": a.metric_segment},
        initialization={
            "checkpoint": str(a.resume) if a.resume else None,
            "source_trainable_mode": prior.get("trainable_mode", "adapter") if a.resume else None,
            "source_attention_backend": prior["adapter"].get("attention_backend", "native")
            if a.resume
            else None,
            "target_attention_backend": attention_backend,
            "backend_migration_explicit": a.allow_backend_migration,
            "preserved_adapter_optimizer_and_rng": bool(a.resume),
            "new_backbone_optimizer": trainable_mode == "full"
            and (not a.resume or prior.get("trainable_mode", "adapter") == "adapter"),
        },
    )
    if trainable_mode == "adapter":
        contract["frozen_sha256"] = base_hash
    else:
        contract["optimizer"].update(
            name="Torch adapter AdamW + container Transformer Engine backbone FusedAdam",
            backbone_master_dtype="float32",
            backbone_moment_dtype="float32",
            native_parameter_dtypes_preserved=True,
            backbone_peak_lr=a.backbone_lr,
            backbone_warmup_steps=a.backbone_warmup_steps,
            backbone_cosine_origin_tokens=full_weight_start_tokens,
            adapter_lr_schedule="original global-position 1e-4 cosine",
        )
    mlflow = None
    if rank == 0:
        (a.output / "CONTRACT.json").write_text(json.dumps(contract, indent=2))
        if a.credentials:
            import mlflow

            from archlab.tracking.mlflow_sync import configure_client

            configure_client(a.credentials)
            creds = json.loads(a.credentials.read_text())
            mlflow.set_tracking_uri(creds["tracking_uri"])
            experiment = (
                "Limite — Full fine tuning"
                if trainable_mode == "full"
                else "Limite — Adapter warmup"
            )
            mlflow.set_experiment(experiment)
            tracking_record = a.output / "MLFLOW.json"
            prior_run = (
                json.loads(tracking_record.read_text())
                if a.resume and tracking_record.exists()
                else None
            )
            mlflow.start_run(
                **(
                    {"run_id": prior_run["run_id"]}
                    if prior_run
                    else {
                        "run_name": f"limite-base-{a.variant}-nemotron10B"
                        + ("-full-from-native2B" if trainable_mode == "full" else "")
                        + ("-tilelang" if attention_backend == "tilelang" else "")
                    }
                )
            )
            mlflow.log_params(
                {
                    k: v
                    for k, v in contract.items()
                    if not isinstance(v, dict) and k not in ("attention_kernel", "normal_backward")
                }
            )
            mlflow.set_tags(
                {
                    "active_attention_kernel": contract["attention_kernel"],
                    "active_attention_backend": attention_backend,
                    "active_normal_backward": normal_backward,
                    "active_trainable_mode": trainable_mode,
                    "active_communication_schedule": schedule,
                    "active_compile_mode": a.compile_mode,
                    "active_cuda_graph": a.cuda_graph,
                    "active_static_gradient_sync": a.static_gradient_sync,
                    "active_ddp_bucket_cap_mib": communication["bucket_cap_mib"],
                    "active_ddp_single_bucket_per_dtype": communication[
                        "fits_one_bucket_per_dtype"
                    ],
                    "active_metric_segment": a.metric_segment or "original",
                    "active_metric_prefix": f"{a.metric_segment}_" if a.metric_segment else "",
                    "origin_checkpoint": source_2b_checkpoint or "from-scratch",
                    "origin_step": full_weight_start_step if trainable_mode == "full" else 0,
                    "origin_tokens": full_weight_start_tokens if trainable_mode == "full" else 0,
                    "origin_attention_backend": source_2b_attention_backend
                    if trainable_mode == "full"
                    else attention_backend,
                }
            )
            mlflow.log_dict(contract, "contract.json")
            if a.resume:
                mlflow.log_dict(contract, f"resumes/step-{step:07d}/contract.json")
                mlflow.set_tags(
                    {
                        "resumed_from_step": step,
                        "active_source_revision": contract["runtime"]["source_revision"],
                        "active_attention_kernel": contract["attention_kernel"],
                    }
                )
            (a.output / "MLFLOW.json").write_text(
                json.dumps(
                    dict(
                        run_id=mlflow.active_run().info.run_id,
                        experiment_id=mlflow.active_run().info.experiment_id,
                        experiment=experiment,
                    )
                )
            )

    def barrier():
        if world > 1:
            dist.barrier()

    def evaluate():
        model.eval()
        tot = torch.zeros(2, device=local)
        with torch.no_grad():
            for j in range(2):
                row = torch.from_numpy(valid[rank + j * world]).to(local)[None]
                tot[0] += loss_sum(model, row[:, :-1], row[:, 1:], a.head_chunk)
                tot[1] += seq
        if world > 1:
            dist.all_reduce(tot)
        model.train()
        return float(tot[0] / tot[1])

    next_checkpoint = (tokens // 2_000_000_000 + 1) * 2_000_000_000
    checkpoint_extra = dict(
        base_snapshot_sha256=base_hash,
        data_contract=data.contract,
        global_batch=batch,
        context=seq,
        runtime=contract["runtime"],
        head_chunk=a.head_chunk,
        checkpoint_head=objective.checkpoint_head,
        compilation=objective.compilation,
        execution_graph=graph_contract(a.cuda_graph, static_gradient_sync=a.static_gradient_sync),
        communication=communication,
        initialization=contract["initialization"],
        full_weight_start_step=full_weight_start_step,
        full_weight_start_tokens=full_weight_start_tokens,
        source_2b_checkpoint=source_2b_checkpoint,
        source_2b_attention_backend=source_2b_attention_backend,
        warmup_schedule=warmup_schedule,
        target_tokens=a.tokens,
        backbone_lr=a.backbone_lr if trainable_mode == "full" else None,
        backbone_warmup_steps=a.backbone_warmup_steps if trainable_mode == "full" else None,
    )
    if trainable_mode == "adapter":
        checkpoint_extra["frozen_sha256"] = base_hash
    prefetch = (
        WindowPrefetcher(
            data,
            step=step,
            rank=rank,
            world_size=world,
            microbatch=a.microbatch,
            accumulation=a.accumulation,
        )
        if a.prefetch_windows
        else None
    )
    try:
        initial_val = evaluate()
        if rank == 0:
            (a.output / "validation.jsonl").open("a").write(
                json.dumps(dict(step=step, tokens=tokens, loss=initial_val)) + "\n"
            )
        if (
            trainable_mode == "full"
            and a.resume
            and prior.get("trainable_mode", "adapter") == "adapter"
        ):
            if rank == 0:
                ckpt = save_adapter(
                    model, optimizer, a.output, step, tokens, a.oss, extra=checkpoint_extra
                )
                (a.output / "LATEST.json").write_text(
                    json.dumps(dict(checkpoint=str(ckpt), step=step, tokens=tokens))
                )
            barrier()
        while tokens < a.tokens and (not a.steps or step < a.steps):
            started = time.perf_counter()
            if training_graph is None:
                optimizer.zero_grad(set_to_none=True)
            prefetched = prefetch.get(step) if prefetch is not None else None
            remaining = min(a.tokens - tokens, batch * seq)
            total = torch.zeros((), device=local)
            if trainable_mode == "full":
                lr, backbone_lr = warmup_learning_rates(step, tokens, warmup_schedule)
            else:
                lr = (
                    1e-4
                    * min(1.0, (step + 1) / 100)
                    * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * tokens / a.tokens)))
                )
            adapter_groups = (
                optimizer.adapter.param_groups
                if trainable_mode == "full"
                else optimizer.param_groups
            )
            for g in adapter_groups:
                g["lr"] = lr
            if trainable_mode == "full":
                for g in optimizer.backbone.param_groups:
                    g["lr"] = backbone_lr
            for acc in range(a.accumulation):
                global_rows = (
                    step * batch + (acc * world + rank) * a.microbatch + np.arange(a.microbatch)
                )
                rows = (
                    prefetched[acc]
                    if prefetched is not None
                    else np.stack([data[int(i)] for i in global_rows])
                )
                host = torch.from_numpy(rows)
                if prefetched is not None:
                    host = host.pin_memory()
                ids = host.to(local, non_blocking=True)
                labels = ids[:, 1:].clone()
                offset = ((acc * world + rank) * a.microbatch) * seq
                positions = (
                    torch.arange(a.microbatch * seq, device=local).view(a.microbatch, seq) + offset
                )
                labels[positions >= remaining] = -100
                sync = (
                    ddp.no_sync()
                    if world > 1
                    and schedule in ("bucketed", "bucketed-fp32")
                    and acc + 1 < a.accumulation
                    else contextlib.nullcontext()
                )
                with sync:
                    if training_graph is None:
                        loss = ddp(ids[:, :-1], labels) * world / remaining
                        loss.backward()
                    else:
                        loss = training_graph.backward(
                            ids[:, :-1], labels, remaining_tokens=remaining
                        )
                        if training_graph.replays == 1:
                            (a.output / f"CUDA_GRAPH-rank{rank}.json").write_text(
                                json.dumps(training_graph.runtime())
                            )
                    total += loss.detach() / world
            if world > 1 and schedule == "deferred":
                if a.static_gradient_sync:
                    training_graph.synchronize_gradients()
                else:
                    synchronize_gradients(train)
            norm = torch.nn.utils.clip_grad_norm_(train, 1.0)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("nonfinite adapter gradient")
            optimizer.step()
            step += 1
            tokens += remaining
            # Detached loss reporting and stop voting share one collective.
            # A positive sum retains the previous any-rank stop behavior.
            control = torch.stack(
                (total, total.new_tensor(int((a.output / "STOP_REQUEST").exists())))
            )
            if world > 1:
                dist.all_reduce(control)
            total, stop = control.unbind()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            if rank == 0:
                # Dense linear useful FLOPs only; omit embeddings and attention, and do not
                # count frozen weights as if their gradients were calculated.
                embed = model.model.base.embed_tokens.weight.numel()
                flops_token = (
                    6 * (contract["trainable_parameters"] - embed)
                    if trainable_mode == "full"
                    else 4 * (contract["frozen_parameters"] - embed)
                    + 6 * contract["trainable_parameters"]
                )
                metric = dict(
                    step=step,
                    tokens=tokens,
                    loss=float(total),
                    gradient_norm=float(norm),
                    lr=lr,
                    seconds=seconds,
                    tokens_per_second=remaining / seconds,
                    linear_mfu_lower_bound=remaining / seconds * flops_token / (world * 2.25e15),
                    peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                )
                if trainable_mode == "full":
                    metric["backbone_lr"] = backbone_lr
                if useful_flops is not None:
                    metric["model_mfu"] = useful_flops.mfu(remaining, seconds, world)
                (a.output / "metrics.jsonl").open("a").write(json.dumps(metric) + "\n")
                print(json.dumps(metric), flush=True)
                if mlflow:
                    mlflow.log_metrics(
                        {
                            (f"{a.metric_segment}_{k}" if a.metric_segment else k): v
                            for k, v in metric.items()
                            if k != "step"
                        },
                        step=step,
                        synchronous=False,
                    )
                if step >= 3:
                    (a.output / "TRAINING_HEALTHY.json").write_text(json.dumps(metric))
            milestone = tokens >= next_checkpoint or tokens >= a.tokens
            if step == 3 or milestone or bool(stop) or (a.steps and step >= a.steps):
                if training_graph is not None:
                    (a.output / f"CUDA_GRAPH-rank{rank}.json").write_text(
                        json.dumps(training_graph.runtime())
                    )
                if trainable_mode == "adapter" and frozen_fingerprint(model) != base_hash:
                    raise RuntimeError("frozen base changed")
                if rank == 0:
                    ckpt = save_adapter(
                        model,
                        optimizer,
                        a.output,
                        step,
                        tokens,
                        a.oss if milestone else None,
                        extra=checkpoint_extra,
                    )
                    (a.output / "LATEST.json").write_text(
                        json.dumps(dict(checkpoint=str(ckpt), step=step, tokens=tokens))
                    )
                barrier()
                if milestone:
                    next_checkpoint += 2_000_000_000
            if step % 500 == 0 or milestone:
                val = evaluate()
                if rank == 0:
                    (a.output / "validation.jsonl").open("a").write(
                        json.dumps(dict(step=step, tokens=tokens, loss=val)) + "\n"
                    )
                    if mlflow:
                        mlflow.log_metric(
                            f"{a.metric_segment}_validation_loss"
                            if a.metric_segment
                            else "validation_loss",
                            val,
                            step=step,
                        )
            if bool(stop):
                break
        if rank == 0:
            if tokens >= a.tokens:
                (a.output / "WARMUP_COMPLETE.json").write_text(
                    (a.output / "LATEST.json").read_text()
                )
            if mlflow:
                mlflow.end_run()
    except BaseException:
        import traceback

        (a.output / f"FAILED-rank{rank}.txt").write_text(traceback.format_exc())
        raise
    finally:
        if prefetch is not None:
            prefetch.close()
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
