"""Diagnose full-model repeated gradients without admitting production training."""

import argparse
import datetime
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "assets", "pilot", "checkpoint", "output", "container-kernel-packages"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace-all-ranks", action="store_true")
    parser.add_argument("--compare-gather", action="store_true")
    parser.add_argument("--compare-adapter", action="store_true")
    parser.add_argument("--allow-core-change", action="store_true")
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(args.container_kernel_packages)
    import torch
    import torch.distributed as dist

    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_data import MathPilot
    from archlab.automodel.deepseek_v41_execution import adapter_optimizers
    from archlab.automodel.deepseek_v41_official_adapter import install_official_adapters
    from archlab.automodel.deepseek_v41_official_execution import (
        build_official_base,
        configure_official_reproducibility,
        official_core_hashes,
    )
    from archlab.automodel.deepseek_v41_official_qualification import tensor_digest
    from archlab.automodel.deepseek_v41_official_training import (
        _update_vector_statistics,
        capture_adapter_training_state,
        optimizer_step,
        restore_in_memory_training_state,
        state_difference_statistics,
    )
    from archlab.automodel.deepseek_v41_training import emit, restore_adapter_checkpoint

    configure_official_reproducibility()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=30),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    rank = dist.get_rank()
    output_created = False
    try:
        status = [None]
        if rank == 0:
            try:
                args.output.mkdir(parents=True, exist_ok=False)
            except OSError as error:
                status[0] = str(error)
        dist.broadcast_object_list(status, src=0)
        if status[0]:
            raise ValueError(status[0])
        output_created = True
        contract = json.loads((args.checkpoint / "COMPLETE.json").read_text())["contract"]
        current_hashes = official_core_hashes()
        changed = [
            key
            for key, value in current_hashes.items()
            if contract["implementation_sha256"].get(key) != value
        ]
        if changed and not args.allow_core_change:
            raise ValueError("diagnose the exact saved model/training implementation")
        model, _, loading = build_official_base(
            weights=args.weights,
            assets=args.assets,
            activation_checkpointing=not args.no_checkpointing,
        )
        adapters = install_official_adapters(model, device="cuda")
        optimizers = adapter_optimizers(model, adapters)
        restore_adapter_checkpoint(args.checkpoint, adapters, optimizers, contract=contract)
        saved = capture_adapter_training_state(adapters, optimizers)
        data = MathPilot(args.pilot, expected_split="train", expected_budget=1_000_000_000)
        inputs, labels, _ = data.batch(rank, device="cuda", smoke_context=128)
        model.train()
        trace, hidden, handles = {}, {}, []
        trace_rank = rank == 0 or args.trace_all_ranks
        sample_positions = torch.arange(0, inputs.shape[1], 16, device=inputs.device)

        def watch_tensor(key, value):
            if trace_rank and isinstance(value, torch.Tensor) and value.requires_grad:

                def gradient(grad):
                    sampled = (
                        grad.index_select(1, sample_positions)
                        if grad.ndim >= 2 and grad.shape[1] == inputs.shape[1]
                        else grad.flatten()[:4096]
                    )
                    trace[key] = sampled.detach().clone()

                value.register_hook(gradient)

        def watch_module(key, module):
            def before(_module, values):
                if values:
                    watch_tensor(key + ".input", values[0])

            def after(_module, _values, result):
                value = (
                    result[0]
                    if isinstance(result, tuple)
                    else getattr(result, "hidden_states", result)
                )
                watch_tensor(key + ".output", value)

            handles.extend(
                (module.register_forward_pre_hook(before), module.register_forward_hook(after))
            )

        if trace_rank:
            for index, layer in enumerate(model.model.layers.values()):
                if index >= 34:
                    watch_module(f"layer.{index}", layer)
                    for name in ("attn_norm", "attn", "ffn_norm", "ffn"):
                        watch_module(f"layer.{index}.{name}", getattr(layer, name))
                    for name in ("wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "wo_a", "wo_b"):
                        watch_module(f"layer.{index}.attn.{name}", getattr(layer.attn, name))
                    for name in ("gate", "experts", "shared_experts"):
                        watch_module(f"layer.{index}.ffn.{name}", getattr(layer.ffn, name))
            watch_module("final_norm", model.model.norm)
        handles.append(
            model.model.norm.register_forward_hook(
                lambda _module, _values, result: hidden.update(value=result.detach().clone())
            )
        )
        report = {
            "production_qualification": False,
            "rank": rank,
            "loading": loading,
            "activation_checkpointing": not args.no_checkpointing,
            "repeats": [],
            "completed": False,
            "changed_core_files": changed,
            "implementation_sha256": current_hashes,
        }
        baseline = baseline_gradients = baseline_trace = None
        if args.compare_gather and args.compare_adapter:
            raise ValueError("compare one boundary at a time")
        modes = (
            ["atomic_adapter", "deterministic_adapter"]
            if args.compare_adapter
            else ["upstream", "private_clone"]
            if args.compare_gather
            else ["current"]
        )
        for iteration in range(args.repeats * len(modes)):
            mode, repeat = modes[iteration // args.repeats], iteration % args.repeats
            if repeat == 0:
                baseline = baseline_gradients = baseline_trace = None
                if args.compare_adapter:
                    for adapter in adapters.values():
                        adapter.backend = "triton" if mode == "atomic_adapter" else "deterministic"
                if args.compare_gather:
                    from nemo_automodel.components.moe.experts import (
                        GroupedExperts,
                        _AllGatherConcatVarlenFn,
                    )

                    from archlab.automodel.deepseek_v41_official_moe import _NonMutatingVarlenGather

                    for module in model.modules():
                        if isinstance(module, GroupedExperts):
                            module._archlab_ep_gather = (
                                _AllGatherConcatVarlenFn
                                if mode == "upstream"
                                else _NonMutatingVarlenGather
                            )
            restore_in_memory_training_state(saved, adapters, optimizers)
            trace.clear()
            metric = optimizer_step(model, optimizers, inputs, labels, learning_rate=1e-5)
            record = {
                "mode": mode,
                "repeat": repeat,
                "metric": metric,
                "hidden_sha256": tensor_digest(hidden["value"]),
            }
            if rank == 0:
                current = capture_adapter_training_state(adapters, optimizers)
                gradients = {
                    i: {
                        name: p.grad.detach().cpu().clone()
                        * max(metric["gradient_norm_before_clip"], 1.0)
                        for name, p in adapter.named_parameters()
                    }
                    for i, adapter in adapters.items()
                }
                if baseline is None:
                    baseline, baseline_gradients = current, gradients
                record["adapter_update"] = _update_vector_statistics(
                    saved["adapters"], baseline["adapters"], current["adapters"]
                )
                record["raw_gradients"] = state_difference_statistics(baseline_gradients, gradients)
                record["per_adapter_raw_gradients"] = {
                    str(i): state_difference_statistics(baseline_gradients[i], gradients[i])
                    for i in gradients
                }
            if trace_rank:
                gradients_trace = {key: value.cpu() for key, value in trace.items()}
                if baseline_trace is None:
                    baseline_trace = gradients_trace
                record["gradient_boundaries"] = {
                    key: {
                        **state_difference_statistics(baseline_trace[key], value),
                        "baseline_norm": float(baseline_trace[key].double().norm()),
                    }
                    for key, value in gradients_trace.items()
                    if key in baseline_trace
                }
            report["repeats"].append(record)
            atomic_write_json(args.output / f"rank{rank}.json", report, allow_nan=False)
            emit("full_replay_diagnostic", mode=mode, repeat=repeat, **metric)
        report["completed"] = True
        atomic_write_json(args.output / f"rank{rank}.json", report, allow_nan=False)
        dist.barrier()
        if rank == 0:
            atomic_write_json(
                args.output / "DIAGNOSTIC_COMPLETE.json",
                {"completed": True, "production_qualification": False},
            )
    except BaseException:
        import traceback

        failure = {"rank": rank, "traceback": traceback.format_exc()}
        print(json.dumps(failure), flush=True)
        if output_created:
            atomic_write_json(args.output / f"failure-rank{rank}.json", failure)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
