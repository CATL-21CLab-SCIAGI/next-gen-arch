"""Qualify the actual pretrained BF16 base on an existing EP GPU group.

This is a disposable numerical/memory test, not the 1B-token training job.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path


def run(args):
    import torch
    import torch.distributed as dist

    from archlab.automodel.deepseek_v41_autograd import training_hidden
    from archlab.automodel.deepseek_v41_data import MathPilot
    from archlab.automodel.deepseek_v41_execution import (
        adapter_optimizers,
        build_replica,
        enable_activation_checkpointing,
        install_training_branches,
        node_local_expert_group,
    )
    from archlab.automodel.deepseek_v41_pytorch import (
        bf16_memory_plan,
        dequantize_base_once,
        install_pytorch_leaves,
    )
    from archlab.automodel.deepseek_v41_runtime import (
        implementation_hashes,
        select_container_kernel_packages,
    )
    from archlab.automodel.deepseek_v41_training import (
        emit,
        optimizer_step,
        restore_adapter_checkpoint,
        save_adapter_checkpoint,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=90), device_id=torch.device("cuda", local_rank))
    report = {"kind": "full-pretrained-pytorch-qualification", "rank": dist.get_rank(), "passed": False,
              "training_launched": False, "torch": str(torch.__version__),
              "container_image": os.environ.get("NGA_CONTAINER_DIGEST"), "tests": []}
    report["implementation_sha256"] = implementation_hashes()
    if dist.get_rank() == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    report["kernel_packages_for_native_oracle"] = select_container_kernel_packages(args.container_kernel_packages)
    group = node_local_expert_group(args.ep_size)
    emit("construct_and_load_start", weights=str(args.weights))
    reference, model, report["load"] = build_replica(assets=args.assets, weights=args.weights, group=group,
                                                    context=max(args.contexts))
    plan = bf16_memory_plan(model)
    emit("bf16_memory_preflight", **plan)
    if not plan["fits"]:
        raise MemoryError(str(plan))
    data = MathPilot(args.pilot, expected_split="train", expected_budget=1_000_000_000)
    # Save compact CPU hidden states before one-time weight conversion. Include
    # contexts where top-512 selection actually prunes keys, not only 128-token
    # inputs on which all compressed keys remain visible.
    oracles = {}
    comparison_contexts = sorted({128, min(2048, max(args.contexts)), max(args.contexts)})
    for context in comparison_contexts:
        inputs, _, _ = data.batch(dist.get_rank(), device="cuda", smoke_context=context, pad_to_full=True)
        hidden_capture, routes, index_routes = [], [], []
        hooks = [model.norm.register_forward_hook(lambda module, values, out, dest=hidden_capture: dest.append(out.detach().cpu().clone()))]
        hooks += [layer.ffn.gate.register_forward_hook(
            lambda module, values, out, dest=routes: dest.append(out[1].detach().cpu().clone())) for layer in model.layers]
        hooks += [layer.attn.indexer.register_forward_hook(
            lambda module, values, out, dest=index_routes: dest.append(out.detach().cpu().clone()))
            for layer in model.layers if layer.attn.indexer is not None]
        begin = time.perf_counter()
        try:
            model(inputs)
        finally:
            for hook in hooks:
                hook.remove()
        oracles[context] = (hidden_capture.pop(), routes, index_routes)
        torch.cuda.synchronize()
        emit("native_forward_complete", context=context, seconds=time.perf_counter() - begin)

    report["conversion"] = dequantize_base_once(model)
    emit("bf16_conversion_complete", **{k: v for k, v in report["conversion"].items() if k != "conversion_shapes"})
    install_pytorch_leaves(reference, query_chunk=args.query_chunk)

    def compare(mode, context):
        reference._archlab_activation_mode = mode
        inputs, labels, targets = data.batch(dist.get_rank(), device="cuda", smoke_context=context, pad_to_full=True)
        native_hidden, routes, index_routes = oracles[context]
        actual_routes, actual_indices = [], []
        hooks = [layer.ffn.gate.register_forward_hook(
            lambda module, values, out: actual_routes.append(out[1].detach().cpu())) for layer in model.layers]
        hooks += [layer.attn.indexer.register_forward_hook(
            lambda module, values, out: actual_indices.append(out.detach().cpu().clone()))
            for layer in model.layers if layer.attn.indexer is not None]
        start = time.perf_counter()
        with torch.no_grad():
            hidden = training_hidden(reference, model, inputs)
            totals = torch.zeros(6, device="cuda", dtype=torch.float64)
            for first in range(0, context, 128):
                last = first + 128
                native_logits = model.head(native_hidden[:, first:last].to(hidden.device), full_logits=True)
                logits = model.head(hidden[:, first:last], full_logits=True)
                logp, logq = native_logits.log_softmax(-1), logits.log_softmax(-1)
                chunk_labels = labels[:, first:last]
                valid = chunk_labels != -100
                if valid.any():
                    totals[0] += torch.nn.functional.cross_entropy(native_logits[valid], chunk_labels[valid], reduction="sum")
                    totals[1] += torch.nn.functional.cross_entropy(logits[valid], chunk_labels[valid], reduction="sum")
                totals[2] += (logits - native_logits).square().sum(dtype=torch.float64)
                totals[3] += native_logits.square().sum(dtype=torch.float64)
                totals[4] += (logp.exp() * (logp - logq)).sum(dtype=torch.float64)
                totals[5] += (logits.argmax(-1) == native_logits.argmax(-1)).sum()
            loss_native, loss = (totals[:2] / targets).tolist()
            result = {"mode": mode, "context": context, "native_loss": loss_native, "loss": loss,
                      "delta_loss": loss - loss_native,
                      "relative_logits_error": float((totals[2] / totals[3]).sqrt()),
                      "native_to_candidate_kl": float(totals[4] / inputs.numel()),
                      "argmax_agreement": float(totals[5] / inputs.numel()),
                      "targets": targets}
        for hook in hooks:
            hook.remove()
        result["expert_set_agreement"] = sum(
            int((a.sort(-1).values == b.sort(-1).values).all(-1).sum())
            for a, b in zip(routes, actual_routes, strict=True)) / (40 * inputs.numel())
        result["sparse_index_row_agreement"] = sum(
            int((a == b).all(-1).sum()) for a, b in zip(index_routes, actual_indices, strict=True)
        ) / (len(index_routes) * inputs.numel())
        torch.cuda.synchronize()
        result["seconds"] = time.perf_counter() - start
        emit("activation_comparison", **result)
        report["tests"].append(result)
        return hidden, result

    accepted = True
    for context in comparison_contexts:
        hidden, rounded = compare("native", context)
        if context == 128:
            rounded_hidden = hidden
        accepted &= rounded["native_to_candidate_kl"] < .02 and abs(rounded["delta_loss"]) < .05
        del hidden
    compare("bf16", 128)
    reference._archlab_activation_mode = "native"
    # Fail closed on meaningful drift; never call a changed model native-exact.
    decision = torch.tensor(int(accepted), device="cuda", dtype=torch.int32)
    dist.all_reduce(decision, op=dist.ReduceOp.MIN)
    if not decision.item():
        report["stop_reason"] = "native-style BF16 forward exceeds qualification drift gate"
    else:
        del oracles
        inputs, labels, _ = data.batch(dist.get_rank(), device="cuda", smoke_context=128, pad_to_full=True)
        adapters = install_training_branches(reference, model)
        with torch.no_grad():
            zero = training_hidden(reference, model, inputs)
        torch.testing.assert_close(zero, rounded_hidden, rtol=0, atol=0)
        del zero, rounded_hidden
        frozen = [(p, p._version) for p in model.parameters() if not p.requires_grad]
        enable_activation_checkpointing(model)
        optimizers = adapter_optimizers(model, adapters)
        for step in range(2):
            metric = optimizer_step(reference, model, optimizers, inputs, labels, learning_rate=1e-5)
            emit("adapter_gradient_step", step=step, **metric)
            for i, adapter in adapters.items():
                for name, p in adapter.named_parameters():
                    if p.grad is None or not bool(p.grad.isfinite().all()):
                        raise AssertionError((i, name, "missing/nonfinite gradient"))
                    if step == 1 and not bool(p.grad.count_nonzero()):
                        raise AssertionError((i, name, "zero second-step gradient"))
            report["tests"].append({"optimizer_step": step, **metric})
        if any(p._version != version or p.grad is not None for p, version in frozen):
            raise AssertionError("frozen base was modified or received parameter gradients")
        contract = {"kind": "disposable-pytorch-qualification", "world": dist.get_world_size(),
                    "index_sha256": report["load"]["index_sha256"]}
        checkpoint_path = args.output / "checkpoint-roundtrip"
        cursor = {"step": 2, "supervised_tokens": 0, "not_production": True}
        save_adapter_checkpoint(checkpoint_path, adapters, optimizers, cursor, contract)
        original = next(iter(adapters.values())).output.weight.detach().clone()
        with torch.no_grad():
            next(iter(adapters.values())).output.weight.add_(1)
        restored = restore_adapter_checkpoint(checkpoint_path, adapters, optimizers, contract=contract)
        torch.testing.assert_close(next(iter(adapters.values())).output.weight, original, rtol=0, atol=0)
        assert restored == cursor
        emit("checkpoint_roundtrip_passed")
        for context in args.contexts:
            inputs, labels, _ = data.batch(dist.get_rank(), device="cuda", smoke_context=context, pad_to_full=True)
            torch.cuda.reset_peak_memory_stats()
            metric = optimizer_step(reference, model, optimizers, inputs, labels, learning_rate=1e-7)
            emit("context_profile", context=context, **metric)
            report["tests"].append({"context": context, **metric})
        report["passed"] = True
    with (args.output / f"rank{dist.get_rank()}.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    dist.barrier()
    if not report["passed"]:
        raise RuntimeError(report["stop_reason"])
    emit("full_qualification_passed")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--container-kernel-packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--query-chunk", type=int, default=32)
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 2048, 4096, 8192, 16384])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
