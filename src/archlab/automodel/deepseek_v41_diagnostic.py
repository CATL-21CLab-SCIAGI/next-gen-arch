"""Localize frozen V4.1 native/replacement drift; never admits or starts training.

Full-tensor finite counts and norms accompany bounded token-row samples at
layer boundaries. Layer equality is explicitly sampled, while final hidden
states are compared in full. No reference equations or runtime files change.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _positions(length, count):
    positions = {0, length - 1}
    for position in (1, 31, 32, 63, 64, 127, 128, 511, 512, 1023, 1024, 2047, 2048):
        if len(positions) >= count:
            break
        if position < length:
            positions.add(position)
    for index in range(count):
        if len(positions) >= count:
            break
        positions.add(index * (length - 1) // max(1, count - 1))
    return sorted(positions)


def _snapshot(tensor, *, sample_rows=32, chunk_rows=256):
    import torch

    value = tensor.detach()
    # All observed boundaries have [B=1,S,...] or flattened [S,...].
    axis = 1 if value.ndim >= 3 and value.shape[0] == 1 else 0
    rows = value.shape[axis]
    positions = _positions(rows, sample_rows)
    selected = torch.tensor(positions, dtype=torch.long, device=value.device)
    sample = value.index_select(axis, selected).to(device="cpu", copy=True)
    totals = torch.zeros(6, device=value.device, dtype=torch.float64)
    minimum = torch.full((), torch.inf, device=value.device, dtype=torch.float64)
    maximum = torch.full((), -torch.inf, device=value.device, dtype=torch.float64)
    for first in range(0, rows, chunk_rows):
        chunk = value.narrow(axis, first, min(chunk_rows, rows - first)).double()
        finite = chunk.isfinite()
        clean = torch.where(finite, chunk, 0)
        totals += torch.stack(
            (
                finite.sum(),
                chunk.isnan().sum(),
                chunk.isposinf().sum(),
                chunk.isneginf().sum(),
                clean.sum(),
                clean.square().sum(),
            )
        )
        minimum = torch.minimum(minimum, chunk.masked_fill(~finite, torch.inf).amin())
        maximum = torch.maximum(maximum, chunk.masked_fill(~finite, -torch.inf).amax())
    finite, nan, posinf, neginf, total, squared = totals.cpu().tolist()
    statistics = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": value.numel(),
        "finite": int(finite) == value.numel(),
        "finite_count": int(finite),
        "nan_count": int(nan),
        "positive_inf_count": int(posinf),
        "negative_inf_count": int(neginf),
        "finite_sum": total,
        "finite_l2": math.sqrt(squared),
        "finite_min": float(minimum.cpu()),
        "finite_max": float(maximum.cpu()),
        "sample_axis": axis,
        "sample_rows": positions,
    }
    return _json_safe(statistics), sample


def _compare_tensors(expected, actual, *, chunk_elements=262144):
    """Bound temporary CPU memory, including the full final-hidden comparison."""
    import torch

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {"same_shape_dtype": False, "equal": False}
    left, right = expected.reshape(-1), actual.reshape(-1)
    equal, same_nonfinite = True, True
    squared_error = squared_reference = max_abs = 0.0
    finite_pairs = 0
    for first in range(0, left.numel(), chunk_elements):
        a, b = (
            left[first : first + chunk_elements].double(),
            right[first : first + chunk_elements].double(),
        )
        equal &= torch.equal(a, b)
        same_nonfinite &= (
            torch.equal(a.isnan(), b.isnan())
            and torch.equal(a.isposinf(), b.isposinf())
            and torch.equal(a.isneginf(), b.isneginf())
        )
        valid = a.isfinite() & b.isfinite()
        difference = torch.where(valid, b - a, 0)
        squared_error += float(difference.square().sum())
        squared_reference += float(torch.where(valid, a, 0).square().sum())
        max_abs = max(max_abs, float(difference.abs().max()))
        finite_pairs += int(valid.sum())
    return _json_safe(
        {
            "same_shape_dtype": True,
            "equal": equal,
            "same_nonfinite_pattern": same_nonfinite,
            "finite_pair_count": finite_pairs,
            "numel": left.numel(),
            "finite_pair_max_abs": max_abs,
            "finite_pair_relative_l2": math.sqrt(squared_error / max(squared_reference, 1e-40)),
        }
    )


def _observed_modules(model):
    yield "embed", model.embed
    for index, layer in enumerate(model.layers):
        prefix = f"layers.{index}"
        if layer.engram is not None:
            yield prefix + ".engram", layer.engram
        for name in ("attn_norm", "attn", "ffn_norm", "ffn"):
            yield prefix + "." + name, getattr(layer, name)
        if layer.attn.compressor is not None:
            yield prefix + ".attn.compressor", layer.attn.compressor
        if layer.attn.indexer is not None:
            yield prefix + ".attn.indexer", layer.attn.indexer
        yield prefix + ".ffn.gate", layer.ffn.gate
        yield prefix, layer
    yield "norm", model.norm


def _capture(
    model,
    forward,
    *,
    context,
    stage,
    emit,
    sample_rows,
    reference=None,
    divergence_relative_l2=1e-3,
):
    import torch

    captures, handles, final_hidden = {}, [], []
    outcome = {
        "context": context,
        "stage": stage,
        "first_nonfinite": None,
        "first_block_nonfinite": None,
        "first_unequal_sample": None,
        "first_sampled_divergence": None,
        "first_block_unequal_sample": None,
        "first_block_sampled_divergence": None,
    }

    def observe(name, tensor):
        statistics, sample = _snapshot(tensor, sample_rows=sample_rows)
        captures[name] = (statistics, sample)
        record = {"context": context, "stage": stage, "boundary": name, **statistics}
        block = name.startswith("layers.") and name.count(".") == 2 and name.endswith(".0")
        if not statistics["finite"] and outcome["first_nonfinite"] is None:
            outcome["first_nonfinite"] = name
        if block and not statistics["finite"] and outcome["first_block_nonfinite"] is None:
            outcome["first_block_nonfinite"] = name
        if reference is not None:
            comparison = _compare_tensors(reference[name][1], sample)
            record["sample_comparison"] = comparison
            unequal = not comparison["equal"]
            relative = comparison.get("finite_pair_relative_l2")
            drift = (
                not comparison.get("same_nonfinite_pattern", False)
                or relative is None
                or relative > divergence_relative_l2
            )
            for condition, key in (
                (unequal, "first_unequal_sample"),
                (drift, "first_sampled_divergence"),
                (block and unequal, "first_block_unequal_sample"),
                (block and drift, "first_block_sampled_divergence"),
            ):
                if condition and outcome[key] is None:
                    outcome[key] = name
        emit("boundary", **record)

    def hook(name, module, values, output):
        if isinstance(output, torch.Tensor):
            observe(name, output)
        elif isinstance(output, (tuple, list)):
            for index, item in enumerate(output):
                if isinstance(item, torch.Tensor):
                    observe(f"{name}.{index}", item)
        if name == "norm":
            final_hidden.append(output.detach().to(device="cpu", copy=True))

    from functools import partial

    for name, module in _observed_modules(model):
        handles.append(module.register_forward_hook(partial(hook, name)))
    start = time.perf_counter()
    try:
        with torch.no_grad():
            forward()
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    outcome["seconds"] = time.perf_counter() - start
    outcome["observed_boundaries"] = len(captures)
    outcome["all_boundaries_finite"] = outcome["first_nonfinite"] is None
    if len(final_hidden) != 1:
        raise RuntimeError("expected exactly one final normalized hidden state")
    if reference is not None and captures.keys() != reference.keys():
        raise RuntimeError("native and candidate observed different module boundaries")
    emit("forward_complete", **outcome)
    return captures, final_hidden[0], outcome


def run(args):
    import torch
    import torch.distributed as dist

    from archlab.automodel.deepseek_v41_autograd import training_hidden
    from archlab.automodel.deepseek_v41_data import MathPilot
    from archlab.automodel.deepseek_v41_execution import build_replica, node_local_expert_group
    from archlab.automodel.deepseek_v41_pytorch import dequantize_base_once, install_pytorch_leaves
    from archlab.automodel.deepseek_v41_runtime import (
        implementation_hashes,
        select_container_kernel_packages,
    )

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=90),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    rank = dist.get_rank()
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    report = {
        "kind": "full-pretrained-forward-diagnostic",
        "rank": rank,
        "training_launched": False,
        "qualification_receipt": False,
        "container_image": os.environ.get("NGA_CONTAINER_DIGEST"),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "nccl": list(torch.cuda.nccl.version()),
        "implementation_sha256": implementation_hashes(),
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matmul": {
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        },
        "layer_comparisons": "bounded token-row samples; finite counts/norms cover all elements",
        "final_hidden_comparisons": "all elements",
        "tests": [],
    }
    stream = (args.output / f"rank{rank}.jsonl").open("x")

    def emit(event, **values):
        record = _json_safe({"event": event, "rank": rank, "unix_time": time.time(), **values})
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        if event != "boundary":
            print(json.dumps(record, allow_nan=False), flush=True)

    try:
        report["kernel_packages"] = select_container_kernel_packages(args.container_kernel_packages)
        emit("construct_and_load_start", weights=str(args.weights))
        group = node_local_expert_group(args.ep_size)
        reference, model, report["load"] = build_replica(
            assets=args.assets, weights=args.weights, group=group, context=max(args.contexts)
        )
        data = MathPilot(args.pilot, expected_split="train", expected_budget=1_000_000_000)
        oracles = {}
        capture_args = {
            "model": model,
            "emit": emit,
            "sample_rows": args.sample_rows,
            "divergence_relative_l2": args.divergence_relative_l2,
        }
        for context in sorted(set(args.contexts)):
            inputs, _, _ = data.batch(rank, device="cuda", smoke_context=context, pad_to_full=True)
            first, hidden, original = _capture(
                forward=lambda inputs=inputs: model(inputs), context=context, stage="native_first", **capture_args
            )
            repeat, repeated_hidden, repeated = _capture(
                forward=lambda inputs=inputs: model(inputs),
                context=context,
                stage="native_repeat",
                reference=first,
                **capture_args,
            )
            repeated["full_hidden_comparison"] = _compare_tensors(hidden, repeated_hidden)
            emit("native_repeat_comparison", **repeated)
            report["tests"].extend((original, repeated))
            oracles[context] = first, hidden
            del repeat, repeated_hidden
        report["conversion"] = dequantize_base_once(model)
        emit(
            "bf16_conversion_complete", converted_modules=report["conversion"]["converted_modules"]
        )
        install_pytorch_leaves(reference, query_chunk=args.query_chunk, activation_mode="native")
        for context, (first, hidden) in oracles.items():
            inputs, _, _ = data.batch(rank, device="cuda", smoke_context=context, pad_to_full=True)
            candidate, candidate_hidden, outcome = _capture(
                forward=lambda inputs=inputs: training_hidden(reference, model, inputs),
                context=context,
                stage="pytorch_native_rounding",
                reference=first,
                **capture_args,
            )
            outcome["full_hidden_comparison"] = _compare_tensors(hidden, candidate_hidden)
            emit("replacement_comparison", **outcome)
            report["tests"].append(outcome)
            del candidate, candidate_hidden
        report["diagnostic_complete"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        emit("diagnostic_failed", error=report["error"])
        raise
    finally:
        stream.close()
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(_json_safe(report), indent=2, allow_nan=False)
        )
    dist.barrier()
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
    parser.add_argument("--sample-rows", type=int, default=32)
    parser.add_argument("--divergence-relative-l2", type=float, default=1e-3)
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 2048, 16384])
    args = parser.parse_args()
    if args.sample_rows < 2 or min(args.contexts) < 2 or args.divergence_relative_l2 <= 0:
        parser.error("positive contexts/threshold and at least two sampled rows are required")
    run(args)


if __name__ == "__main__":
    main()
