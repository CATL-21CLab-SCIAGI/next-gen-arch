"""Compare the official full backbone with the released BF16-compute reference."""

from __future__ import annotations

import gc
import hashlib
import json
import socket
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from archlab.artifacts import atomic_write_json
from archlab.automodel.deepseek_v41_official_execution import official_core_hashes
from archlab.automodel.deepseek_v41_official_reference import (
    build_matching_precision_reference,
    reference_hidden,
)
from archlab.automodel.deepseek_v41_official_training import frozen_head_unsharded
from archlab.automodel.deepseek_v41_training import emit


@contextmanager
def _block_samples(layers, context):
    """Retain small CPU samples to localize a failed full-checkpoint comparison."""
    captured, handles = {}, []
    positions = sorted({i for i in (0, 1, 31, 32, 63, 64, 127, context - 1) if i < context})

    def capture(index, module, inputs, output):
        captured[index] = tuple(value.detach()[:, positions].cpu().clone() for value in output[:2])

    from functools import partial

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(partial(capture, index)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _compare_block_samples(expected, actual):
    from archlab.automodel.deepseek_v41_diagnostic import _compare_tensors

    if expected.keys() != actual.keys():
        raise ValueError("official and reference traces cover different blocks")
    return [
        {
            "layer_0based": index,
            "streams": _compare_tensors(expected[index][0], actual[index][0]),
            "pre_mix": _compare_tensors(expected[index][1], actual[index][1]),
        }
        for index in expected
    ]


def _collective_finite(tensor, label):
    good = torch.tensor(int(bool(tensor.isfinite().all())), device="cuda", dtype=torch.int32)
    dist.all_reduce(good, op=dist.ReduceOp.MIN)
    if not good.item():
        raise FloatingPointError(f"nonfinite {label} on at least one rank")


@torch.no_grad()
def compare_hidden(reference, actual, labels, head, *, chunk_size=128):
    if reference.shape != actual.shape or labels.shape != actual.shape[:2]:
        raise ValueError("parity hidden/label shapes differ")
    totals = torch.zeros(7, dtype=torch.float64, device="cuda")
    for first in range(0, actual.shape[1], chunk_size):
        last = first + chunk_size
        ref_logits = F.linear(
            reference[:, first:last].to(device="cuda", dtype=torch.float32), head.weight
        )
        logits = F.linear(actual[:, first:last].to(device="cuda", dtype=torch.float32), head.weight)
        targets = labels[:, first:last]
        valid = targets != -100
        if valid.any():
            totals[0] += F.cross_entropy(ref_logits[valid], targets[valid], reduction="sum")
            totals[1] += F.cross_entropy(logits[valid], targets[valid], reduction="sum")
        totals[2] += (logits - ref_logits).square().sum(dtype=torch.float64)
        totals[3] += ref_logits.square().sum(dtype=torch.float64)
        logp, logq = ref_logits.log_softmax(-1), logits.log_softmax(-1)
        totals[4] += (logp.exp() * (logp - logq)).sum(dtype=torch.float64)
        totals[5] += (ref_logits.argmax(-1) == logits.argmax(-1)).sum()
        totals[6] += valid.sum()
    _collective_finite(totals, "full-vocabulary parity metrics")
    count = int(totals[6])
    if count < 1:
        raise ValueError("parity window contains no supervised target")
    reference_loss, loss = (totals[:2] / count).tolist()
    result = {
        "reference_loss": reference_loss,
        "loss": loss,
        "delta_loss": loss - reference_loss,
        "relative_logits_error": float((totals[2] / totals[3]).sqrt()),
        "native_to_candidate_kl": float(totals[4] / labels.numel()),
        "argmax_agreement": float(totals[5] / labels.numel()),
        "supervised_targets": count,
    }
    result["passed"] = result["native_to_candidate_kl"] < 0.02 and abs(result["delta_loss"]) < 0.05
    return result


def tensor_digest(tensor):
    """Exact shape/dtype/byte identity with bounded device-to-host copies."""
    value = tensor.detach().contiguous()
    digest = hashlib.sha256(json.dumps([list(value.shape), str(value.dtype)]).encode())
    for chunk in value.reshape(-1).split(4 * 1024 * 1024):
        digest.update(chunk.cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def batch_digest(inputs, labels):
    return {"inputs": tensor_digest(inputs), "labels": tensor_digest(labels)}


def validate_prepared_reference(prepared, *, rank, contexts):
    if (
        prepared["rank"] != rank
        or prepared["contexts"] != list(contexts)
        or set(prepared["outputs"]) != set(contexts)
        or prepared["reference_loading"]["expert_owner_ranks"]
        != list(range(rank // 8 * 8, rank // 8 * 8 + 8))
        or prepared["reference_loading"]["engram_owner_ranks"] != list(range(32))
        or not prepared["head_sha256"]
    ):
        raise ValueError(
            "reference must cover this rank and all contexts with matching EP8/Engram32 ownership"
        )


@torch.no_grad()
def prepare_official_reference(
    *, assets, weights, pilot, output: Path, contexts=(128, 2048, 16384)
):
    """Run the independent EP8 oracle before allocating the official backbone.

    Matching expert owners also matches the native per-expert GEMM batch shapes.
    Keeping only CPU outputs between phases avoids the previous co-resident16K
    memory pressure without changing reference arithmetic or ownership.
    """
    rank = dist.get_rank()
    if dist.get_world_size() != 32:
        raise ValueError("reference preparation requires the production32-rank world")
    output.mkdir(parents=True, exist_ok=True)
    groups = [dist.new_group(list(range(first, first + 8))) for first in range(0, 32, 8)]
    group = groups[rank // 8]
    hosts = [None] * 8
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    if len(set(hosts)) != 1:
        raise ValueError("reference expert groups must match the node-local production EP8")
    prepared = {"rank": rank, "contexts": list(contexts), "outputs": {}}
    reference = native = None
    try:
        reference, native, loading = build_matching_precision_reference(
            assets=assets,
            weights=weights,
            expert_group=group,
            engram_group=dist.group.WORLD,
            context=max(contexts),
        )
        prepared["reference_loading"] = loading
        prepared["head_sha256"] = tensor_digest(native.head.weight)
        receipt = {key: value for key, value in prepared.items() if key != "outputs"}
        receipt["tests"] = []
        for context in contexts:
            inputs, labels, _ = pilot.batch(
                rank, device="cuda", smoke_context=context, pad_to_full=True
            )
            with _block_samples(native.layers, context) as samples:
                hidden = reference_hidden(native, inputs)
            _collective_finite(hidden, f"reference hidden at context {context}")
            identity = batch_digest(inputs, labels)
            prepared["outputs"][context] = {
                "hidden": hidden.cpu().clone(),
                "blocks": samples,
                "batch_sha256": identity,
            }
            receipt["tests"].append(
                {
                    "context": context,
                    "finite": True,
                    "batch_sha256": identity,
                    "hidden_sha256": tensor_digest(hidden),
                }
            )
            atomic_write_json(output / f"reference-rank{rank}.json", receipt, allow_nan=False)
            emit(
                "official_native_reference_complete",
                context=context,
                hidden_shape=list(hidden.shape),
                local_allocated_gib=torch.cuda.memory_allocated() / 2**30,
            )
            del hidden, inputs, labels
    finally:
        if reference is not None:
            reference.shared_attn = reference.SharedAttentionRuntime()
        del native, reference
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        dist.destroy_process_group(group)
    emit("official_reference_released", local_allocated_gib=torch.cuda.memory_allocated() / 2**30)
    return prepared


@torch.no_grad()
def qualify_official_forward(
    model, setup, *, pilot, output: Path, prepared_reference, contexts=(128, 2048, 16384)
):
    """Compare all ranks with the independently prepared matching-EP8 oracle."""
    rank = dist.get_rank()
    validate_prepared_reference(prepared_reference, rank=rank, contexts=contexts)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    report = {
        "kind": "official-v41-matched-precision-forward",
        "rank": rank,
        "passed": False,
        "implementation_sha256": official_core_hashes(),
        "tests": [],
        "reference_compute": "decoded-bf16-dense-experts-native-kv-index-rounding",
        "reference_loading": prepared_reference["reference_loading"],
        "reference_lifetime": "EP8-oracle-before-official-backbone-allocation",
        "reference_head_sha256": prepared_reference["head_sha256"],
        "thresholds": {"mean_kl": 0.02, "absolute_ce_delta": 0.05},
    }
    baseline = None
    for context in contexts:
        inputs, labels, _ = pilot.batch(
            rank, device="cuda", smoke_context=context, pad_to_full=True
        )
        expected = prepared_reference["outputs"][context]
        if batch_digest(inputs, labels) != expected["batch_sha256"]:
            raise ValueError(
                "official and reference must use the exact same input and supervised labels"
            )
        with _block_samples(model.model.layers.values(), context) as samples:
            hidden = model(input_ids=inputs, return_hidden_states=True).hidden_states
        _collective_finite(hidden, f"official hidden at context {context}")
        emit("official_forward_complete", context=context, hidden_shape=list(hidden.shape))
        with frozen_head_unsharded(model.lm_head) as head:
            if context == contexts[0]:
                decision = torch.tensor(
                    int(tensor_digest(head.weight) == prepared_reference["head_sha256"]),
                    device="cuda",
                    dtype=torch.int32,
                )
                dist.all_reduce(decision, op=dist.ReduceOp.MIN)
                if not decision.item():
                    raise ValueError("official and reference output-head weights differ")
            metric = compare_hidden(expected["hidden"], hidden, labels, head)
        metric["context"] = context
        metric["sampled_block_comparisons"] = _compare_block_samples(expected["blocks"], samples)
        report["tests"].append(metric)
        atomic_write_json(output / f"rank{rank}.json", report, allow_nan=False)
        emit(
            "official_reference_comparison",
            **{k: v for k, v in metric.items() if k != "sampled_block_comparisons"},
        )
        if context == 128:
            baseline = hidden.cpu().clone()
        del hidden, inputs, labels
        context_passed = torch.tensor(int(metric["passed"]), device="cuda", dtype=torch.int32)
        dist.all_reduce(context_passed, op=dist.ReduceOp.MIN)
        if not context_passed.item():
            raise ValueError(
                f"official forward parity failed at {context}; preserving early traces"
            )
    report["passed"] = True
    atomic_write_json(output / f"rank{rank}.json", report, allow_nan=False)
    torch.cuda.empty_cache()
    emit("official_forward_qualified", local_allocated_gib=torch.cuda.memory_allocated() / 2**30)
    return report, baseline
