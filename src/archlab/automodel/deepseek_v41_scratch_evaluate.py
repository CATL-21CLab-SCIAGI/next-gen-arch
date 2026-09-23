"""One-variant, eight-GPU evaluation of a matched scratch checkpoint."""

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


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def json_lines(text):
    # Unicode separators may occur inside JSON strings; only LF delimits JSONL.
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def checkpoint_marker(path, variant, tokens):
    marker = json.loads((path / "COMPLETE.json").read_text())
    c = marker["contract"]
    if (
        marker["format"] != "archlab-v41-full-sharded-v1"
        or marker["world_size"] != 8
        or c["world_size"] != 8
        or c["variant"] != variant
    ):
        raise ValueError("wrong scratch checkpoint variant or mesh")
    if (
        c["format"] != "archlab-v41-scratch-comparison-v1"
        or marker["cursor"]["supervised_tokens"] != tokens
    ):
        raise ValueError("wrong checkpoint contract or matched token count")
    return marker


def restore(model, path, marker):
    import torch
    import torch.distributed as dist

    from archlab.automodel.deepseek_v41_full_checkpoint import _checksum
    from archlab.optimizers.sharded_adafactor import local_tensor

    rank = dist.get_rank()
    folder = path / f"rank-{rank:02d}"
    m = json.loads((folder / "MANIFEST.json").read_text())
    for key in ("world_size", "cursor", "contract"):
        if m[key] != marker[key]:
            raise ValueError(f"rank manifest {key} mismatch")
    if m["rank"] != rank:
        raise ValueError("rank identity mismatch")
    named = list(model.named_parameters()) + list(model.named_buffers())
    count = 0
    size = 0
    with torch.no_grad():
        for (name, tensor), entry in zip(named, m["tensors"], strict=True):
            local = local_tensor(tensor)
            if (
                name != entry["name"]
                or list(local.shape) != entry["shape"]
                or list(tensor.shape) != entry["global_shape"]
                or str(local.dtype) != entry["dtype"]
                or not local.is_contiguous()
            ):
                raise ValueError(f"checkpoint tensor mismatch: {name}")
            flat = local.view(-1)
            offset = 0
            for chunk in entry["chunks"]:
                host = torch.load(folder / chunk["file"], map_location="cpu", weights_only=True)
                if (
                    host.dtype != local.dtype
                    or host.numel() != chunk["elements"]
                    or _checksum(host) != chunk["sha256"]
                ):
                    raise ValueError(f"checksum mismatch: {name}")
                flat[offset : offset + host.numel()].copy_(host)
                offset += host.numel()
                size += host.numel() * host.element_size()
                count += 1
                del host
            if offset != flat.numel():
                raise ValueError(f"incomplete tensor: {name}")
    model.requires_grad_(False)
    model.eval()
    dist.barrier()
    if any(local_tensor(p).device.type != "cuda" for p in model.parameters()):
        raise ValueError("weights must remain on GPU")
    return {
        "rank": rank,
        "cursor": marker["cursor"],
        "tensor_entries": len(named),
        "verified_chunks": count,
        "bytes": size,
        "all_weight_checksums_verified": True,
        "optimizer_loaded": False,
        "cpu_weight_offload": False,
    }


def token_metrics(model, ids, labels):
    import torch

    from archlab.automodel.deepseek_v41_official_training import frozen_head_unsharded

    hidden = model(
        input_ids=ids, attention_mask=labels != -100, return_hidden_states=True
    ).hidden_states
    positions = torch.where(labels.flatten() != -100)[0]
    total = torch.zeros(5, device="cuda", dtype=torch.float64)
    total[0] = len(positions)
    with frozen_head_unsharded(model.lm_head) as head:
        x = hidden.flatten(0, 1)
        y = labels.flatten()
        for indices in positions.split(128):
            if not indices.numel():
                continue
            logits = torch.nn.functional.linear(x[indices].float(), head.weight)
            if not bool(logits.isfinite().all()):
                raise ValueError("nonfinite logits")
            logp = logits.log_softmax(-1)
            targets = y[indices]
            total[1] += -logp.gather(1, targets[:, None]).sum(dtype=torch.float64)
            total[2] += (logits.argmax(-1) == targets).sum()
            total[3] += (logits.topk(5, -1).indices == targets[:, None]).any(-1).sum()
            total[4] += -(logp.exp() * logp).sum(dtype=torch.float64)
    return total


def score_job(model, job):
    import torch

    from archlab.automodel.deepseek_v41_official_training import frozen_head_unsharded

    ids = torch.tensor([job["input_ids"]], device="cuda", dtype=torch.long)
    hidden = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids, dtype=torch.bool),
        return_hidden_states=True,
    ).hidden_states
    with frozen_head_unsharded(model.lm_head) as head:
        if "fast_targets" in job:
            logits = torch.nn.functional.linear(
                hidden[0, job["prefix_length"] - 1].float(), head.weight
            )
            value = logits.log_softmax(-1)[job["fast_targets"]].tolist()
        else:
            targets = torch.tensor(job["targets"], device="cuda")
            first = job["prefix_length"] - 1
            value = 0.0
            for offset in range(0, len(targets), 128):
                selected = targets[offset : offset + 128]
                logits = torch.nn.functional.linear(
                    hidden[0, first + offset : first + offset + len(selected)].float(), head.weight
                )
                value += float(
                    logits.log_softmax(-1).gather(1, selected[:, None]).sum(dtype=torch.float64)
                )
    if not all(math.isfinite(x) for x in (value if isinstance(value, list) else [value])):
        raise ValueError("nonfinite choice likelihood")
    return value


def choice_result(case, scores):
    if len(scores) != len(case["choices"]) or any(
        x is None or not math.isfinite(x) for x in scores
    ):
        raise ValueError("incomplete choice scores")
    lengths = [1] * len(scores) if case["task"] == "mmlu" else [len(x) for x in case["choices"]]
    if min(lengths) <= 0:
        raise ValueError("empty choice")
    prediction = max(range(len(scores)), key=scores.__getitem__)
    normalized = [x / n for x, n in zip(scores, lengths, strict=True)]
    norm = max(range(len(scores)), key=normalized.__getitem__)
    return {
        "prediction": prediction,
        "normalized_prediction": norm,
        "accuracy": int(prediction == case["answer"]),
        "accuracy_norm": int(norm == case["answer"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["normal", "simplicial"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-tokens", type=int, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(Path(os.environ["ARCHLAB_CONTAINER_KERNEL_PACKAGES"]))
    import torch
    import torch.distributed as dist

    from archlab.automodel import deepseek_v41_scratch_construct as construction
    from archlab.automodel.deepseek_v41_scratch_data import ScratchData
    from archlab.automodel.deepseek_v41_scratch_training import (
        build_batches,
        canonical_contract,
        training_runtime,
    )

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=90),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    rank = dist.get_rank()
    began = time.monotonic()
    try:
        if dist.get_world_size() != 8:
            raise ValueError("scratch evaluation requires 8 GPUs")
        marker = checkpoint_marker(args.checkpoint, args.variant, args.checkpoint_tokens)
        source = Path(construction.__file__).resolve().parents[1]
        for relative, digest in marker["contract"]["implementation_sha256"].items():
            if hashlib.sha256((source / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"trained source changed: {relative}")
        data_contract = json.loads((args.data / "CONTRACT.json").read_text())
        if data_contract != marker["contract"]["data_contract"]:
            raise ValueError("scratch data contract mismatch")
        bm = json.loads((args.benchmarks / "MANIFEST.json").read_text())
        cases_text = (args.benchmarks / "cases.jsonl").read_text()
        cases = json_lines(cases_text)
        jobs = json.loads((args.benchmarks / "jobs.json").read_text())
        if (
            hashlib.sha256(cases_text.encode()).hexdigest() != bm["cases_sha256"]
            or hashlib.sha256(
                json.dumps(jobs, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            != bm["jobs_digest"]
        ):
            raise ValueError("benchmark data changed")
        if max(len(j["input_ids"]) for j in jobs) > 2048:
            raise ValueError("benchmark exceeds trained context")
        contract = {
            "variant": args.variant,
            "checkpoint": str(args.checkpoint),
            "cursor": marker["cursor"],
            "world_size": 8,
            "cpu_weight_offload": False,
            "validation_targets": 1000000,
            "benchmark_counts": bm["counts"],
            "benchmark_cases_sha256": bm["cases_sha256"],
            "benchmark_jobs_digest": bm["jobs_digest"],
            "prompts_sha256": bm["prompts_sha256"],
            "benchmark_overlap_with_scratch_training": "not assessed",
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        if rank == 0:
            write(out / "RUN_CONTRACT.json", contract)
        model, _, gates, loading = construction.construct_scratch(
            base_config=os.environ["ARCHLAB_DEEPSEEK_V41_BASE_CONFIG"],
            assets=os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"],
            variant=args.variant,
        )
        if canonical_contract(training_runtime(loading)) != marker["contract"]["runtime"]:
            raise ValueError("constructed runtime/model differs from training")
        receipt = restore(model, args.checkpoint, marker)
        write(out / f"rank-{rank:02d}-restore.json", receipt)
        for gate in gates:
            gate._track_load_balance = False
        with torch.no_grad():
            # Check eval-mode forward against the unchanged train-mode forward at
            # these exact restored weights, before producing benchmark scores.
            data = ScratchData(args.data, wait_seconds=30)
            ids, labels, count = build_batches(data, marker["cursor"]["window_cursor"], rank)[0]
            parity = []
            for training in (True, False):
                model.train(training)
                hidden = model(
                    input_ids=ids, attention_mask=labels != -100, return_hidden_states=True
                ).hidden_states
                loss = model.lm_head.loss(hidden, labels)
                values = torch.tensor([float(loss), count], device="cuda", dtype=torch.float64)
                dist.all_reduce(values)
                parity.append(float(values[0] / values[1]))
                del hidden
            if not all(math.isfinite(x) for x in parity) or abs(parity[0] - parity[1]) > 1e-4:
                raise ValueError(f"train/eval forward mismatch: {parity}")
            if rank == 0:
                write(
                    out / "FORWARD_QUALIFIED.json",
                    {
                        "passed": True,
                        "training_mode_ce": parity[0],
                        "eval_mode_ce": parity[1],
                        "absolute_error": abs(parity[0] - parity[1]),
                        "tolerance": 1e-4,
                        "method": "same restored weights and next training batch; no optimizer step",
                    },
                )
            del ids, labels, data
            validation = ScratchData(args.data, split="validation", wait_seconds=30)
            total = torch.zeros(5, device="cuda", dtype=torch.float64)
            round_index = 0
            while int(total[0]) < 1000000:
                index = round_index * 8 + rank
                ids, labels, count = validation.batch([index], device="cuda")
                stats = token_metrics(model, ids, labels)
                if int(stats[0]) != count:
                    raise ValueError("validation target mask differs")
                packets = [None] * 8
                dist.all_gather_object(
                    packets,
                    {
                        "window": index,
                        "targets": count,
                        "nll": float(stats[1]),
                        "top1": int(stats[2]),
                        "top5": int(stats[3]),
                        "entropy": float(stats[4]),
                    }
                    if count
                    else None,
                )
                dist.all_reduce(stats)
                if not int(stats[0]):
                    raise ValueError("validation ended early")
                total += stats
                round_index += 1
                if rank == 0:
                    for packet in packets:
                        if packet is not None:
                            append(out / "validation-windows.jsonl", packet)
                    append(
                        out / "progress.jsonl",
                        {
                            "event": "validation",
                            "round": round_index,
                            "targets": int(total[0]),
                            "seconds": time.monotonic() - began,
                        },
                    )
            if int(total[0]) != 1000000:
                raise ValueError("validation budget mismatch")
            if rank == 0:
                t = total.tolist()
                write(
                    out / "HELDOUT_FINEWEB.json",
                    {
                        "targets": int(t[0]),
                        "cross_entropy": t[1] / t[0],
                        "perplexity": math.exp(t[1] / t[0]),
                        "top1_token_accuracy": t[2] / t[0],
                        "top5_token_accuracy": t[3] / t[0],
                        "mean_predictive_entropy": t[4] / t[0],
                    },
                )
            lookup = {c["id"]: c for c in cases}
            scores = {c["id"]: [None] * len(c["choices"]) for c in cases}
            completed = set()
            records = []
            for start in range(0, len(jobs), 8):
                index = start + rank
                real = index < len(jobs)
                job = jobs[index] if real else jobs[start]
                score = score_job(model, job)
                packets = [None] * 8
                dist.all_gather_object(packets, {"job": job, "score": score} if real else None)
                if rank == 0:
                    for packet in packets:
                        if packet is None:
                            continue
                        j = packet["job"]
                        key = j["id"]
                        if "fast_targets" in j:
                            scores[key] = packet["score"]
                        else:
                            scores[key][j["choice"]] = packet["score"]
                        if key not in completed and all(x is not None for x in scores[key]):
                            c = lookup[key]
                            row = {
                                "id": key,
                                "task": c["task"],
                                "subject": c.get("subject"),
                                "answer": c["answer"],
                                "scores": scores[key],
                                "result": choice_result(c, scores[key]),
                            }
                            records.append(row)
                            completed.add(key)
                            append(out / "multiple-choice.jsonl", row)
                    append(
                        out / "progress.jsonl",
                        {
                            "event": "multiple_choice",
                            "jobs": min(start + 8, len(jobs)),
                            "jobs_total": len(jobs),
                            "cases": len(records),
                            "seconds": time.monotonic() - began,
                        },
                    )
            if rank == 0:
                if len(records) != len(cases):
                    raise ValueError("missing benchmark cases")
                summary = {}
                for task in sorted({r["task"] for r in records}):
                    rows = [r for r in records if r["task"] == task]
                    summary[task] = {
                        "questions": len(rows),
                        **{
                            key: sum(r["result"][key] for r in rows) / len(rows)
                            for key in ("accuracy", "accuracy_norm")
                        },
                    }
                write(out / "MULTIPLE_CHOICE.json", summary)
        dist.barrier()
        if rank == 0:
            write(
                out / "COMPLETE.json",
                {
                    "passed": True,
                    "variant": args.variant,
                    "cursor": marker["cursor"],
                    "seconds": time.monotonic() - began,
                    "heldout_fineweb": json.loads((out / "HELDOUT_FINEWEB.json").read_text()),
                    "multiple_choice": json.loads((out / "MULTIPLE_CHOICE.json").read_text()),
                },
            )
    except BaseException:
        write(out / f"rank-{rank:02d}-failure.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
