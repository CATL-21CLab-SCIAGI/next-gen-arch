"""GPU-resident paired V4.1 evaluation on an exclusive 32-GPU allocation.

Execute this file with the checkpoint's immutable source on PYTHONPATH. The
qualified official FSDP32/EP8/Engram32 backend remains unchanged. No optimizer
is restored and no training process is inspected or signaled by this program.
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


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def path_value(value):
    if value.startswith("env:"):
        return Path(os.environ[value[4:]]).resolve()
    if value.startswith("package:"):
        import archlab

        return Path(archlab.__file__).resolve().parent / value[8:]
    raise ValueError("use a portable recipe path")


def enabled(adapters, value):
    for module in adapters:
        module.adapter_enabled = value


def unwrapped(layer):
    while hasattr(layer, "_checkpoint_wrapped_module"):
        layer = layer._checkpoint_wrapped_module
    return layer


def paired_jobs(tokenizer, cases, templates):
    from jinja2 import Environment, StrictUndefined

    env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)
    jobs = []
    for row in cases:
        name = "mmlu" if row["task"] == "mmlu" else "arc_challenge"
        context = env.from_string(templates["multiple_choice"][name]).render(**row)
        choices = (
            [chr(65 + i) for i in range(len(row["choices"]))]
            if row["task"] == "mmlu"
            else row["choices"]
        )
        pairs = []
        for text in choices:
            prefix = tokenizer.encode(context, add_special_tokens=False)
            combined = tokenizer.encode(context + " " + text, add_special_tokens=False)
            if not prefix or combined[: len(prefix)] != prefix or len(combined) == len(prefix):
                raise ValueError("ambiguous benchmark continuation boundary")
            pairs.append((prefix, combined[len(prefix) :]))
        if all(len(target) == 1 and prefix == pairs[0][0] for prefix, target in pairs):
            jobs.append(
                {
                    "id": row["id"],
                    "tokens": pairs[0][0],
                    "prefix_length": len(pairs[0][0]),
                    "fast_targets": [target[0] for _, target in pairs],
                }
            )
        else:
            for choice, (prefix, target) in enumerate(pairs):
                jobs.append(
                    {
                        "id": row["id"],
                        "choice": choice,
                        "tokens": prefix + target[:-1],
                        "prefix_length": len(prefix),
                        "targets": target,
                    }
                )
    return jobs


def append(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def gather(row):
    import torch.distributed as dist

    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, row)
    return rows


def mc_evaluate(model, connections, jobs, cases, output):
    import torch
    import torch.distributed as dist

    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_official_training import frozen_head_unsharded
    from archlab.evaluation.capability import paired_statistics

    rank, world = dist.get_rank(), dist.get_world_size()
    scores = {
        r["id"]: {mode: [None] * len(r["choices"]) for mode in ("base", "adapted")} for r in cases
    }
    for index, start in enumerate(range(0, len(jobs), world)):
        real = start + rank < len(jobs)
        job = jobs[start + rank] if real else jobs[start]
        ids = torch.tensor([job["tokens"]], device="cuda")
        result = {}
        began = time.monotonic()
        order = ("base", "adapted") if index % 2 == 0 else ("adapted", "base")
        for mode in order:
            enabled(connections, mode == "adapted")
            hidden = model(input_ids=ids, return_hidden_states=True).hidden_states
            with frozen_head_unsharded(model.lm_head) as head:
                if "fast_targets" in job:
                    logits = torch.nn.functional.linear(
                        hidden[:, job["prefix_length"] - 1].float(), head.weight
                    )[0]
                    result[mode] = logits.log_softmax(-1)[job["fast_targets"]].tolist()
                else:
                    first = job["prefix_length"] - 1
                    targets = torch.tensor(job["targets"], device="cuda")
                    total = 0.0
                    for offset in range(0, len(targets), 128):
                        labels = targets[offset : offset + 128]
                        logits = torch.nn.functional.linear(
                            hidden[0, first + offset : first + offset + len(labels)].float(),
                            head.weight,
                        )
                        total += float(logits.log_softmax(-1).gather(1, labels[:, None]).sum())
                    result[mode] = total
            del hidden
        packets = gather({"job": job, "scores": result} if real else None)
        if rank == 0:
            for packet in packets:
                if packet is None:
                    continue
                j = packet["job"]
                for mode, value in packet["scores"].items():
                    if "fast_targets" in j:
                        scores[j["id"]][mode] = value
                    else:
                        scores[j["id"]][mode][j["choice"]] = value
            event = {
                "event": "mc_round",
                "round": index + 1,
                "jobs_complete": min(start + world, len(jobs)),
                "jobs_total": len(jobs),
                "seconds": time.monotonic() - began,
            }
            append(output / "rounds.jsonl", event)
            print(json.dumps(event), flush=True)
    if rank != 0:
        return None
    records = []
    for row in cases:
        record = {**row, "scores": scores[row["id"]]}
        lengths = (
            [1] * len(row["choices"]) if row["task"] == "mmlu" else [len(x) for x in row["choices"]]
        )
        for mode, values in record["scores"].items():
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError("incomplete or nonfinite benchmark scores")
            normal = [value / length for value, length in zip(values, lengths, strict=True)]
            record[mode] = {
                "acc": int(max(range(len(values)), key=values.__getitem__) == row["answer"]),
                "acc_norm": int(max(range(len(normal)), key=normal.__getitem__) == row["answer"]),
            }
        records.append(record)
        append(output / "multiple-choice-pairs.jsonl", record)
    summary = {}
    for task in sorted({r["task"] for r in records}):
        subset = [r for r in records if r["task"] == task]
        summary[task] = {
            metric: paired_statistics(
                [r["base"][metric] for r in subset], [r["adapted"][metric] for r in subset]
            )
            for metric in ("acc", "acc_norm")
        }
    atomic_write_json(output / "MULTIPLE_CHOICE.json", summary, allow_nan=False)
    return summary


def full_validation(model, connections, data, output):
    import torch
    import torch.distributed as dist

    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_official_training import frozen_head_unsharded

    rank, world = dist.get_rank(), dist.get_world_size()
    sums = torch.zeros(4, 9, device="cuda", dtype=torch.float64)
    for round_index in range(math.ceil(len(data) / world)):
        index = round_index * world + rank
        inputs, labels, count = data.batch(index, device="cuda")
        began = time.monotonic()
        hidden = {}
        for mode in ("base", "adapted"):
            enabled(connections, mode == "adapted")
            hidden[mode] = model(input_ids=inputs, return_hidden_states=True).hidden_states
        totals = torch.zeros(9, device="cuda", dtype=torch.float64)
        with frozen_head_unsharded(model.lm_head) as head:
            for first in range(0, inputs.shape[1], 128):
                target = labels[:, first : first + 128]
                valid = target != -100
                if not bool(valid.any()):
                    continue
                logp = {}
                prediction = {}
                top5 = {}
                for mode in ("base", "adapted"):
                    logits = torch.nn.functional.linear(
                        hidden[mode][:, first : first + 128].float(), head.weight
                    )[valid]
                    logp[mode] = logits.log_softmax(-1)
                    prediction[mode] = logits.argmax(-1)
                    top5[mode] = logits.topk(5, -1).indices
                targets = target[valid]
                n = targets.numel()
                totals[0] += n
                totals[1] += -logp["base"].gather(1, targets[:, None]).sum(dtype=torch.float64)
                totals[2] += -logp["adapted"].gather(1, targets[:, None]).sum(dtype=torch.float64)
                totals[3] += (prediction["base"] == targets).sum()
                totals[4] += (prediction["adapted"] == targets).sum()
                totals[5] += (top5["base"] == targets[:, None]).any(-1).sum()
                totals[6] += (top5["adapted"] == targets[:, None]).any(-1).sum()
                totals[7] += (logp["base"].exp() * (logp["base"] - logp["adapted"])).sum(
                    dtype=torch.float64
                )
                totals[8] += (prediction["base"] == prediction["adapted"]).sum()
        if int(totals[0]) != count:
            raise ValueError("validation target-mask mismatch")
        sums[0] += totals
        if index < len(data):
            sums[("low", "medium", "high").index(data.windows[index]["mode"]) + 1] += totals
        del hidden, inputs, labels
        dist.barrier()
        if rank == 0:
            event = {
                "event": "validation_round",
                "round": round_index + 1,
                "rounds": math.ceil(len(data) / world),
                "seconds": time.monotonic() - began,
            }
            append(output / "rounds.jsonl", event)
            print(json.dumps(event), flush=True)
    dist.all_reduce(sums)
    if not bool(sums.isfinite().all()) or int(sums[0, 0]) != 1000000:
        raise ValueError("invalid full validation totals")
    if rank != 0:
        return None

    def summarize(values):
        n, lb, la, tb, ta, kb, ka, kl, agree = values.tolist()
        return {
            "supervised_targets": int(n),
            "base": {
                "cross_entropy": lb / n,
                "perplexity": math.exp(lb / n),
                "top1_token_accuracy": tb / n,
                "top5_token_accuracy": kb / n,
            },
            "adapted": {
                "cross_entropy": la / n,
                "perplexity": math.exp(la / n),
                "top1_token_accuracy": ta / n,
                "top5_token_accuracy": ka / n,
            },
            "base_to_adapted_kl": kl / n,
            "token_argmax_agreement": agree / n,
        }

    result = {
        name: summarize(row)
        for name, row in zip(("overall", "low", "medium", "high"), sums, strict=True)
    }
    atomic_write_json(output / "FULL_VALIDATION.json", result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--container-kernel-packages", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(args.container_kernel_packages)
    import torch
    import torch.distributed as dist
    import yaml
    from torch.distributed.tensor import DTensor
    from transformers import PreTrainedTokenizerFast

    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_data import MathPilot
    from archlab.automodel.deepseek_v41_official_adapter import install_official_adapters
    from archlab.automodel.deepseek_v41_official_execution import (
        build_official_base,
        configure_official_reproducibility,
        official_core_hashes,
        runtime_identity,
    )

    config = yaml.safe_load(args.recipe.read_text())
    if (
        config["world_size"] != 32
        or config["ep_size"] != 8
        or config["cpu_weight_offload"]
        or not config["exclusive_gpu_allocation"]
    ):
        raise ValueError("use the qualified exclusive GPU-only world32/EP8 contract")
    assets, weights, checkpoint, data_root, train_root, val_root, output = (
        path_value(config[k])
        for k in (
            "assets",
            "weights",
            "adapter_checkpoint",
            "data",
            "train_pilot",
            "validation_pilot",
            "output",
        )
    )
    marker = json.loads((checkpoint / "COMPLETE.json").read_text())
    saved = marker["contract"]
    for path, key in (
        (weights / "model.safetensors.index.json", "checkpoint_index_sha256"),
        (weights / "config.json", "base_config_sha256"),
        (assets / "inference/config.json", "reference_config_sha256"),
    ):
        if sha(path) != saved[key]:
            raise ValueError(f"wrong frozen base identity: {key}")
    if official_core_hashes() != saved["implementation_sha256"]:
        raise ValueError("execute with the checkpoint immutable source on PYTHONPATH")
    manifest = json.loads((data_root / "MANIFEST.json").read_text())
    if sha(data_root / "cases.jsonl") != manifest["cases_sha256"]:
        raise ValueError("evaluation cases changed")
    cases = lines(data_root / "cases.jsonl")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(assets, local_files_only=True)
    templates = yaml.safe_load(path_value(config["prompts"]).read_text())
    jobs = paired_jobs(tokenizer, cases, templates)
    if max(len(job["tokens"]) for job in jobs) > config["context"]:
        raise ValueError("untruncated benchmark exceeds context")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "passed": True,
                    "cases": len(cases),
                    "jobs": len(jobs),
                    "rounds": math.ceil(len(jobs) / 32),
                    "max_context": max(len(j["tokens"]) for j in jobs),
                }
            )
        )
        return
    configure_official_reproducibility()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(minutes=30), device_id=torch.device("cuda", local)
    )
    rank = dist.get_rank()
    try:
        if dist.get_world_size() != 32:
            raise ValueError("evaluation needs all32 exclusive GPUs")
        status = [None]
        if rank == 0:
            try:
                output.mkdir(parents=True, exist_ok=False)
            except OSError as error:
                status[0] = str(error)
        dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise ValueError(status[0])
        identity = runtime_identity()
        for field, key in (
            ("container_image", "container_image"),
            ("packages", "runtime"),
            ("cuda", "cuda"),
            ("nccl", "nccl"),
            ("reproducibility", "reproducibility"),
        ):
            if identity[field] != saved[key]:
                raise ValueError(f"evaluation runtime differs from trained checkpoint: {field}")
        model, setup, loading = build_official_base(
            weights=weights, assets=assets, ep_size=8, activation_checkpointing=False
        )
        for parameter in model.parameters():
            local_tensor = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            if local_tensor.device.type != "cuda":
                raise ValueError("CPU weight offload is forbidden")
        model.eval()
        train = MathPilot(train_root, expected_split="train", expected_budget=1000000000)
        inputs, _, _ = train.batch(rank, device="cuda", smoke_context=128, pad_to_full=True)
        bare = model(input_ids=inputs, return_hidden_states=True).hidden_states
        adapters = install_official_adapters(model, device="cuda", backend="deterministic")
        zero = model(input_ids=inputs, return_hidden_states=True).hidden_states
        good = torch.tensor(int(torch.equal(bare, zero)), device="cuda")
        dist.all_reduce(good, op=dist.ReduceOp.MIN)
        if not good.item():
            raise ValueError("zero-adapter identity failed")
        del bare, zero, inputs, train
        f = checkpoint / "adapter-state.pt"
        if f.stat().st_size != marker["state_bytes"] or sha(f) != marker["state_sha256"]:
            raise ValueError("adapter checkpoint checksum mismatch")
        state = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
        if (
            state["contract"] != saved
            or state["cursor"] != marker["cursor"]
            or set(state["adapters"]) != {str(i) for i in adapters}
        ):
            raise ValueError("adapter checkpoint contract mismatch")
        for index, adapter in adapters.items():
            adapter.load_state_dict(state["adapters"][str(index)], strict=True)
        del state
        model.requires_grad_(False)
        connections = [unwrapped(model.model.layers[str(i)]).attn_hc for i in adapters]
        if rank == 0:
            atomic_write_json(
                output / "RUN.json",
                {
                    "config": config,
                    "checkpoint": str(checkpoint),
                    "cursor": marker["cursor"],
                    "loading": loading,
                    "evaluator_sha256": sha(Path(__file__)),
                    "recipe_sha256": sha(args.recipe),
                    "data_manifest_sha256": sha(data_root / "MANIFEST.json"),
                    "zero_identity_passed": True,
                    "cpu_weight_offload": False,
                },
                allow_nan=False,
            )
        mc = mc_evaluate(model, connections, jobs, cases, output)
        validation = MathPilot(val_root, expected_split="validation", expected_budget=1000000)
        math_results = full_validation(model, connections, validation, output)
        if rank == 0:
            atomic_write_json(
                output / "COMPLETE.json",
                {
                    "passed": True,
                    "checkpoint_cursor": marker["cursor"],
                    "multiple_choice": mc,
                    "heldout_math": math_results,
                    "cpu_weight_offload": False,
                },
                allow_nan=False,
            )
    except BaseException:
        if output.is_dir():
            atomic_write_json(
                output / f"FAILED-rank{rank}.json", {"traceback": traceback.format_exc()}
            )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
