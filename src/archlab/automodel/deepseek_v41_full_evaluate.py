# ruff: noqa: I001  # Preserve the checkpoint-qualified import order.
"""Paired GPU-only evaluation of full V4.1 checkpoints on their16-rank mesh."""

from __future__ import annotations
import argparse
from contextlib import ExitStack
import datetime
import hashlib  # noqa: F401 — preserve checkpoint-qualified executable AST
import json
import math
import os
from pathlib import Path
import time
import traceback
import subprocess


def resolve(value):
    if value.startswith("env:"):
        return Path(os.environ[value[4:]]).resolve()
    if value.startswith("package:"):
        return Path(__file__).resolve().parents[1] / value[8:]
    raise ValueError(f"use an environment or package path: {value}")


def append(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def gather(packet):
    import torch.distributed as dist

    packets = [None] * dist.get_world_size()
    dist.all_gather_object(packets, packet)
    return packets


def construct(checkpoint, variant, *, weights, assets, tiny, output):
    import torch
    import torch.distributed as dist
    from archlab.artifacts import atomic_write_json
    from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
    from archlab.automodel.deepseek_v41_full_eval_construct import build_eval_shell
    from archlab.automodel.deepseek_v41_official_adapter import install_official_adapters
    from archlab.automodel.deepseek_v41_full_boundaries import install_full_training_boundaries
    from archlab.automodel.deepseek_v41_full_indexer import install_trainable_indexers
    from archlab.automodel.deepseek_v41_full_eval_checkpoint import restore_weights, marker_contract

    saved = marker_contract(checkpoint, variant=variant, expected_tokens=None if tiny else 91035439)
    # Keep CheckpointWrapper names and the exact trained forward implementations.
    # No activations or gradients are recorded because evaluation disables grad.
    model, setup, loading = build_eval_shell(weights=weights, assets=assets, tiny=tiny)
    for field in ("container_image", "packages", "cuda", "nccl", "reproducibility"):
        if loading[field] != saved["contract"]["runtime"][field]:
            raise ValueError(f"wrong trained runtime: {field}")
    config = V41AdapterConfig(width=256) if tiny else V41AdapterConfig()
    adapters = install_official_adapters(  # noqa: F841 — preserve checkpoint-qualified executable AST
        model,
        config,
        layer_indices=(1, 3, 5) if tiny else (4, 9, 14, 19, 24, 29, 34, 39),
        device="cuda",
        variant=variant,
        backend="deterministic" if variant == "simplicial" else "flash-attn-deterministic",
    )
    install_full_training_boundaries(model)
    install_trainable_indexers(model)
    receipt = restore_weights(
        model, checkpoint, variant=variant, expected_tokens=None if tiny else 91035439
    )
    atomic_write_json(
        output / f"{variant}-rank{dist.get_rank():02d}-restore.json",
        {"loading": loading, **receipt},
    )
    torch.cuda.empty_cache()
    return model


def qualify_forward(models, pilots, checkpoints, output, *, tiny):
    import torch
    import torch.distributed as dist
    from archlab.automodel.deepseek_v41_loss import frozen_head_loss
    from archlab.automodel.deepseek_v41_official_training import frozen_head_unsharded
    from archlab.automodel.deepseek_v41_official_mesh_probe import _batch
    from archlab.artifacts import atomic_write_json

    rank, world = dist.get_rank(), dist.get_world_size()
    report = {}
    for variant, model in models.items():
        marker = json.loads((checkpoints[variant] / "COMPLETE.json").read_text())
        if tiny:
            expected = json.loads(
                (checkpoints[variant].parent / "rank-00-restored-update.json").read_text()
            )["loss"]
            batches = []
            for micro in range(2):
                ids, labels = _batch(rank + micro * world, 128)
                batches.append((ids, labels, int((labels != -100).sum())))
        else:
            step = marker["cursor"]["step"]
            reference_file = checkpoints[variant].parents[1] / "train-metrics.jsonl"
            reference = next(
                json.loads(line)
                for line in reference_file.read_text().split("\n")
                if line.strip() and json.loads(line)["step"] == step + 1
            )
            expected = reference["loss"]
            batches = [
                pilots.batch(step * 32 + micro * world + rank, device="cuda") for micro in range(2)
            ]
        total = torch.zeros(2, device="cuda", dtype=torch.float64)
        for ids, labels, count in batches:
            hidden = model(input_ids=ids, return_hidden_states=True).hidden_states
            with frozen_head_unsharded(model.lm_head) as head:
                total[0] += frozen_head_loss(hidden, labels, head).double()
            total[1] += count
            del hidden
        dist.all_reduce(total)
        observed = float(total[0] / total[1])
        difference = observed - expected
        if not math.isfinite(observed) or abs(difference) > 1e-4:
            raise ValueError(
                f"{variant}: restored inference differs from saved training forward: {observed} vs {expected}"
            )
        report[variant] = {
            "passed": True,
            "training_reference_ce": expected,
            "eval_ce": observed,
            "absolute_ce_error": abs(difference),
            "tolerance": 1e-4,
            "targets": int(total[1]),
        }
        if rank == 0:
            print(
                json.dumps(
                    {"event": "eval_forward_qualified", "variant": variant, **report[variant]}
                ),
                flush=True,
            )
    if rank == 0:
        atomic_write_json(output / "FORWARD_QUALIFICATION.json", report, allow_nan=False)
    return report


def one_math_pair(models, inputs, labels, *, order, head_context=None):
    import torch
    from archlab.automodel.deepseek_v41_official_training import frozen_head_unsharded

    hidden = {
        variant: models[variant](input_ids=inputs, return_hidden_states=True).hidden_states
        for variant in order
    }
    positions = torch.where(labels[0] != -100)[0]
    # Count, NLL/top1/top5/entropy per model, two KL directions, argmax agreement.
    totals = torch.zeros(12, device=inputs.device, dtype=torch.float64)
    totals[0] = len(positions)
    with ExitStack() as stack:
        context = frozen_head_unsharded if head_context is None else head_context
        heads = {v: stack.enter_context(context(models[v].lm_head)) for v in models}
        for ids in positions.split(128):
            targets = labels[0, ids]
            logp = {}
            pred = {}
            for index, variant in enumerate(("simplicial", "normal")):
                logits = torch.nn.functional.linear(
                    hidden[variant][0, ids].float(), heads[variant].weight
                )
                if not bool(logits.isfinite().all()):
                    raise FloatingPointError("nonfinite validation logits")
                logp[variant] = logits.log_softmax(-1)
                pred[variant] = logits.argmax(-1)
                first = 1 + index * 4
                totals[first] += -logp[variant].gather(1, targets[:, None]).sum(dtype=torch.float64)
                totals[first + 1] += (pred[variant] == targets).sum()
                totals[first + 2] += (logits.topk(5, -1).indices == targets[:, None]).any(-1).sum()
                totals[first + 3] += -(logp[variant].exp() * logp[variant]).sum(dtype=torch.float64)
            s, n = logp["simplicial"], logp["normal"]
            totals[9] += (s.exp() * (s - n)).sum(dtype=torch.float64)
            totals[10] += (n.exp() * (n - s)).sum(dtype=torch.float64)
            totals[11] += (pred["simplicial"] == pred["normal"]).sum()
    if not bool(totals.isfinite().all()):
        raise FloatingPointError("nonfinite paired math metrics")
    values = totals.tolist()
    result = {"targets": int(values[0])}
    for i, variant in enumerate(("simplicial", "normal")):
        result[variant] = dict(
            zip(("nll", "top1", "top5", "entropy"), values[1 + 4 * i : 5 + 4 * i], strict=True)
        )
    result.update(
        simplicial_to_normal_kl_sum=values[9],
        normal_to_simplicial_kl_sum=values[10],
        argmax_agreement_count=int(values[11]),
    )
    return result


def math_evaluate(models, pilot, output, *, limit=None, head_context=None):
    import torch
    import torch.distributed as dist
    from archlab.evaluation.deepseek_v41_compare_metrics import math_summary
    from archlab.artifacts import atomic_write_json

    rank, world = dist.get_rank(), dist.get_world_size()
    count = min(len(pilot), limit) if limit else len(pilot)
    records = []
    for round_index, start in enumerate(range(0, count, world)):
        index = start + rank
        real = index < count
        began = time.monotonic()
        if real:
            inputs, labels, targets = pilot.batch(index, device="cuda")
            item = pilot.windows[index]
        else:
            inputs = torch.zeros(1, 128, device="cuda", dtype=torch.long)
            labels = torch.full_like(inputs, -100)
            targets = 0
        order = ("simplicial", "normal") if round_index % 2 == 0 else ("normal", "simplicial")
        value = one_math_pair(models, inputs, labels, order=order, head_context=head_context)
        if value["targets"] != targets:
            raise ValueError("validation mask lost or duplicated targets")
        packet = (
            {
                **value,
                "pilot_index": index,
                "mode": item["mode"],
                "has_tools": item["has_tools"],
                "problem_sha256": item["problem_sha256"],
                "source_length": item["length"],
            }
            if real
            else None
        )
        packets = gather(packet)
        if rank == 0:
            for row in packets:
                if row is not None:
                    records.append(row)
                    append(output / "math-pairs.jsonl", row)
            event = {
                "event": "heldout_math_round",
                "round": round_index + 1,
                "windows_complete": min(start + world, count),
                "windows_total": count,
                "seconds": time.monotonic() - began,
            }
            append(output / "progress.jsonl", event)
            print(json.dumps(event), flush=True)
    if rank == 0:
        if not limit and sum(r["targets"] for r in records) != 1000000:
            raise ValueError("full validation target count differs")
        summary = math_summary(records)
        atomic_write_json(output / "HELDOUT_MATH.json", summary, allow_nan=False)
        return summary
    return None


def score_job(model, job, *, head_context=None):
    import torch
    from archlab.automodel.deepseek_v41_official_training import frozen_head_unsharded

    ids = torch.tensor([job["input_ids"]], device="cuda", dtype=torch.long)
    hidden = model(input_ids=ids, return_hidden_states=True).hidden_states
    context = frozen_head_unsharded if head_context is None else head_context
    with context(model.lm_head) as head:
        if "fast_targets" in job:
            logits = torch.nn.functional.linear(
                hidden[0, job["prefix_length"] - 1].float(), head.weight
            )
            value = logits.log_softmax(-1)[job["fast_targets"]].tolist()
            if not all(math.isfinite(x) for x in value):
                raise FloatingPointError("nonfinite multiple-choice score")
            return value
        targets = torch.tensor(job["targets"], device="cuda")
        first = job["prefix_length"] - 1
        total = 0.0
        for offset in range(0, len(targets), 128):
            selected = targets[offset : offset + 128]
            logits = torch.nn.functional.linear(
                hidden[0, first + offset : first + offset + len(selected)].float(), head.weight
            )
            total += float(
                logits.log_softmax(-1).gather(1, selected[:, None]).sum(dtype=torch.float64)
            )
        if not math.isfinite(total):
            raise FloatingPointError("nonfinite continuation likelihood")
        return total


def mc_evaluate(models, cases, jobs, output, *, head_context=None):
    import torch.distributed as dist
    from archlab.evaluation.deepseek_v41_compare_metrics import score_choices, mc_summary
    from archlab.artifacts import atomic_write_json

    rank, world = dist.get_rank(), dist.get_world_size()
    scores = {r["id"]: {v: [None] * len(r["choices"]) for v in models} for r in cases}
    lookup = {r["id"]: r for r in cases}
    completed = set()
    records = []
    for round_index, start in enumerate(range(0, len(jobs), world)):
        real = start + rank < len(jobs)
        job = jobs[start + rank] if real else jobs[start]
        began = time.monotonic()
        order = ("simplicial", "normal") if round_index % 2 == 0 else ("normal", "simplicial")
        values = {v: score_job(models[v], job, head_context=head_context) for v in order}
        packets = gather({"job": job, "scores": values} if real else None)
        if rank == 0:
            for packet in packets:
                if packet is None:
                    continue
                j = packet["job"]
                entry = scores[j["id"]]
                for variant, value in packet["scores"].items():
                    if "fast_targets" in j:
                        entry[variant] = value
                    else:
                        entry[variant][j["choice"]] = value
                if j["id"] not in completed and all(None not in x for x in entry.values()):
                    case = lookup[j["id"]]
                    record = {**case, "scores": entry, "result": score_choices(case, entry)}
                    records.append(record)
                    completed.add(j["id"])
                    append(output / "multiple-choice-pairs.jsonl", record)
            event = {
                "event": "multiple_choice_round",
                "round": round_index + 1,
                "jobs_complete": min(start + world, len(jobs)),
                "jobs_total": len(jobs),
                "cases_complete": len(completed),
                "cases_total": len(cases),
                "seconds": time.monotonic() - began,
            }
            append(output / "progress.jsonl", event)
            print(json.dumps(event), flush=True)
    if rank == 0:
        if len(records) != len(cases):
            raise ValueError("incomplete multiple-choice evaluation")
        result = mc_summary(records)
        atomic_write_json(output / "MULTIPLE_CHOICE.json", result, allow_nan=False)
        return result
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(args.recipe.read_text())
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(resolve(config["container_kernel_packages"]))
    import torch
    import torch.distributed as dist
    from transformers import PreTrainedTokenizerFast
    from archlab.artifacts import atomic_write_json
    from archlab.evaluation.deepseek_v41_compare_data import (
        sha,
        read_jsonl,
        build_jobs,
        jobs_digest,
    )
    from archlab.automodel.deepseek_v41_official_execution import configure_official_reproducibility

    if config["world_size"] != 16 or config["ep_size"] != 8 or config["cpu_weight_offload"]:
        raise ValueError("use the exclusive GPU-only16-rank evaluation contract")
    assets, weights, data_root, train_root, val_root, output = (
        resolve(config[k])
        for k in ("assets", "weights", "data", "train_pilot", "validation_pilot", "output")
    )
    checkpoints = {
        v: resolve(config[("tiny_" if args.tiny else "") + v + "_checkpoint"])
        for v in ("simplicial", "normal")
    }
    manifest = json.loads((data_root / "MANIFEST.json").read_text())
    if (
        sha(data_root / "cases.jsonl") != manifest["cases_sha256"]
        or sha(resolve(config["prompts"])) != manifest["prompts_sha256"]
    ):
        raise ValueError("evaluation cases or prompts changed")
    cases = read_jsonl(data_root / "cases.jsonl")
    templates = yaml.safe_load(resolve(config["prompts"]).read_text())
    tokenizer = PreTrainedTokenizerFast.from_pretrained(assets, local_files_only=True)
    jobs = build_jobs(tokenizer, cases, templates)
    if jobs_digest(jobs) != manifest["jobs_digest"]:
        raise ValueError("runtime tokenization differs from the sealed job plan")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "passed": True,
                    "cases": len(cases),
                    "jobs": len(jobs),
                    "rounds": math.ceil(len(jobs) / 16),
                    "max_context": max(len(j["input_ids"]) for j in jobs),
                }
            )
        )
        return
    from archlab.automodel.deepseek_v41_data import MathPilot

    source_root = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        commit != os.environ["NGA_EXPECTED_COMMIT"]
        or subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain"], text=True
        ).strip()
    ):
        raise ValueError("use the recorded clean evaluation source snapshot")
    evaluation_files = [
        "automodel/deepseek_v41_full_evaluate.py",
        "automodel/deepseek_v41_full_eval_checkpoint.py",
        "automodel/deepseek_v41_full_eval_construct.py",
        "evaluation/deepseek_v41_compare_data.py",
        "evaluation/deepseek_v41_compare_metrics.py",
    ]
    evaluation_hashes = {name: sha(source_root / "src/archlab" / name) for name in evaluation_files}
    configure_official_reproducibility()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.manual_seed(42)
    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(minutes=90), device_id=torch.device("cuda", local)
    )
    rank = dist.get_rank()
    try:
        if dist.get_world_size() != 16:
            raise ValueError("evaluation requires exactly16 GPUs")
        status = [None]
        if rank == 0:
            try:
                output.mkdir(parents=True, exist_ok=False)
            except OSError as error:
                status[0] = str(error)
        dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise ValueError(status[0])
        run = {
            "config": config,
            "checkpoint_paths": {v: str(p) for v, p in checkpoints.items()},
            "data_manifest": manifest,
            "recipe_sha256": sha(args.recipe),
            "evaluator_sha256": sha(Path(__file__)),
            "tiny": args.tiny,
            "cpu_weight_offload": False,
            "shared_resident_model_pair": True,
            "project_commit": commit,
            "evaluation_source_sha256": evaluation_hashes,
        }
        if rank == 0:
            atomic_write_json(output / "RUN.json", run, allow_nan=False)
        models = {}
        for variant in ("simplicial", "normal"):
            models[variant] = construct(
                checkpoints[variant],
                variant,
                weights=weights,
                assets=assets,
                tiny=args.tiny,
                output=output,
            )
        train = (
            None
            if args.tiny
            else MathPilot(train_root, expected_split="train", expected_budget=1000000000)
        )
        qualification = qualify_forward(models, train, checkpoints, output, tiny=args.tiny)
        validation = MathPilot(val_root, expected_split="validation", expected_budget=1000000)
        if sha(val_root / "PILOT_READY.json") != manifest["validation_manifest_sha256"]:
            raise ValueError("validation pilot identity differs")
        if args.tiny:
            # Exercise real paired metrics and the frozen FSDP head lifetime on
            # short deterministic data, without scoring a random model as results.
            from archlab.automodel.deepseek_v41_official_mesh_probe import _batch

            ids, labels = _batch(rank, 128)
            packet = one_math_pair(models, ids, labels, order=("simplicial", "normal"))
            if packet["targets"] != 120:
                raise ValueError("tiny paired metric target count differs")
            probe_cases = [
                next(row for row in cases if row["task"] == task)
                for task in ("mmlu", "arc_challenge", "piqa")
            ]
            probe_jobs = build_jobs(tokenizer, probe_cases, templates)
            mc = mc_evaluate(models, probe_cases, probe_jobs, output)  # noqa: F841 — preserve checkpoint-qualified executable AST
            atomic_write_json(
                output / f"rank{rank:02d}-qualified.json", {"passed": True, "math_metrics": packet}
            )
            dist.barrier()
            if rank == 0:
                atomic_write_json(
                    output / "COMPLETE.json",
                    {
                        "passed": True,
                        "world_size": 16,
                        "qualification": qualification,
                        "not_benchmark_results": True,
                        "evaluation_source_sha256": evaluation_hashes,
                        "data_manifest_sha256": sha(data_root / "MANIFEST.json"),
                    },
                )
            return
        admitted = json.loads((resolve(config["qualification"]) / "COMPLETE.json").read_text())
        if (
            not admitted["passed"]
            or admitted["world_size"] != 16
            or admitted.get("evaluation_source_sha256") != evaluation_hashes
            or admitted.get("data_manifest_sha256") != sha(data_root / "MANIFEST.json")
        ):
            raise ValueError("the same evaluator and data must pass tiny16 frozen paired inference")
        heldout = math_evaluate(models, validation, output)
        multiple_choice = mc_evaluate(models, cases, jobs, output)
        if rank == 0:
            result = {
                "passed": True,
                "complete": True,
                "checkpoint_tokens": 91035439,
                "world_size": 16,
                "cpu_weight_offload": False,
                "qualification": qualification,
                "heldout_math": heldout,
                "multiple_choice": multiple_choice,
                "benchmark_scope": manifest["benchmark_scope"],
            }
            atomic_write_json(output / "COMPLETE.json", result, allow_nan=False)
            print(
                json.dumps({"event": "paired_evaluation_complete", "output": str(output)}),
                flush=True,
            )
        dist.barrier()
    except BaseException:
        if output.is_dir():
            atomic_write_json(
                output / f"FAILED-rank{rank:02d}.json", {"traceback": traceback.format_exc()}
            )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
