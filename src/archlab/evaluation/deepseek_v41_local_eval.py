"""Paired local-B300 capability and teacher-forced math evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def append(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def encode_pair(tokenizer, context, continuation):
    n = len(context) - len(context.rstrip())
    if n:
        continuation = context[-n:] + continuation
        context = context[:-n]
    prefix = tokenizer.encode(context, add_special_tokens=False)
    both = tokenizer.encode(context + continuation, add_special_tokens=False)
    if not prefix or both[: len(prefix)] != prefix or len(both) == len(prefix):
        raise ValueError("ambiguous continuation token boundary")
    return prefix, both[len(prefix) :]


def batch_jobs(jobs, max_pairs, max_tokens):
    batch = []
    length = 0
    for job in sorted(jobs, key=lambda job: len(job["input_ids"])):
        trial = max(length, len(job["input_ids"]))
        if batch and (len(batch) == max_pairs or 2 * (len(batch) + 1) * trial > max_tokens):
            yield batch
            batch = []
            length = 0
        if 2 * len(job["input_ids"]) > max_tokens:
            raise ValueError("untruncated example exceeds batch token budget")
        batch.append(job)
        length = max(length, len(job["input_ids"]))
    if batch:
        yield batch


def qualify(engine, pilot, reference_receipts, output, contexts):
    import torch

    from archlab.artifacts import atomic_write_json

    report = {
        "checks": [],
        "passed": False,
        "metric": "absolute-CE-delta-to-archived-released-BF16-reference",
        "limit": 0.05,
    }
    for context in contexts:
        count = 8 if context <= 2048 else 2
        batches = [
            pilot.batch(rank, device="cpu", smoke_context=context, pad_to_full=True)
            for rank in range(count)
        ]
        ids = torch.cat([b[0] for b in batches])
        labels = torch.cat([b[1] for b in batches]).cuda()
        start = time.monotonic()
        _, hidden = engine.forward(ids, adapted=False, return_hidden=True)
        checks = []
        for rank in range(count):
            totals = torch.zeros(2, device="cuda", dtype=torch.float64)
            for first in range(0, context, 128):
                logits = engine.model.head(
                    hidden[rank : rank + 1, first : first + 128], full_logits=True
                )
                target = labels[rank : rank + 1, first : first + 128]
                valid = target != -100
                totals[0] += torch.nn.functional.cross_entropy(
                    logits[valid], target[valid], reduction="sum"
                )
                totals[1] += valid.sum()
            reference = json.loads((reference_receipts / f"rank{rank}.json").read_text())
            expected = next(
                row["reference_loss"] for row in reference["tests"] if row["context"] == context
            )
            ce = float(totals[0] / totals[1])
            delta = ce - expected
            checks.append(
                {
                    "rank_window": rank,
                    "loss": ce,
                    "reference_loss": expected,
                    "delta_loss": delta,
                    "passed": abs(delta) < 0.05,
                }
            )
        report["checks"].append(
            {"context": context, "seconds": time.monotonic() - start, "windows": checks}
        )
        atomic_write_json(output / "FORWARD_QUALIFICATION.json", report, allow_nan=False)
        print(
            json.dumps(
                {
                    "event": "eval_forward_qualification",
                    "context": context,
                    "seconds": time.monotonic() - start,
                    "max_abs_ce_delta": max(abs(r["delta_loss"]) for r in checks),
                }
            ),
            flush=True,
        )
        if not all(row["passed"] for row in checks):
            raise ValueError("local inference failed archived-reference CE qualification")
    report["passed"] = True
    atomic_write_json(output / "FORWARD_QUALIFICATION.json", report, allow_nan=False)


def multiple_choice(engine, cases, templates, output, config):
    import torch
    from jinja2 import Environment, StrictUndefined

    from archlab.artifacts import atomic_write_json
    from archlab.evaluation.capability import paired_statistics

    env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)
    jobs = []
    records = {}
    for row in cases:
        template = templates["multiple_choice"][
            "mmlu" if row["task"] == "mmlu" else "arc_challenge"
        ]
        context = env.from_string(template).render(**row)
        choices = (
            [chr(65 + i) for i in range(len(row["choices"]))]
            if row["task"] == "mmlu"
            else row["choices"]
        )
        pairs = [encode_pair(engine.tokenizer, context, " " + choice) for choice in choices]
        record = {
            **row,
            "choice_lengths": [len(choice) for choice in choices],
            "base_scores": [None] * len(choices),
            "adapted_scores": [None] * len(choices),
        }
        records[row["id"]] = record
        if all(len(target) == 1 and prefix == pairs[0][0] for prefix, target in pairs):
            jobs.append(
                {
                    "id": row["id"],
                    "input_ids": pairs[0][0],
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
                        "input_ids": prefix + target[:-1],
                        "prefix_length": len(prefix),
                        "target": target,
                    }
                )
    written = set()
    for number, batch in enumerate(
        batch_jobs(jobs, config["max_batch_size"] // 2, config["max_padded_tokens"]), 1
    ):
        n = len(batch)
        length = max(len(job["input_ids"]) for job in batch)
        if length > config["context"]:
            raise ValueError("benchmark prompt exceeds context; no truncation permitted")
        ids = torch.zeros(2 * n, length, dtype=torch.long)
        for index, job in enumerate(batch):
            seq = torch.tensor(job["input_ids"])
            ids[index, : len(seq)] = seq
            ids[index + n, : len(seq)] = seq
        start = time.monotonic()
        _, hidden = engine.forward(ids, adapted=[False] * n + [True] * n, return_hidden=True)
        if number == 1 and torch.equal(hidden[:n], hidden[n:]):
            raise RuntimeError("trained adapters did not change any hidden state")
        for index, job in enumerate(batch):
            record = records[job["id"]]
            for row_index, mode in ((index, "base"), (index + n, "adapted")):
                if "fast_targets" in job:
                    logits = engine.model.head(
                        hidden[
                            row_index : row_index + 1,
                            job["prefix_length"] - 1 : job["prefix_length"],
                        ],
                        full_logits=True,
                    )[0, 0]
                    record[mode + "_scores"] = logits.log_softmax(-1)[job["fast_targets"]].tolist()
                else:
                    score = 0.0
                    target = torch.tensor(job["target"], device="cuda")
                    first = job["prefix_length"] - 1
                    for offset in range(0, len(target), 64):
                        chunk = target[offset : offset + 64]
                        logits = engine.model.head(
                            hidden[
                                row_index : row_index + 1,
                                first + offset : first + offset + len(chunk),
                            ],
                            full_logits=True,
                        )[0]
                        score += float(logits.log_softmax(-1).gather(1, chunk[:, None]).sum())
                    record[mode + "_scores"][job["choice"]] = score
            if (
                None not in record["base_scores"]
                and None not in record["adapted_scores"]
                and job["id"] not in written
            ):
                for mode in ("base", "adapted"):
                    scores = record[mode + "_scores"]
                    normal = [s / k for s, k in zip(scores, record["choice_lengths"], strict=True)]
                    if not all(math.isfinite(s) for s in scores):
                        raise FloatingPointError("nonfinite multiple-choice score")
                    pred = max(range(len(scores)), key=scores.__getitem__)
                    norm = max(range(len(normal)), key=normal.__getitem__)
                    record[mode] = {
                        "prediction": pred,
                        "normalized_prediction": norm,
                        "acc": int(pred == record["answer"]),
                        "acc_norm": int(norm == record["answer"]),
                    }
                append(output / "multiple-choice-pairs.jsonl", record)
                written.add(job["id"])
        event = {
            "event": "multiple_choice_batch",
            "batch": number,
            "jobs": n,
            "context": length,
            "completed_pairs": len(written),
            "planned_pairs": len(cases),
            "seconds": time.monotonic() - start,
            "cache_hits": engine.cache.hits,
            "cache_misses": engine.cache.misses,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        append(output / "batches.jsonl", event)
        print(json.dumps(event), flush=True)
        del hidden, ids
    if len(written) != len(cases):
        raise RuntimeError("incomplete multiple-choice evaluation")
    summary = {}
    for task in sorted({row["task"] for row in cases}):
        rows = [r for r in records.values() if r["task"] == task]
        summary[task] = {
            metric: paired_statistics(
                [r["base"][metric] for r in rows], [r["adapted"][metric] for r in rows]
            )
            for metric in ("acc", "acc_norm")
        }
    atomic_write_json(output / "MULTIPLE_CHOICE.json", summary, allow_nan=False)
    return summary


def math_metrics(engine, pilot, selected, output, config):
    import torch

    from archlab.artifacts import atomic_write_json

    jobs = []
    for item in selected:
        ids, labels, count = pilot.batch(item["pilot_index"], device="cpu")
        if count != item["targets"]:
            raise ValueError("held-out target count changed")
        jobs.append({"input_ids": ids[0].tolist(), "labels": labels[0], "record": item})
    records = []
    for number, batch in enumerate(
        batch_jobs(jobs, config["max_batch_size"] // 2, config["max_padded_tokens"]), 1
    ):
        n = len(batch)
        length = max(len(job["input_ids"]) for job in batch)
        ids = torch.zeros(2 * n, length, dtype=torch.long)
        for index, job in enumerate(batch):
            seq = torch.tensor(job["input_ids"])
            ids[index, : len(seq)] = seq
            ids[index + n, : len(seq)] = seq
        start = time.monotonic()
        _, hidden = engine.forward(ids, adapted=[False] * n + [True] * n, return_hidden=True)
        if number == 1 and torch.equal(hidden[:n], hidden[n:]):
            raise RuntimeError("trained adapters did not change any hidden state")
        for index, job in enumerate(batch):
            labels = job["labels"].cuda()
            positions = torch.where(labels != -100)[0]
            sums = {mode: {"nll": 0.0, "top1": 0, "top5": 0} for mode in ("base", "adapted")}
            kl = 0.0
            agreement = 0
            for offset in range(0, len(positions), 64):
                selected_positions = positions[offset : offset + 64]
                targets = labels[selected_positions]
                probabilities = {}
                predictions = {}
                for row_index, mode in ((index, "base"), (index + n, "adapted")):
                    logits = engine.model.head(
                        hidden[row_index, selected_positions].unsqueeze(0), full_logits=True
                    )[0]
                    if not bool(logits.isfinite().all()):
                        raise FloatingPointError("nonfinite math logits")
                    logp = logits.log_softmax(-1)
                    probabilities[mode] = logp
                    sums[mode]["nll"] += float(
                        -logp.gather(1, targets[:, None]).sum(dtype=torch.float64)
                    )
                    predictions[mode] = logits.argmax(-1)
                    sums[mode]["top1"] += int((predictions[mode] == targets).sum())
                    sums[mode]["top5"] += int(
                        (logits.topk(5, dim=-1).indices == targets[:, None]).any(-1).sum()
                    )
                kl += float(
                    (
                        probabilities["base"].exp()
                        * (probabilities["base"] - probabilities["adapted"])
                    ).sum(dtype=torch.float64)
                )
                agreement += int((predictions["base"] == predictions["adapted"]).sum())
            item = job["record"]
            row = {
                "pilot_index": item["pilot_index"],
                "mode": item["mode"],
                "has_tools": item["has_tools"],
                "problem_sha256": item["problem_sha256"],
                "source_length": item["length"],
                "targets": len(positions),
                "base": sums["base"],
                "adapted": sums["adapted"],
                "base_to_adapted_kl_sum": kl,
                "argmax_agreement_count": agreement,
            }
            if len(positions) != item["targets"]:
                raise ValueError("incorrect math target mask")
            append(output / "math-pairs.jsonl", row)
            records.append(row)
        event = {
            "event": "math_batch",
            "batch": number,
            "windows": n,
            "context": length,
            "completed_windows": len(records),
            "planned_windows": len(selected),
            "seconds": time.monotonic() - start,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        append(output / "batches.jsonl", event)
        print(json.dumps(event), flush=True)
        del hidden, ids

    def aggregate(rows):
        total = sum(row["targets"] for row in rows)
        result = {"windows": len(rows), "supervised_targets": total}
        for mode in ("base", "adapted"):
            nll = sum(row[mode]["nll"] for row in rows) / total
            result[mode] = {
                "cross_entropy": nll,
                "perplexity": math.exp(nll),
                "top1_token_accuracy": sum(row[mode]["top1"] for row in rows) / total,
                "top5_token_accuracy": sum(row[mode]["top5"] for row in rows) / total,
            }
        result["base_to_adapted_kl"] = sum(row["base_to_adapted_kl_sum"] for row in rows) / total
        result["token_argmax_agreement"] = (
            sum(row["argmax_agreement_count"] for row in rows) / total
        )
        return result

    summary = {
        "overall": aggregate(records),
        "by_difficulty": {
            mode: aggregate([row for row in records if row["mode"] == mode])
            for mode in ("low", "medium", "high")
        },
    }
    atomic_write_json(output / "MATH_METRICS.json", summary, allow_nan=False)
    return summary


def resolve(value):
    if value.startswith("env:"):
        return Path(os.environ[value[4:]]).resolve(strict=True)
    if value.startswith("package:"):
        return Path(__file__).resolve().parents[1] / value[8:]
    raise ValueError("recipe paths must be launch-injected or package-relative")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-receipts", type=Path, required=True)
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(Path("/usr/local/lib/python3.12/dist-packages"))
    import socket
    import traceback

    import torch
    import yaml

    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_data import MathPilot
    from archlab.automodel.deepseek_v41_local_inference import LocalV41Inference

    config = yaml.safe_load(args.recipe.read_text())
    torch.set_grad_enabled(False)
    assets, weights, checkpoint = (
        resolve(config[key]) for key in ("assets", "weights", "adapter_checkpoint")
    )
    marker = json.loads((checkpoint / "COMPLETE.json").read_text())
    contract = marker["contract"]
    for file, key in (
        (weights / "model.safetensors.index.json", "checkpoint_index_sha256"),
        (weights / "config.json", "base_config_sha256"),
        (assets / "inference/config.json", "reference_config_sha256"),
    ):
        if sha(file) != contract[key]:
            raise ValueError(f"wrong base identity: {key}")
    root = Path(__file__).resolve().parents[1]
    source_names = [
        "architectures/deepseek_v41_adapter.py",
        "architectures/simplicial_attention.py",
        "architectures/simplicial_kernels.py",
        "architectures/simplicial_deterministic.py",
        "architectures/ordered_reduction.py",
        "architectures/deepseek_v41_math.py",
    ]
    for name in source_names:
        if sha(root / name) != contract["implementation_sha256"][name]:
            raise ValueError(f"changed trained adapter implementation: {name}")
    manifest = json.loads((args.data / "MANIFEST.json").read_text())
    if (
        sha(args.data / "cases.jsonl") != manifest["cases_sha256"]
        or sha(args.data / "math_windows.json") != manifest["math_windows_sha256"]
    ):
        raise ValueError("sealed evaluation data changed")
    cases = [
        json.loads(line)
        for line in (args.data / "cases.jsonl").read_text().split("\n")
        if line.strip()
    ]
    selected = json.loads((args.data / "math_windows.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    source_names += [
        "automodel/deepseek_v41_local_inference.py",
        "automodel/deepseek_v41_cpu_store.py",
        "evaluation/deepseek_v41_local_eval.py",
        "evaluation/deepseek_v41_local_data.py",
    ]
    provenance = {
        "recipe_sha256": sha(args.recipe),
        "data_manifest_sha256": sha(args.data / "MANIFEST.json"),
        "checkpoint_sha256": marker["state_sha256"],
        "checkpoint_cursor": marker["cursor"],
        "hostname": socket.gethostname(),
        "source_sha256": {name: sha(root / name) for name in source_names},
        "config": config,
        "prompts_sha256": sha(resolve(config["prompts"])),
    }
    atomic_write_json(args.output / "RUN.json", provenance, allow_nan=False)
    try:
        engine = LocalV41Inference(
            assets=assets,
            weights=weights,
            adapter_checkpoint=checkpoint,
            context=config["context"],
            max_batch_size=config["max_batch_size"],
            expert_cache_gib=config["expert_cache_gib"],
            resident_cpu_experts=True,
        )
        atomic_write_json(args.output / "LOADING.json", engine.report, allow_nan=False)
        train = MathPilot(
            resolve(config["train_pilot"]), expected_split="train", expected_budget=1000000000
        )
        qualify(
            engine,
            train,
            args.reference_receipts,
            args.output,
            config["protocol"]["prefill_parity_contexts"],
        )
        del train
        templates = yaml.safe_load(resolve(config["prompts"]).read_text())
        mc = multiple_choice(engine, cases, templates, args.output, config)
        validation = MathPilot(
            resolve(config["validation_pilot"]),
            expected_split="validation",
            expected_budget=1000000,
        )
        math_results = math_metrics(engine, validation, selected, args.output, config)
        if math_results["overall"]["supervised_targets"] != manifest["math_supervised_targets"]:
            raise ValueError("math evaluation budget mismatch")
        summary = {
            "passed": True,
            "complete": True,
            "pilot_not_full_benchmark": True,
            "checkpoint_cursor": marker["cursor"],
            "multiple_choice": mc,
            "heldout_math": math_results,
            "loading": engine.report,
            "exact_overlap_consumed_training": manifest["exact_consumed_training_overlaps"],
        }
        atomic_write_json(args.output / "COMPLETE.json", summary, allow_nan=False)
        print(json.dumps({"event": "local_evaluation_complete", **summary}), flush=True)
    except BaseException:
        atomic_write_json(args.output / "FAILED.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
