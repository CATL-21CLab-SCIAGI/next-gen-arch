"""Screen the released adapter-free model against a fixed subset of saved SFT evaluations.

The evaluator uses the existing 32 GPUs with a separate, bounded allocator. It
waits for both restarted actors' real-update receipts before touching CUDA.
Inference uses the original released checkpoint and the established likelihood
scorer; recorded normal/simplicial scores refer to their matched SFT parents.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import subprocess
import time
import traceback
from collections import Counter
from pathlib import Path

from archlab.artifacts import atomic_write_json, sha256_file


def read_reference(path):
    # JSON strings may contain Unicode line separators. File iteration splits
    # physical JSONL records without treating those characters as new records.
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_cases(records, *, seed=20260923, mmlu_per_subject=2, arc_count=32, piqa_count=64):
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("reference evaluation has duplicate case IDs")
    available = [
        row
        for row in records
        if row.get("exact_overlap_consumed_training") is False
        and row.get("exact_overlap_planned_training") is False
    ]
    selected = []
    subjects = sorted({row["subject"] for row in available if row["task"] == "mmlu"})
    if len(subjects) != 57:
        raise ValueError("the existing lightweight evaluation must cover all 57 MMLU subjects")
    for subject in subjects:
        group = sorted(
            (row for row in available if row["task"] == "mmlu" and row["subject"] == subject),
            key=lambda r: r["id"],
        )
        selected.extend(random.Random(f"{seed}:mmlu:{subject}").sample(group, mmlu_per_subject))
    for task, count in (("arc_challenge", arc_count), ("piqa", piqa_count)):
        group = sorted((row for row in available if row["task"] == task), key=lambda r: r["id"])
        selected.extend(random.Random(f"{seed}:{task}").sample(group, count))
    return sorted(selected, key=lambda row: row["id"])


def summarize(records):
    from archlab.evaluation.capability import paired_statistics

    result = {}
    for task in sorted({row["task"] for row in records}):
        rows = [row for row in records if row["task"] == task]
        result[task] = {}
        for variant in ("normal", "simplicial"):
            result[task][variant] = {}
            for metric in ("accuracy", "accuracy_norm"):
                comparison = paired_statistics(
                    [row["original"][metric] for row in rows],
                    [row["result"][variant][metric] for row in rows],
                )
                comparison["significant_regression"] = comparison["delta_95pct_interval_pp"][1] < 0
                comparison["delta_definition"] = "saved step-4537 SFT model minus released original"
                result[task][variant][metric] = comparison
    return result


def training_ready(paths):
    receipts = []
    for path in paths:
        root = Path(path)
        if list(root.glob("rank-*-failure.json")) or (root / "STOPPED.json").exists():
            raise RuntimeError(f"training is failed or stopped: {root.name}")
        required = [
            root / name
            for name in (
                "TRAINING_ADMITTED.json",
                "REAL_UPDATE_VERIFIED.json",
                "MEMORY_QUALIFICATION.json",
            )
        ]
        if not all(p.exists() for p in required):
            return None
        admitted, real, memory = (json.loads(p.read_text()) for p in required)
        if not all(value.get("passed") is True for value in (admitted, real, memory)):
            raise ValueError("training admission, real update, and memory receipts must all pass")
        if memory.get("minimum_driver_free_gib", 0) < 64:
            raise ValueError("training has not demonstrated the shared evaluation reserve")
        receipts.append(
            {
                "path": str(root),
                "optimizer_step": real["optimizer_step"],
                "contract_digest": admitted["contract_digest"],
                "memory_qualification_sha256": sha256_file(required[2]),
            }
        )
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, action="append", required=True)
    parser.add_argument("--container-kernel-packages", type=Path, required=True)
    parser.add_argument("--memory-budget-gib", type=float, default=56.0)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--mmlu-per-subject", type=int, default=2)
    parser.add_argument("--arc-count", type=int, default=32)
    parser.add_argument("--piqa-count", type=int, default=64)
    parser.add_argument("--wait-hours", type=float, default=12.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    if len(args.training_run) != 2 or not 0 < args.memory_budget_gib <= 56 or args.wait_hours <= 0:
        parser.error(
            "provide both actors, a positive evaluator budget <=56 GiB, and a positive wait limit"
        )
    if min(args.mmlu_per_subject, args.arc_count, args.piqa_count) < 1:
        parser.error("lightweight subset sizes must be positive")
    records = read_reference(args.reference)
    reference_run = args.reference.parent / "RUN.json"
    recorded = json.loads(reference_run.read_text())
    complete_path = args.reference.parent / "COMPLETE.json"
    complete = json.loads(complete_path.read_text()) if complete_path.exists() else {}
    if (
        complete.get("passed") is not True
        or recorded.get("matched_steps") is not True
        or any(
            recorded.get("cursors", {}).get(v, {}).get("step") != 4537
            for v in ("normal", "simplicial")
        )
    ):
        raise ValueError(
            "reference predictions must belong to the complete matched step-4537 evaluation"
        )
    original_config = json.loads((args.weights / "config.json").read_text())
    if original_config.get("archlab") or (args.weights / "DIRECT_CHECKPOINT.json").exists():
        raise ValueError(
            "baseline requires the released original checkpoint, not a fine-tuned export"
        )
    selected = select_cases(
        records,
        seed=args.seed,
        mmlu_per_subject=args.mmlu_per_subject,
        arc_count=args.arc_count,
        piqa_count=args.piqa_count,
    )
    prompts = Path(__file__).resolve().parents[1] / "prompts/capability_regression.yaml"
    provenance = {
        "reference": str(args.reference),
        "reference_sha256": sha256_file(args.reference),
        "reference_run_sha256": sha256_file(reference_run),
        "reference_checkpoints": recorded["checkpoint_paths"],
        "prompts_sha256": sha256_file(prompts),
        "seed": args.seed,
        "case_ids": [row["id"] for row in selected],
        "counts": dict(Counter(row["task"] for row in selected)),
        "selection_uses_scores": False,
        "comparison_checkpoint_step": 4537,
        "scoring_mode": "existing-choice-likelihood-no-generation",
        "reference_scope": "matched adapted SFT parents; not a new RL checkpoint",
        "original_weights": str(args.weights),
        "original_config_sha256": sha256_file(args.weights / "config.json"),
        "original_index_sha256": sha256_file(args.weights / "model.safetensors.index.json"),
    }
    import yaml
    from transformers import PreTrainedTokenizerFast

    from archlab.evaluation.deepseek_v41_compare_data import build_jobs, jobs_digest

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.assets, local_files_only=True)
    jobs = build_jobs(tokenizer, selected, yaml.safe_load(prompts.read_text()))
    longest = max(len(row["input_ids"]) for row in jobs)
    if longest > args.max_input_tokens:
        raise ValueError("selected lightweight cases exceed the untruncated input budget")
    provenance.update(
        jobs_digest=jobs_digest(jobs), inference_jobs=len(jobs), maximum_input_tokens=longest
    )
    if args.preflight_only:
        print(json.dumps(provenance, indent=2))
        return
    deadline = time.monotonic() + args.wait_hours * 3600
    while (admission := training_ready(args.training_run)) is None:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "training did not reach the required real-update and memory admission"
            )
        time.sleep(10)
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    packages = select_container_kernel_packages(args.container_kernel_packages)
    import torch
    import torch.distributed as dist

    from archlab.automodel.deepseek_v41_full_evaluate import score_job
    from archlab.automodel.deepseek_v41_live_window import inference_head
    from archlab.automodel.deepseek_v41_official_execution import build_official_base
    from archlab.automodel.deepseek_v41_rl_memory import configure_gpu_budget
    from archlab.automodel.deepseek_v41_rl_training import Activity
    from archlab.evaluation.deepseek_v41_compare_metrics import score_choice_vector

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    budget = configure_gpu_budget(args.memory_budget_gib)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=90),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    activity = None
    owned = False
    try:
        if world != 32:
            raise ValueError("shared original-model evaluation uses all existing 32 ranks with EP8")
        error = [None]
        if rank == 0:
            try:
                args.output.mkdir(parents=True, exist_ok=False)
            except OSError as caught:
                error[0] = str(caught)
        dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise FileExistsError(error[0])
        owned = True
        if rank == 0:
            activity = Activity(args.output)
            activity.set("loading_original", tiny=args.tiny)
        model, setup, loading = build_official_base(
            weights=args.weights,
            assets=args.assets,
            ep_size=8,
            activation_checkpointing=False,
            tiny=args.tiny,
        )
        model.eval()
        if any(
            "simplicial_adapter" in name or p.requires_grad for name, p in model.named_parameters()
        ):
            raise ValueError("original baseline must have no adapter and no trainable parameter")
        provenance.update(
            runtime=loading,
            resolved_kernel_packages=packages,
            allocator=budget,
            project_commit=subprocess.check_output(
                ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            training_admission=admission,
            adapter_present=False,
            optimizer_created=False,
            tiny=args.tiny,
            shared_gpu=True,
        )
        if rank == 0:
            atomic_write_json(args.output / "RUN.json", provenance, allow_nan=False)
        scores = {row["id"]: [None] * len(row["choices"]) for row in selected}
        lookup = {row["id"]: row for row in selected}
        completed = {}
        rounds = 1 if args.tiny else (len(jobs) + world - 1) // world
        for round_index in range(rounds):
            ready = [None]
            if rank == 0:
                try:
                    ready[0] = {"passed": training_ready(args.training_run) is not None}
                except (ValueError, RuntimeError) as caught:
                    ready[0] = {"passed": False, "reason": str(caught)}
            dist.broadcast_object_list(ready, src=0)
            if not ready[0]["passed"]:
                raise RuntimeError(f"training admission lost: {ready[0]}")
            index = round_index * world + rank
            real = index < len(jobs)
            job = jobs[index] if real else jobs[round_index * world]
            value = score_job(model, job, head_context=inference_head)
            packets = [None] * world
            dist.all_gather_object(packets, {"job": job, "score": value} if real else None)
            if rank == 0:
                for packet in packets:
                    if packet is None:
                        continue
                    row, result = packet["job"], packet["score"]
                    if "fast_targets" in row:
                        scores[row["id"]] = result
                    else:
                        scores[row["id"]][row["choice"]] = result
                    if row["id"] not in completed and None not in scores[row["id"]]:
                        case = lookup[row["id"]]
                        completed[row["id"]] = {
                            **case,
                            "original_scores": scores[row["id"]],
                            "original": score_choice_vector(case, scores[row["id"]]),
                        }
                activity.set(
                    "scoring",
                    rounds_done=round_index + 1,
                    rounds_total=rounds,
                    cases_complete=len(completed),
                    cases_total=len(selected),
                )
        peak = {
            "rank": rank,
            "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        peaks = [None] * world
        dist.all_gather_object(peaks, peak)
        if rank == 0:
            if not args.tiny and len(completed) != len(selected):
                raise ValueError("lightweight original-model evaluation is incomplete")
            if not args.tiny:
                ordered = [completed[row["id"]] for row in selected]
                (args.output / "predictions.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered)
                )
                atomic_write_json(
                    args.output / "COMPARISON.json", summarize(ordered), allow_nan=False
                )
            atomic_write_json(
                args.output / "COMPLETE.json",
                {
                    "passed": True,
                    "tiny": args.tiny,
                    "cases": len(completed),
                    "optimizer_updates": 0,
                    "memory": peaks,
                    "scope": "small paired regression screen; not a proof of general-task equivalence",
                },
                allow_nan=False,
            )
            activity.set("complete", tiny=args.tiny)
        del model, setup
    except BaseException:
        if owned:
            atomic_write_json(
                args.output / f"rank-{rank:02d}-failure.json", {"traceback": traceback.format_exc()}
            )
        raise
    finally:
        if activity:
            activity.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
