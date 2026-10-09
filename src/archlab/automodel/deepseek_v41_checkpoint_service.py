"""Keep a pinned baseline on GPUs for paired evaluation and persistent team chat.

This entry point has no optimizer and never invokes a training update. It uses
the qualified 16-rank checkpoint restore and resident evaluation/chat routines.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    args = parser.parse_args()

    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(Path(os.environ["ARCHLAB_CONTAINER_KERNEL_PACKAGES"]))
    import torch
    import torch.distributed as dist

    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_control import distributed_policy, pin_checkpoint
    from archlab.automodel.deepseek_v41_data import MathPilot
    from archlab.automodel.deepseek_v41_full_validation import evaluate_pilot, make_plan
    from archlab.automodel.deepseek_v41_live_window import Activity, construct_secondary, run_window
    from archlab.automodel.deepseek_v41_official_execution import configure_official_reproducibility
    from archlab.serving.openai_chat import ChatFront

    source = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        commit != os.environ["NGA_EXPECTED_COMMIT"]
        or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ).strip()
    ):
        raise ValueError("Use the recorded immutable serving source.")
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
    front = None
    activity = None
    lease = args.checkpoint / ".checkpoint-readers/persistent-baseline-service.json"
    try:
        if dist.get_world_size() != 16:
            raise ValueError("The full checkpoint service requires the trained 16-rank mesh.")
        policy = distributed_policy(args.policy)
        if not policy.get("persistent_chat"):
            raise ValueError("Persistent serving must be explicitly selected.")
        marker = json.loads((args.checkpoint / "COMPLETE.json").read_text())
        cursor = marker["cursor"]
        if marker["contract"]["variant"] != "normal" or cursor["step"] not in policy["pin_steps"]:
            raise ValueError("The baseline must be the explicitly pinned normal checkpoint.")
        args.output.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            pin_checkpoint(args.checkpoint, cursor, policy)
            atomic_write_json(
                lease, {"owner": "persistent-baseline-service", "output": str(args.output)}
            )
            atomic_write_json(
                args.output / "RUN.json",
                {
                    "project_commit": commit,
                    "checkpoint": str(args.checkpoint),
                    "cursor": cursor,
                    "optimizer_created": False,
                    "training_updates_enabled": False,
                    "persistent_chat": True,
                    "cpu_weight_offload": False,
                    "service_source_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                },
            )
            front = ChatFront(
                policy["chat_host"],
                policy["chat_port"],
                policy["chat_token_file"],
                max_tokens=policy["chat_max_tokens"],
            )
            front.set_state("loading")
            activity = Activity(args.baseline_output, cursor)
            activity.phase = "restoring"
            activity.stage = "frozen-baseline"
            activity.__enter__()
        registry = json.loads(Path(policy["source_registry"]).read_text())
        assets = Path(os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"])
        weights = Path(os.environ["ARCHLAB_DEEPSEEK_V41_CHECKPOINT"])
        model, setup, receipt = construct_secondary(
            args.checkpoint, "normal", weights=weights, assets=assets, registry=registry
        )
        atomic_write_json(args.output / f"normal-rank-{rank:02d}-restore.json", receipt)
        validation = MathPilot(
            Path(policy["validation_pilot"]), expected_split="validation", expected_budget=1000000
        )
        plan = make_plan(validation, 64000)
        metric = evaluate_pilot(model, validation, plan)
        reference = [
            json.loads(line)
            for line in (args.baseline_output / "validation.jsonl").read_text().splitlines()
            if line.strip() and json.loads(line)["step"] == cursor["step"]
        ][-1]
        if (
            reference["plan_sha256"] != metric["plan_sha256"]
            or abs(reference["loss"] - metric["loss"]) > 1e-4
        ):
            raise ValueError("Frozen baseline differs from the saved resident training forward.")
        if rank == 0:
            atomic_write_json(
                args.output / "FROZEN_BASELINE_QUALIFICATION.json",
                {
                    "passed": True,
                    "cursor": cursor,
                    "validation_ce": metric["loss"],
                    "resident_training_ce": reference["loss"],
                    "ce_error": metric["loss"] - reference["loss"],
                    "targets": metric["targets"],
                    "plan_sha256": metric["plan_sha256"],
                },
            )
            activity.__exit__(None, None, None)
            activity = None
        batches = [validation.batch(rank, device="cuda")]
        while True:
            policy = distributed_policy(args.policy)
            stop = torch.tensor(
                int((args.baseline_output / "STOP_SERVING").exists()), device="cuda"
            )
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if bool(stop):
                break
            if front:
                front.set_state("evaluating")
            # nn.Module.zero_grad has the same buffer-clearing interface needed
            # by run_window. No optimizer exists in this serving process.
            run_window(
                model,
                model,
                cursor,
                args.checkpoint,
                policy,
                assets=assets,
                weights=weights,
                front=front,
                output=args.baseline_output,
                verification_batches=batches,
                final=True,
            )
        # Keep the distributed setup alive for the entire model lifetime.
        del model, setup
    except BaseException:
        atomic_write_json(
            args.output / f"rank-{rank:02d}-failure.json", {"traceback": traceback.format_exc()}
        )
        raise
    finally:
        if activity:
            activity.__exit__(None, None, None)
        if front:
            front.close()
        dist.destroy_process_group()
        if rank == 0:
            lease.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
