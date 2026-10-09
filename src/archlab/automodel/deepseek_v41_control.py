"""Optimizer-boundary checkpoint and evaluation policy for the full comparison."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

UPDATE_SOURCE_SHA256 = "69b5c11e0a20714a4eeb3d6411228058bf000c8a233f9b4143389aa2ba722b65"
CONTROL_IMPLEMENTATIONS = frozenset(
    {
        "automodel/deepseek_v41_full_training.py",
        "automodel/deepseek_v41_full_eval_checkpoint.py",
        "automodel/deepseek_v41_full_eval_construct.py",
        "automodel/deepseek_v41_full_evaluate.py",
        "automodel/deepseek_v41_control.py",
        "automodel/deepseek_v41_live_window.py",
        "serving/openai_chat.py",
        "evaluation/deepseek_v41_live_metrics.py",
    }
)


def read_policy(path):
    policy = json.loads(Path(path).read_text())
    if policy.get("format") != "archlab-full-eval-policy-v1":
        raise ValueError("unknown full-training control policy")
    for field in ("checkpoint_interval_steps", "evaluation_interval_steps", "chat_window_seconds"):
        if type(policy.get(field)) is not int or policy[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if policy["checkpoint_interval_steps"] != policy["evaluation_interval_steps"]:
        raise ValueError("scheduled evaluations require a checkpoint at the same step")
    if any(type(step) is not int or step <= 0 for step in policy.get("pin_steps", [])):
        raise ValueError("pinned optimizer steps must be positive integers")
    return policy


def distributed_policy(path):
    import torch.distributed as dist

    packet = [None]
    if dist.get_rank() == 0:
        try:
            packet[0] = {"policy": read_policy(path)}
        except Exception as error:
            packet[0] = {"error": str(error)}
    dist.broadcast_object_list(packet, src=0)
    if "error" in packet[0]:
        raise ValueError(packet[0]["error"])
    return packet[0]["policy"]


def pending_evaluation(policy, variant):
    for path in sorted(Path(policy["evaluation_requests"]).glob(f"{variant}-step-*.json")):
        request = json.loads(path.read_text())
        result = Path(policy["benchmark_results"]) / path.name
        if not result.exists():
            return request
    return None


def checkpoint_due(step, policy):
    return step > 0 and (
        step % policy["checkpoint_interval_steps"] == 0 or step in policy.get("pin_steps", [])
    )


def evaluation_due(step, policy):
    return step > 0 and step % policy["evaluation_interval_steps"] == 0


def verify_pinned_cursor(cursor, policy):
    expected = policy.get("pinned_tokens", {}).get(str(cursor["step"]))
    if expected is not None and cursor["supervised_tokens"] != expected:
        raise ValueError("pinned checkpoint differs from the matched data cursor")


def pin_checkpoint(checkpoint, cursor, policy):
    from archlab.artifacts import atomic_write_json

    checkpoint = Path(checkpoint)
    if cursor["step"] not in policy.get("pin_steps", []):
        return False
    verify_pinned_cursor(cursor, policy)
    marker = json.loads((checkpoint / "COMPLETE.json").read_text())
    if marker["cursor"] != cursor:
        raise ValueError("cannot pin an incomplete or mismatched checkpoint")
    atomic_write_json(
        checkpoint / "PINNED.json",
        {
            "format": "archlab-checkpoint-pin-v1",
            "reason": "matched attention comparison",
            "cursor": cursor,
            "variant": marker["contract"]["variant"],
            "automatic_retention": False,
        },
    )
    return True


def publish_evaluation_request(root, checkpoint, cursor, variant):
    from archlab.artifacts import atomic_write_json

    root, checkpoint = Path(root), Path(checkpoint)
    marker = json.loads((checkpoint / "COMPLETE.json").read_text())
    if marker["cursor"] != cursor or marker["contract"]["variant"] != variant:
        raise ValueError("evaluation request requires a complete matching checkpoint")
    path = root / f"{variant}-step-{cursor['step']:06d}.json"
    value = {
        "format": "archlab-checkpoint-evaluation-request-v1",
        "variant": variant,
        "checkpoint": str(checkpoint.resolve()),
        "cursor": cursor,
        "checkpoint_marker_sha256": hashlib.sha256(
            (checkpoint / "COMPLETE.json").read_bytes()
        ).hexdigest(),
    }
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("a scheduled evaluation checkpoint was replaced")
    else:
        atomic_write_json(checkpoint / "EVAL_PENDING.json", {"request": str(path.resolve())})
        atomic_write_json(path, value)
    return path


def admit_control_upgrade(saved, current, update_function):
    """Permit scheduling/serving changes only, with identical training math."""
    if saved == current:
        return
    operational = {"project_commit", "implementation_sha256", "control_protocol"}
    if {k: v for k, v in saved.items() if k not in operational} != {
        k: v for k, v in current.items() if k not in operational
    }:
        raise ValueError("control upgrade changed the model, optimizer, mesh, or runtime")
    old, new = saved["implementation_sha256"], current["implementation_sha256"]
    protected = (set(old) | set(new)) - CONTROL_IMPLEMENTATIONS
    if any(old.get(name) != new.get(name) for name in protected):
        raise ValueError("control upgrade changed a protected implementation")
    if (
        hashlib.sha256(inspect.getsource(update_function).encode()).hexdigest()
        != UPDATE_SOURCE_SHA256
    ):
        raise ValueError("control upgrade changed the qualified optimizer update")


def verify_checkpoint_sources(contract, current_source, trained_source):
    """Check full archived provenance and all forward implementations in use."""
    import subprocess

    current_source, trained_source = Path(current_source), Path(trained_source)
    if (
        subprocess.check_output(
            ["git", "-C", str(trained_source), "rev-parse", "HEAD"], text=True
        ).strip()
        != contract["project_commit"]
    ):
        raise ValueError("wrong immutable trained source checkout")
    if subprocess.check_output(
        ["git", "-C", str(trained_source), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("trained source checkout is no longer immutable")
    for relative, digest in contract["implementation_sha256"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("invalid source identity path")
        recorded = trained_source / "src/archlab" / relative
        if hashlib.sha256(recorded.read_bytes()).hexdigest() != digest:
            raise ValueError(f"archived trained implementation changed: {relative}")
        if relative not in CONTROL_IMPLEMENTATIONS:
            active = current_source / "src/archlab" / relative
            if hashlib.sha256(active.read_bytes()).hexdigest() != digest:
                raise ValueError(f"serving changed a protected forward implementation: {relative}")
