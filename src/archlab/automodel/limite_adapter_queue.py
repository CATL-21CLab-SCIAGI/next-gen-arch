"""Independent two-node finetuning -> qualified RL handoff per variant."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def evaluation_ready(gate):
    if "gates" in gate:
        gates = gate["gates"]
        if not isinstance(gates, list) or not gates:
            raise ValueError("evaluation gate requires a nonempty gates list")
        return all(evaluation_ready(child) for child in gates)
    state_path = Path(gate["state"])
    if not state_path.exists():
        return False
    state = json.loads(state_path.read_text())
    if state.get("status") in ("failed", "memory_overflow"):
        return False
    return all(
        state.get("stages", {}).get(name, {}).get("status") == "complete"
        and state["stages"][name].get("plan_sha256") == digest
        for name, digest in gate["stages"].items()
    ) and bool(gate["stages"])


def release_evaluation_pause(gate):
    pause = Path(gate["pause_file"])
    try:
        contents = pause.read_text()
    except FileNotFoundError:
        return  # The other node can release the same owned pause first.
    if contents != gate["pause_contents"]:
        raise ValueError("pause request changed while waiting for evaluation")
    pause.unlink(missing_ok=True)


def check_handoff(plan, marker, qualification):
    """Bind full-weight RL to this variant's completed, qualified parent."""
    if marker["tokens"] != 10_000_000_000:
        raise ValueError("RL handoff requires exactly 10B warmup tokens")
    if qualification.get("passed") is not True:
        raise ValueError("RL qualification did not pass")
    # Older adapter plans retain their original handoff contract. New full-weight
    # plans explicitly declare scope, so an old READY file cannot start them.
    mode = plan.get("trainable_mode")
    if mode is None:
        return
    backend = plan["attention_backend"]
    parent = json.loads((Path(marker["checkpoint"]) / "COMPLETE.json").read_text())
    if (
        parent.get("tokens") != marker["tokens"]
        or parent.get("trainable_mode") != mode
        or parent["adapter"].get("attention_backend", "native") != backend
        or parent["adapter"]["variant"] != plan["variant"]
    ):
        raise ValueError("RL parent differs from the full-weight finetuning contract")
    if (
        qualification.get("trainable_mode") != mode
        or qualification.get("attention_backend") != backend
    ):
        raise ValueError("RL qualification differs from the finetuning scope or backend")
    specification = qualification["variants"][plan["variant"]]
    if specification["revision"] != plan.get("rl_revision", plan["warmup"]["revision"]):
        raise ValueError("qualified RL source differs from the sealed finetuning revision")
    arguments = specification["arguments"]
    for flag, expected in (
        ("--trainable-mode", mode),
        ("--attention-backend", backend),
        ("--variant", plan["variant"]),
        ("--warmup", plan["output"]),
    ):
        index = arguments.index(flag) if flag in arguments else -1
        if (
            arguments.count(flag) != 1
            or index + 1 >= len(arguments)
            or arguments[index + 1] != expected
        ):
            raise ValueError("qualified RL arguments differ from this finetuning plan: " + flag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--wait-for-evaluation", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    nodes = len(plan.get("nodes", ["master-0", "worker-0"]))
    if nodes not in (1, 2) or args.node_rank >= nodes:
        raise ValueError("queue node rank differs from its one- or two-node allocation")
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=True)
    state = root / f"queue-node{args.node_rank}.json"

    def stopped():
        return (root / "STOP_REQUEST").exists() or (root.parent / "STOP_REQUEST").exists()

    def stage(name, spec):
        source = Path(spec["source"])
        if (source / "SOURCE_REVISION").read_text().strip() != spec["revision"]:
            raise ValueError("queue source revision does not match sealed plan")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={nodes}",
            "--nproc_per_node=8",
            f"--node_rank={args.node_rank}",
            f"--master_addr={plan['master_addr']}",
            f"--master_port={spec['port']}",
            "-m",
            spec["module"],
            *spec["arguments"],
        ]
        env = dict(os.environ, ARCHLAB_SOURCE_REVISION=spec["revision"])
        env["PYTHONPATH"] = str(source / "src") + os.pathsep + plan["runtime_overlay"]
        atomic_json(state, dict(stage=name, status="running", command=command, started=time.time()))
        with (root / f"{name}-node{args.node_rank}.log").open("a") as log:
            subprocess.run(
                command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    try:
        if args.wait_for_evaluation:
            gate = json.loads(args.wait_for_evaluation.read_text())
            while not evaluation_ready(gate):
                atomic_json(state, dict(stage="rl", status="waiting_for_evaluation", gate=str(args.wait_for_evaluation)))
                if (root.parent / "STOP_REQUEST").exists():
                    return
                time.sleep(15)
            release_evaluation_pause(gate)
        if stopped():
            atomic_json(state, dict(stage="queue", status="paused"))
            return
        if not (root / "WARMUP_COMPLETE.json").exists():
            stage("warmup", plan["warmup"])
        if stopped() or not (root / "WARMUP_COMPLETE.json").exists():
            atomic_json(state, dict(stage="warmup", status="paused"))
            return
        marker = json.loads((root / "WARMUP_COMPLETE.json").read_text())
        if marker["tokens"] != 10_000_000_000:
            raise ValueError("RL handoff requires exactly 10B warmup tokens")
        ready = Path(plan["rl_ready"])
        atomic_json(state, dict(stage="rl", status="waiting_for_qualification"))
        while not ready.exists():
            if stopped():
                atomic_json(state, dict(stage="rl", status="paused"))
                return
            time.sleep(15)
        qualification = json.loads(ready.read_text())
        check_handoff(plan, marker, qualification)
        finished = root / "rl" / "FINISHED.json"
        if finished.exists() and json.loads(finished.read_text()).get("status") == "complete":
            atomic_json(state, dict(stage="rl", status="complete", finished=time.time()))
            return
        stage("rl", qualification["variants"][plan["variant"]])
        status = json.loads(finished.read_text())["status"]
        atomic_json(state, dict(stage="rl", status="complete" if status == "complete" else "paused", finished=time.time()))
    except BaseException as exc:
        atomic_json(
            state, dict(stage="queue", status="failed", error=repr(exc), failed=time.time())
        )
        raise


if __name__ == "__main__":
    main()
