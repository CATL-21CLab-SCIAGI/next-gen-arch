"""Run matched scratch width pairs on an explicitly retained four-node allocation.

Each arm owns two eight-GPU nodes. Shared storage records numerical
qualification; production queues advance independently after each completed run. A successful run ends at the sealed dataset's exact 10B target budget.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from archlab.architectures.deepseek_v41_scratch import (
    SCRATCH_WIDTHS,
    scaling_experts_per_token,
    scratch_adapter_head_dim,
)
from archlab.artifacts import atomic_write_json

TOKEN_BUDGET = 10_000_000_000


def read(path):
    return json.loads(Path(path).read_text())


def validate_plan(plan):
    if plan["token_budget_per_run"] != TOKEN_BUDGET:
        raise ValueError("each run must have its own 10B supervised-token allowance")
    runs = plan["runs"]
    base = [r for r in runs if not r.get("supplemental")]
    if len(base) != 8 or {(r["width"], r["variant"]) for r in base} != {
        (width, variant) for width in SCRATCH_WIDTHS for variant in ("normal", "simplicial")
    }:
        raise ValueError("plan must contain the eight requested base runs")
    for key in ("output", "qualification"):
        if len({r[key] for r in runs}) != len(runs):
            raise ValueError(f"duplicate {key} directories")
    groups = {}
    for variant in ("normal", "simplicial"):
        queue = sorted((r for r in runs if r["variant"] == variant), key=lambda r: r["queue_index"])
        groups[variant] = tuple(queue[0]["nodes"])
        if len(groups[variant]) != 2 or len(set(groups[variant])) != 2:
            raise ValueError("each variant requires two distinct nodes")
        if [r["queue_index"] for r in queue] != list(range(len(queue))) or queue[0]["width"] != 640:
            raise ValueError("each independent queue must start with a fresh d640 run")
        if [r["width"] for r in queue if not r.get("supplemental")] != [640, 128, 384, 1280]:
            raise ValueError("supplemental runs must preserve the base width sequence")
        for i, run in enumerate(queue):
            if run.get("supplemental"):
                if variant != "normal" or not 1 < i < len(queue) - 2:
                    raise ValueError("the extra sweep belongs after normal d128 and before d384")
                if not 0 < run.get("training_token_budget", TOKEN_BUDGET) <= TOKEN_BUDGET:
                    raise ValueError("supplemental token use exceeds the per-run allowance")
        if any(tuple(r["nodes"]) != groups[variant] for r in queue):
            raise ValueError("a variant must retain its assigned two nodes")
        ports = [r["master_port"] + phase for r in queue for phase in (0, 1)]
        if len(set(ports)) != len(ports) or any(not 1024 <= port <= 65535 for port in ports):
            raise ValueError("qualification and training need distinct rendezvous ports")
    if (
        set(groups["normal"]) & set(groups["simplicial"])
        or (set(groups["normal"]) | set(groups["simplicial"]) != set(plan["nodes"]))
        or len(plan["nodes"]) != 4
    ):
        raise ValueError("the two variants must partition the four existing nodes")


def validate_qualification_pair(normal, simplicial):
    roots = [Path(row["qualification"]) for row in (normal, simplicial)]
    qualifications = [read(root / "QUALIFIED.json") for root in roots]
    contracts = [copy.deepcopy(q["contract"]) for q in qualifications]
    if not all(
        q["passed"] and q["world_size"] == 16 and q["context"] == 2048 for q in qualifications
    ):
        raise ValueError("both complete sixteen-GPU production-shape qualifications are required")
    width = normal["width"]
    if width != simplicial["width"]:
        raise ValueError("pair widths differ")
    parameters = []
    for variant, contract in zip(("normal", "simplicial"), contracts, strict=True):
        if contract.pop("variant") != variant or contract["runtime"].pop("variant") != variant:
            raise ValueError("pair variants differ from their assignments")
        parameters.append(contract["runtime"].pop("parameters"))
        if contract["runtime"]["geometry"]["text_config"]["hidden_size"] != width:
            raise ValueError("qualified model width differs from the plan")
        text = contract["runtime"]["geometry"]["text_config"]
        if (
            text["moe_intermediate_size"],
            text["num_experts_per_tok"],
            text["n_shared_experts"],
        ) != (128, scaling_experts_per_token(width), 1):
            raise ValueError("qualified MoE does not preserve the effective 3.0 width ratio")
        if contract["target_supervised_tokens"] != TOKEN_BUDGET:
            raise ValueError("qualified budget differs")
    expected_difference = 8 * (
        2 * width * 2 * scratch_adapter_head_dim(width) + scratch_adapter_head_dim(width)
    )
    if parameters[1] - parameters[0] != expected_difference:
        raise ValueError("parameter difference extends beyond the declared simplicial branch")
    if contracts[0] != contracts[1]:
        raise ValueError("paired contracts differ outside the attention treatment")
    fingerprints = []
    for rank in range(16):
        reports = [read(root / f"rank-{rank:02d}-qualified.json") for root in roots]
        if not all(
            r["passed"] and r["exact_checkpoint_restore"] and r["identical_next_update_loss"]
            for r in reports
        ):
            raise ValueError(f"rank {rank} failed checkpoint or numerical qualification")
        values = [read(root / f"rank-{rank:02d}-initial.json")["common_sha256"] for root in roots]
        if values[0] != values[1]:
            raise ValueError(f"rank {rank} paired backbone initialization differs")
        fingerprints.append(values[0])
    text = contracts[0]["runtime"]["geometry"]["text_config"]
    engram = sum(text["engram_num_embeddings"]) * text["engram_head_dim"]
    return {
        "passed": True,
        "width": width,
        "parameters": dict(zip(("normal", "simplicial"), parameters, strict=True)),
        "engram_lookup_parameters": engram,
        "parameter_difference": expected_difference,
        "common_initial_sha256_by_rank": fingerprints,
        "token_budget_per_run": TOKEN_BUDGET,
    }


def partner(plan, run):
    return next(
        r
        for r in plan["runs"]
        if not r.get("supplemental")
        and r["width"] == run["width"]
        and r["variant"] != run["variant"]
    )


def wait_for_qualification(plan, run, *, timeout=6 * 3600):
    peer = partner(plan, run)
    deadline = time.monotonic() + timeout
    target = Path(peer["qualification"]) / "QUALIFIED.json"
    while not target.exists():
        if any((Path(plan["root"]) / f"FAILED-{node}.json").exists() for node in peer["nodes"]):
            raise RuntimeError("peer qualification failed")
        if (Path(plan["root"]) / "STOP_REQUEST").exists():
            raise RuntimeError("campaign stop requested before admission")
        if time.monotonic() > deadline:
            raise TimeoutError(f"waiting for qualification: {target}")
        time.sleep(15)


def run_phase(plan, run, node, mode, environment, state_path):
    directory = Path(run["qualification"] if mode == "qualify" else run["output"])
    directory.mkdir(parents=True, exist_ok=True)
    node_rank = run["nodes"].index(node)
    leader = plan["nodes"][run["nodes"][0]]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=2",
        "--nproc-per-node=8",
        "--node-rank",
        str(node_rank),
        "--master-addr",
        leader["ip"],
        "--master-port",
        str(run["master_port"] + (1 if mode == "train" else 0)),
        "--max-restarts=0",
        "--module",
        run.get("training_module", "archlab.automodel.deepseek_v41_scratch_training"),
        "--variant",
        run["variant"],
        "--width",
        str(run["width"]),
        "--sparse-backend",
        "batched",
        "--scaling-study",
        "--mode",
        mode,
        "--output",
        str(directory),
        "--context",
        "2048",
    ]
    command += run.get("extra_args", [])
    if mode == "train":
        command += ["--qualification", run["qualification"]]
        if run.get("resume_checkpoint"):
            command += ["--resume", run["resume_checkpoint"]]
    started = time.time()
    suffix = ""
    if mode == "train" and run.get("resume_checkpoint"):
        suffix = "-resume-" + Path(run["resume_checkpoint"]).name
    log_path = directory / f"launcher-{node}{suffix}.log"
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=run.get("source", plan["source"]),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        atomic_write_json(
            directory / f"LAUNCH-{node}{suffix}.json",
            {
                "command": command,
                "pid": process.pid,
                "started_at_epoch": started,
                "source_commit": run.get("source_commit", plan["source_commit"]),
                "node": node,
                "node_rank": node_rank,
                "local_gpu_count": 8,
                "variant_gpu_count": 16,
                "gpu_type": "B300",
            },
        )
        try:
            while True:
                code = process.poll()
                atomic_write_json(
                    state_path,
                    {
                        "mode": mode,
                        "width": run["width"],
                        "variant": run["variant"],
                        "pid": process.pid,
                        "state": "running" if code is None else "exited",
                        "exit_code": code,
                        "started_at_epoch": started,
                        "observed_at_epoch": time.time(),
                        "allocated_gpu_seconds": (time.time() - started) * 8,
                    },
                )
                if code is not None:
                    if code:
                        raise RuntimeError(f"{mode} exited {code}: {log_path}")
                    break
                if any(
                    (Path(plan["root"]) / f"FAILED-{peer}.json").exists()
                    for peer in run["nodes"]
                    if peer != node
                ):
                    raise RuntimeError("the other node in this variant failed")
                if mode == "train" and (Path(plan["root"]) / "STOP_REQUEST").exists():
                    (directory / "STOP_REQUEST").touch(exist_ok=True)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=180)
    marker = directory / ("QUALIFIED.json" if mode == "qualify" else "COMPLETE.json")
    deadline = time.monotonic() + 120
    while not marker.is_file() and time.monotonic() < deadline:
        time.sleep(2)
    if not marker.is_file():
        raise RuntimeError(f"{mode} exited without its completion marker")


def phase_environment(plan, run):
    environment = os.environ.copy()
    environment.update(run.get("environment", plan["environment"]))
    environment["NGA_EXPECTED_COMMIT"] = run.get("source_commit", plan["source_commit"])
    return environment


def verify_completed_run(run):
    """A completed run must retain its exact budget and five OSS checkpoints."""
    output = Path(run["output"])
    complete = read(output / "COMPLETE.json")
    budget = run.get("training_token_budget", TOKEN_BUDGET)
    if not complete["passed"] or complete["supervised_tokens"] != budget:
        raise ValueError("training ended without its exact declared token budget")
    catalog = read(output / "EVAL_CHECKPOINTS.json")["checkpoints"]
    if [r["milestone_tokens"] for r in catalog] != list(
        range(budget // 5, budget + 1, budget // 5)
    ):
        raise ValueError("five OSS evaluation checkpoints are required before completion")
    for row in catalog:
        path = Path(row["path"])
        if not path.is_symlink() or str(path.resolve(strict=True)) != row["oss_path"]:
            raise ValueError("evaluation checkpoint lost its OSS link")
    return catalog


def qualify_local_pair(plan, run, node, state):
    """Qualify both treatments on this arm's GPUs without waiting for its peer.

    A geometry revision invalidates earlier qualifications. The paired control
    is a short numerical qualification, not another production run.
    """
    peer = partner(plan, run)
    for key in ("source", "source_commit", "extra_args", "environment"):
        if run.get(key, plan.get(key)) != peer.get(key, plan.get(key)):
            raise ValueError(f"local qualification pair differs in {key}")
    control = {
        **peer,
        "nodes": run["nodes"],
        "master_port": run["master_port"],
        "qualification": run["qualification"] + "-paired-control",
    }
    pair = sorted((run, control), key=lambda r: r["variant"])
    for candidate in pair:
        marker = Path(candidate["qualification"]) / "QUALIFIED.json"
        if not marker.is_file():
            run_phase(plan, candidate, node, "qualify", phase_environment(plan, candidate), state)
    return validate_qualification_pair(*pair)


def training_queue(plan, node):
    """Read a pinned supplemental plan at the d128 boundary, without arm barriers."""
    runs = sorted((r for r in plan["runs"] if node in r["nodes"]), key=lambda r: r["queue_index"])
    order = plan.get("production_width_order")
    if order is not None:
        if sorted(order) != sorted(SCRATCH_WIDTHS):
            raise ValueError("production width priority must include every base width exactly once")
        runs.sort(key=lambda run: order.index(run["width"]))
    for run in runs:
        yield run
        extension = plan.get("supplemental_plan_path")
        if (
            extension
            and run["variant"] == "normal"
            and run["width"] == 128
            and not run.get("supplemental")
        ):
            additions = read(extension)["runs"]
            if not additions or any(not r.get("supplemental") for r in additions):
                raise ValueError("supplemental plan must declare its independent sweep cells")
            combined = copy.deepcopy(plan)
            combined.pop("supplemental_plan_path", None)
            combined["runs"] = []
            for variant in ("normal", "simplicial"):
                base = sorted(
                    (r for r in plan["runs"] if r["variant"] == variant),
                    key=lambda r: r["queue_index"],
                )
                queue = base[:2] + additions + base[2:] if variant == "normal" else base
                combined["runs"].extend({**r, "queue_index": i} for i, r in enumerate(queue))
            validate_plan(combined)
            for addition in additions:
                if node not in addition["nodes"]:
                    raise ValueError("supplemental plan changed node ownership")
                yield addition


def finish_handoff(plan, node, *, handoff_path=None):
    """Retire a stopped queue supervisor after its live child exits successfully.

    The child keeps running and becomes a zombie owned by the stopped parent;
    Linux preserves its actual exit status until the handoff reads it. Never
    signal the GPU child or infer successful exit from disappearance alone.
    """
    root = Path(plan["root"])
    path = Path(handoff_path) if handoff_path is not None else root / f"LOOP_HANDOFF-{node}.json"
    handoff = read(path)
    if handoff.get("finished"):
        if handoff.get("resume_checkpoint"):
            run = next(
                r
                for r in plan["runs"]
                if not r.get("supplemental")
                and r["variant"] == handoff["variant"]
                and r["width"] == handoff["width"]
            )
            run["resume_checkpoint"] = handoff["resume_checkpoint"]
        return
    if handoff["mode"] not in ("qualify", "train", "complete") or handoff["variant"] not in ("normal", "simplicial"):
        raise ValueError("handoff requires a recorded qualification or checkpointed stop")
    parent, child = handoff["worker_pid"], handoff["phase_pid"]

    def stat(pid, start):
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        if fields[21] != start:
            raise ValueError("handoff process identity changed")
        return fields

    if stat(parent, handoff["worker_start_ticks"])[2] != "T":
        raise ValueError("previous supervisor must be stopped at the recorded boundary")
    # Natural completion can take days; a queue revision must not stop training.
    deadline = time.monotonic() + (30 * 86400 if handoff["mode"] == "complete" else 6 * 3600)
    while True:
        fields = stat(child, handoff["phase_start_ticks"])
        if int(fields[3]) != parent:
            raise ValueError("phase no longer belongs to the recorded supervisor")
        if fields[2] == "Z":
            code = os.waitstatus_to_exitcode(int(fields[51]))
            if code != 0:
                teardown = handoff.get("checkpointed_teardown", {})
                ranks = teardown.get("ranks", [])
                if (
                    handoff["mode"] != "train"
                    or teardown.get("phase_pid") != child
                    or len(ranks) != 8
                    or len({rank["pid"] for rank in ranks}) != 8
                    or not all(rank.get("in_destroy") and rank.get("returncode") == 0 for rank in ranks)
                    or teardown.get("verification", {}).get("ranks") != 16
                ):
                    raise RuntimeError(f"qualification exited {code} before handoff")
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("qualification did not finish before handoff timeout")
        if handoff["mode"] == "complete":
            atomic_write_json(root / f"WORKER-{node}.json", {
                "mode": "train", "width": handoff["width"], "variant": handoff["variant"],
                "pid": child, "state": "running", "queue_revision": "waiting_for_natural_completion",
                "observed_at_epoch": time.time(), "started_at_epoch": handoff["started_at_epoch"],
                "allocated_gpu_seconds": (time.time() - handoff["started_at_epoch"]) * 8,
            })
        time.sleep(15)
    run = next(
        r
        for r in plan["runs"]
        if not r.get("supplemental")
        and r["variant"] == handoff["variant"]
        and r["width"] == handoff["width"]
    )
    if not read(Path(run["qualification"]) / "QUALIFIED.json")["passed"]:
        raise ValueError("phase exited without successful qualification")
    if handoff["mode"] == "complete":
        verify_completed_run(run)
    if handoff["mode"] == "train":
        candidates = sorted((Path(run["output"]) / "recovery").glob("step-*/COMPLETE.json"))
        if not candidates:
            raise ValueError("training stopped without a complete recovery checkpoint")
        marker = read(candidates[-1])
        if marker["contract"] != read(Path(run["qualification"]) / "QUALIFIED.json")["contract"]:
            raise ValueError("recovery checkpoint changed the qualified training contract")
        if marker["cursor"]["step"] < handoff["minimum_step"]:
            raise ValueError("recovery checkpoint predates the requested maintenance stop")
        run["resume_checkpoint"] = str(candidates[-1].parent)
        handoff["resume_checkpoint"] = run["resume_checkpoint"]
        output = Path(run["output"])
        stop = output / "STOP_REQUEST"
        archived = output / f"MAINTENANCE_STOP-{handoff['minimum_step']}.json"
        # Both node supervisors share this directory. Serialize the check and
        # rename so neither can remove a newer, unrelated operator request.
        with (output / "maintenance-stop.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            recorded = stop if stop.exists() else archived
            if not recorded.exists():
                raise ValueError("recorded maintenance stop request is missing")
            if hashlib.sha256(recorded.read_bytes()).hexdigest() != handoff["stop_request_sha256"]:
                raise ValueError("refusing to clear an unrelated stop request")
            if recorded == stop:
                try:
                    stop.rename(archived)
                except FileNotFoundError:
                    # NAS attribute caches can still expose the old name after
                    # the peer archives it. Accept only the identical request.
                    if not archived.is_file() or hashlib.sha256(archived.read_bytes()).hexdigest() != handoff["stop_request_sha256"]:
                        raise
        original_initialization = output / "INITIALIZATION_BEFORE_MAINTENANCE"
        original_initialization.mkdir(exist_ok=True)
        for initial in output.glob("rank-*-initial.json"):
            saved = original_initialization / initial.name
            if not saved.exists():
                saved.write_bytes(initial.read_bytes())
    os.kill(parent, signal.SIGTERM)
    os.kill(parent, signal.SIGCONT)
    handoff.update(finished=True, phase_exit_code=code, finished_at_epoch=time.time())
    atomic_write_json(path, handoff)
    # Give the old supervisor time to release its exclusive filesystem lock.
    time.sleep(2)


def worker(plan, node, *, resume=False):
    root = Path(plan["root"])
    expected_hostname = plan["nodes"][node]["hostname"]
    if socket.gethostname() != expected_hostname:
        raise ValueError("worker is on the wrong DLC node")
    if not read(root / "RL_RETIRED.json")["all_gpus_free"]:
        raise ValueError("checkpointed RL retirement has not completed")
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    )
    if busy.strip():
        raise ValueError("node has GPU processes before scratch admission")
    runs = sorted((r for r in plan["runs"] if node in r["nodes"]), key=lambda r: r["queue_index"])
    state = root / f"WORKER-{node}.json"
    for preflight in plan.get("preflight_runs", []):
        if node in preflight["nodes"]:
            receipt = Path(preflight["qualification"]) / "QUALIFIED.json"
            if resume and receipt.is_file():
                if not read(receipt)["passed"]:
                    raise ValueError("cannot reuse a failed preflight qualification")
                continue
            run_phase(plan, preflight, node, "qualify", phase_environment(plan, preflight), state)
    # Qualify all shapes early. Only admission compares arms; production has
    # no step, checkpoint, or completion barrier with the other variant.
    for run in runs:
        if run.get("supplemental") or run.get("qualify_pair_locally"):
            continue
        if (root / "STOP_REQUEST").exists():
            return
        if resume and (Path(run["qualification"]) / "QUALIFIED.json").is_file():
            if not read(Path(run["qualification"]) / "QUALIFIED.json")["passed"]:
                raise ValueError("cannot reuse a failed qualification")
            continue
        run_phase(plan, run, node, "qualify", phase_environment(plan, run), state)
    completed = 0
    for run in training_queue(plan, node):
        if (root / "STOP_REQUEST").exists():
            return
        if resume and (Path(run["output"]) / "COMPLETE.json").is_file():
            verify_completed_run(run)
            completed += 1
            continue
        if run.get("supplemental"):
            run_phase(plan, run, node, "qualify", phase_environment(plan, run), state)
        else:
            if run.get("qualify_pair_locally"):
                receipt = qualify_local_pair(plan, run, node, state)
            else:
                wait_for_qualification(plan, run)
                pair = sorted((run, partner(plan, run)), key=lambda r: r["variant"])
                receipt = validate_qualification_pair(*pair)
            if node == run["nodes"][0]:
                suffix = f"-on-{run['variant']}" if run.get("qualify_pair_locally") else ""
                atomic_write_json(root / f"PAIR_QUALIFIED-d{run['width']}{suffix}.json", receipt)
        run_phase(plan, run, node, "train", phase_environment(plan, run), state)
        catalog = verify_completed_run(run)
        if node == run["nodes"][0]:
            atomic_write_json(Path(run["output"]) / "FINAL_CHECKPOINT_VERIFIED.json", catalog[-1])
        completed += 1
    atomic_write_json(
        state, {"state": "complete", "observed_at_epoch": time.time(), "runs": completed}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--handoff", action="store_true")
    parser.add_argument("--handoff-record", type=Path)
    parser.add_argument("--resume-admission", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    validate_plan(plan)
    root = Path(plan["root"])
    if args.resume_admission:
        if args.handoff:
            raise ValueError("admission restart and training handoff are distinct operations")
        for run in training_queue(plan, args.node):
            if list(Path(run["output"]).glob("launcher-*.log")):
                raise ValueError(
                    "admission restart cannot restart production; use checkpoint handoff"
                )
    if args.handoff:
        finish_handoff(plan, args.node, handoff_path=args.handoff_record)
    with (root / f"worker-{args.node}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            worker(plan, args.node, resume=args.handoff or args.resume_admission)
        except BaseException:
            atomic_write_json(
                root / f"FAILED-{args.node}.json",
                {
                    "observed_at_epoch": time.time(),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


if __name__ == "__main__":
    main()
