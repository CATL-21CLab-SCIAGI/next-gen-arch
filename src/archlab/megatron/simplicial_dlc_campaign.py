"""Fail-closed four-node DP32 pilots in an existing container; no allocation control.

The supervisor uses SSH only to launch bounded project workers. A worker owns
one torchrun process group, records its identity, and never controls services.
SIGTERM to the supervisor requests a native checkpointed stop on every rank.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from archlab.megatron.simplicial_campaign import source_hashes, write_json


def await_evidence(path, timeout=120):
    """Allow remote NAS metadata to become visible after successful SSH exits."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"completed workers did not publish {path.name}") from None
            time.sleep(1)


def validate_probes(root, source):
    expected = source_hashes(source / "src/archlab")
    reference = None
    for arm in "ABC":
        directory = root / arm
        evidence = json.loads((directory / "PROBE_COMPLETE.json").read_text())
        required = ("all_model_weights_bitwise_restored", "scheduler_restored",
                    "optimizer_tensors_bitwise_restored", "all_ranks_passed")
        if not all(evidence.get(k) is True for k in required) or evidence.get("dp_world_size") != 32:
            raise ValueError(f"{arm}: missing DP32 checkpoint/optimizer proof")
        contract = json.loads((directory / "RUN_CONTRACT.json").read_text())
        if contract["arm"] != arm or contract["mode"] != "probe":
            raise ValueError("incorrect probe arm/mode")
        topology = contract["topology"]
        if any(topology[k] != 1 for k in ("tp", "pp", "ep", "cp", "expt_tp")) or any(
            topology[k] != 32 for k in ("dp", "expt_dp")
        ):
            raise ValueError("probe topology is not DP-only with 32 replicas")
        if any(expected.get(k) != v for k, v in contract["source_sha256"].items()):
            raise ValueError("probe source mismatch")
        records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        trains = [r for r in records if r["event"] == "train"]
        if [r["step"] for r in trains] != [1, 2, 3]:
            raise ValueError("three native probe updates required")
        paired = {k: contract[k] for k in ("runtime", "topology", "seed", "global_batch", "micro_batch",
                   "data_order", "evaluation_distribution", "eval_sequences", "manifest_sha256",
                   "train_prefixes", "heldout_prefixes", "source_sha256")}
        paired["init"] = json.loads((directory / "INITIALIZATION.json").read_text())["common_parameter_sha256"]
        paired["data"] = [r["first_four_microbatches_sha256"] for r in trains]
        paired["heldout"] = sorted({r["heldout_tokens_sha256"] for r in records if r["event"] == "eval"})
        if len(paired["heldout"]) != 1:
            raise ValueError("fixed evaluation window drift")
        if reference is not None and paired != reference:
            raise ValueError("A/B/C probe contracts or paired data/initialization differ")
        reference = paired
    return reference


def remote_environment(options):
    return {"PYTHONPATH": f"{options.source}/src:{options.megatron_root}",
            "CUDA_HOME": "/usr/local/cuda", "CPATH": "/usr/local/cuda/targets/x86_64-linux/include",
            "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas", "OMP_NUM_THREADS": "4",
            "CUDA_DEVICE_MAX_CONNECTIONS": "32", "NCCL_DEBUG": "WARN",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1", "NGA_CONTAINER_DIGEST": options.container}


def ssh_command(options, host, command):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
            "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={options.known_hosts}",
            "-i", str(options.identity), f"root@{host}", shlex.join(command)]


def worker(options):
    root, arm, node = options.campaign_dir, options.worker_arm, options.worker_node
    with Path("/tmp/archlab-simplicial-dlc.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = root / arm
        command = [options.python, "-m", "torch.distributed.run", "--nnodes=4",
                   "--nproc-per-node=8", f"--node-rank={node}",
                   f"--master-addr={options.hosts[0]}", f"--master-port={options.port}",
                   "--module", "archlab.megatron.simplicial_pilot", "--arm", arm,
                   "--mode", options.phase, "--run-dir", str(run),
                   "--data-root", str(options.data_root), "--tokenizer", str(options.tokenizer),
                   "--attention-backend", "auto", "--gradient-accumulation-fusion"]
        if options.phase == "probe":
            command.extend(["--steps", "3", "--global-batch", "256", "--eval-sequences", "4",
                            "--eval-interval", "3", "--save-interval", "3"])
        if options.phase == "train" or arm != "A":
            reference_root = options.probe_root if options.phase == "train" else root
            command.extend(["--initialization-reference", str(reference_root / "A/INITIALIZATION.json")])
        state_file = root / f"{arm}-node-{node}.json"
        state = {"host": socket.gethostname(), "node_rank": node, "command": command,
                 "worker_pid": os.getpid(), "started_unix": time.time(), "status": "running"}
        environment = os.environ | remote_environment(options)
        # Do not inherit the DLC controller's node-level RANK/WORLD_SIZE as
        # worker ranks; torchrun supplies the actual 32-way topology.
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                    "NVTE_GROUPED_LINEAR_SINGLE_PARAM"):
            environment.pop(key, None)
        with (root / f"{arm}-node-{node}.log").open("x") as log:
            child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["torchrun_pid"] = child.pid
            write_json(state_file, state)

            def stop(*_):
                if run.is_dir():
                    write_json(run / f"STOP_REQUESTED.node{node}.json", {"requested_unix": time.time()})
                    # Pilot reads the shared supervisor marker. This fallback
                    # handles a worker receiving a direct signal itself.
                    if node == 0:
                        write_json(run / "STOP_REQUESTED.json", {"requested_unix": time.time()})

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            result = child.wait()
        state.update(status="complete" if result == 0 else "failed", exit_code=result,
                     finished_unix=time.time())
        write_json(state_file, state)
        return result


def supervisor(options):
    root = options.campaign_dir
    if (root / "CAMPAIGN.json").exists():
        raise ValueError("campaign is fresh-only; never overwrite an existing run")
    expected = source_hashes(options.source / "src/archlab")
    if options.phase == "train":
        paired = validate_probes(options.probe_root, options.source)
        if paired["micro_batch"] != 4 or paired["seed"] != 42:
            raise ValueError("probe batch/seed drift")
    state = {"status": "running", "phase": options.phase, "supervisor_pid": os.getpid(),
             "source_sha256": expected, "source_commit": options.source_commit,
             "hosts": options.hosts, "arms": {arm: "queued" for arm in "ABC"}}
    write_json(root / "CAMPAIGN.json", state)
    stopping = {"requested": False}

    def stop(*_):
        stopping["requested"] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for arm in "ABC":
        if stopping["requested"]:
            break
        if source_hashes(options.source / "src/archlab") != expected:
            raise RuntimeError("source snapshot changed")
        state["active_arm"] = arm
        state["arms"][arm] = "running"
        write_json(root / "CAMPAIGN.json", state)
        children, logs = [], []
        for node, host in enumerate(options.hosts):
            command = ["env", *(f"{k}={v}" for k, v in remote_environment(options).items()),
                       options.python, "-m", "archlab.megatron.simplicial_dlc_campaign",
                       *(arg for arg in sys.argv[1:] if arg != "--detach"),
                       "--worker-node", str(node), "--worker-arm", arm]
            log = (root / f"{arm}-ssh-{node}.log").open("x")
            logs.append(log)
            children.append(subprocess.Popen(ssh_command(options, host, command),
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
        failure = False
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                failure = True
                stopping["requested"] = True
            if stopping["requested"] and (root / arm).is_dir():
                marker = root / arm / "STOP_REQUESTED.json"
                if not marker.exists():
                    write_json(marker, {"requested_unix": time.time(), "peer_failed": failure})
            time.sleep(1)
        for log in logs:
            log.close()
        results = [p.returncode for p in children]
        completed = root / arm / ("PROBE_COMPLETE.json" if options.phase == "probe" else "TRAINING_COMPLETE.json")
        evidence = None
        if not any(results) and not stopping["requested"]:
            evidence = await_evidence(completed)
        if any(results) or evidence is None:
            state["arms"][arm] = "stopped" if stopping["requested"] and not failure else "failed"
            state.update(status=state["arms"][arm], exit_codes=results)
            write_json(root / "CAMPAIGN.json", state)
            return
        if options.phase == "train" and evidence.get("consumed_tokens") != 3003121664:
            raise RuntimeError("incorrect training token budget")
        state["arms"][arm] = "complete"
        write_json(root / "CAMPAIGN.json", state)
    state["status"] = "stopped" if stopping["requested"] else "complete"
    if options.phase == "probe" and state["status"] == "complete":
        validate_probes(root, options.source)
    write_json(root / "CAMPAIGN.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("probe", "train"), required=True)
    parser.add_argument("--probe-root", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--hosts", nargs=4, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--port", type=int, default=23567)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worker-node", type=int, choices=range(4))
    parser.add_argument("--worker-arm", choices=("A", "B", "C"))
    options = parser.parse_args()
    if options.phase == "train" and options.probe_root is None:
        parser.error("training requires --probe-root")
    if options.worker_node is not None:
        if options.worker_arm is None:
            parser.error("worker requires an arm")
        sys.exit(worker(options))
    root = options.campaign_dir
    root.mkdir(parents=True, exist_ok=True)
    if options.detach:
        if any(root.iterdir()):
            raise ValueError("refusing existing campaign contents")
        command = [sys.executable, "-m", "archlab.megatron.simplicial_dlc_campaign",
                   *(arg for arg in sys.argv[1:] if arg != "--detach")]
        with (root / "supervisor.log").open("x") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        write_json(root / "SUPERVISOR.json", {"pid": process.pid, "command": command})
        print(f"DLC {options.phase} supervisor {process.pid}: {root}", flush=True)
        return
    try:
        supervisor(options)
    except Exception as error:
        write_json(root / "FAILURE.json", {"error": repr(error), "time_unix": time.time()})
        if (root / "CAMPAIGN.json").exists():
            state = json.loads((root / "CAMPAIGN.json").read_text())
            state.update(status="failed", error=repr(error))
            write_json(root / "CAMPAIGN.json", state)
        raise


if __name__ == "__main__":
    main()
