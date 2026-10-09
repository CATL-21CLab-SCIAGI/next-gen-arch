"""Complete a requested RL checkpointed stop, then start a frozen scaling plan."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import shlex
import subprocess
import time
import traceback
from pathlib import Path

from archlab.artifacts import atomic_write_json


def ssh(alias, command, *, timeout=600):
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", alias, shlex.join(command)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"{alias}: {result.stderr[-4000:]} {result.stdout[-1000:]}")
    return result.stdout


def launch(plan, node):
    root = Path(plan["root"])
    source = Path(plan["source"])
    command = [
        "/opt/venv/bin/python",
        "-u",
        "-m",
        "archlab.automodel.deepseek_v41_scaling_campaign",
        "--plan",
        str(root / "PLAN.json"),
        "--node",
        node,
    ]
    code = f"""
import json,os,subprocess,time
from pathlib import Path
root=Path({str(root)!r})
marker=root/{("LAUNCH-" + node + ".json")!r}
assert not marker.exists(), 'worker already launched'
environment=os.environ.copy()
environment.update({plan["environment"]!r})
with (root/{("worker-" + node + ".log")!r}).open('x') as log:
    process=subprocess.Popen({command!r},cwd={str(source)!r},env=environment,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={{'pid':process.pid,'command':{command!r},'started_at_epoch':time.time()}}
marker.write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
"""
    return json.loads(ssh(plan["nodes"][node]["ssh_alias"], ["/opt/venv/bin/python", "-c", code]))


def transition(plan, rl_root, project):
    root = Path(plan["root"])

    def record(stage, **values):
        result = {"stage": stage, "observed_at_epoch": time.time(), **values}
        atomic_write_json(root / "TRANSITION_STATUS.json", result)
        print(json.dumps(result), flush=True)

    record("waiting_for_native_rl_checkpoint")
    deadline = time.monotonic() + 5 * 3600
    while not (root / "RL_CHECKPOINT_SAVED.json").is_file():
        if (root / "RL_CHECKPOINT_STOP_FAILURE.json").exists():
            raise RuntimeError("native RL checkpoint failed; retaining the quiesced state")
        if time.monotonic() > deadline:
            raise TimeoutError("native RL checkpoint has not finished")
        time.sleep(15)
    saved = json.loads((root / "RL_CHECKPOINT_SAVED.json").read_text())
    quiesced = json.loads((root / "RL_DRIVER_QUIESCED.json").read_text())
    master = plan["nodes"]["master-0"]["ssh_alias"]
    runtime = [
        "env",
        "PYTHONPATH=" + str(project / "src"),
        "/opt/venv/bin/python",
        "-m",
        "archlab.serving.isolated_sglang_runtime",
        "--config",
        str(rl_root / "runtime-master-0-inspect.json"),
        "--",
    ]
    record("verifying_completed_rl_checkpoint", checkpoint=saved["checkpoint"])
    readback = ssh(
        master,
        runtime
        + [
            "-m",
            "archlab.megatron.miles_v41_saved_checkpoint",
            "--run-root",
            str(rl_root),
            "--iteration",
            str(saved["completed_rollout"]),
            "--output",
            str(root / "RL_CHECKPOINT_READBACK.json"),
        ],
        timeout=1800,
    )
    (root / "rl-checkpoint-readback.log").write_text(readback)
    record("retiring_native_rl_workers")
    retired = ssh(
        master,
        runtime
        + [
            "-m",
            "archlab.megatron.miles_v41_stop",
            "retire",
            "--run-root",
            str(rl_root),
            "--output",
            str(root),
            "--address",
            plan["rl_address"],
            "--driver-pid",
            str(quiesced["driver_pid"]),
        ],
        timeout=900,
    )
    (root / "rl-retire.log").write_text(retired)
    observations = {}
    for _ in range(24):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                node: pool.submit(
                    ssh,
                    item["ssh_alias"],
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                    timeout=60,
                )
                for node, item in plan["nodes"].items()
            }
            observations = {node: future.result().strip() for node, future in futures.items()}
        if not any(observations.values()):
            break
        time.sleep(5)
    if any(observations.values()):
        raise RuntimeError(f"GPU processes remain after native retirement: {observations}")
    atomic_write_json(
        root / "RL_RETIRED.json",
        {
            "all_gpus_free": True,
            "gpu_count": 32,
            "gpu_type": "B300",
            "allocation_retained": plan["allocation"],
            "observed_at_epoch": time.time(),
            "node_gpu_processes": observations,
            "checkpoint": saved["checkpoint"],
            "checkpoint_readback": str(root / "RL_CHECKPOINT_READBACK.json"),
        },
    )
    record("launching_all_eight_qualifications")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {node: pool.submit(launch, plan, node) for node in plan["nodes"]}
        launched = {node: future.result() for node, future in futures.items()}
    record("scaling_workers_launched", workers=launched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    with (Path(plan["root"]) / "transition.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            transition(plan, args.rl_root.resolve(), args.project.resolve())
        except BaseException:
            atomic_write_json(
                Path(plan["root"]) / "TRANSITION_FAILED.json",
                {
                    "observed_at_epoch": time.time(),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


if __name__ == "__main__":
    main()
