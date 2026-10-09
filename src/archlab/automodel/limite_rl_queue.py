"""Qualify and launch RL on one persistent, already submitted DLC allocation.

Workers use the existing exact-process adoption protocol. Completion or failure
only changes queue state; this controller never stops or resubmits a cloud job.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.automodel.limite_math_queue import (
    FORMAT as MATH_QUEUE_FORMAT,
)
from archlab.automodel.limite_math_queue import (
    QueueController,
    SSHWorkers,
    final_checkpoint,
    pinned_json,
)
from archlab.automodel.limite_math_queue import (
    validate_plan as validate_math_plan,
)

FORMAT = "archlab-limite-rl-queue-v1"
TERMINAL = {"Stopped", "Failed", "FailedReserving", "Succeeded", "Deleted"}
BOOTSTRAP = r'''
import json,pathlib,subprocess,sys
q=json.load(sys.stdin)
python=q['python']
if not pathlib.Path(python).exists():
 subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(pathlib.Path(python).parents[1])],check=True)
code=r"""import importlib.metadata as m,json,pathlib,subprocess,sys
q=json.load(sys.stdin)
before={name:m.version(name) for name in q['core']}
assert before==q['core'],'container-owned runtime differs'
constraints=pathlib.Path('/tmp/archlab-rl-core-constraints.txt')
constraints.write_text(''.join(name+'=='+version+'\n' for name,version in before.items()))
missing=[]
for name,version in q['external'].items():
 try: actual=m.version(name)
 except m.PackageNotFoundError: actual=None
 if actual!=version: missing.append(name+'=='+version)
try:m.version('datasets')
except m.PackageNotFoundError:missing.append('datasets')
if missing:subprocess.run([sys.executable,'-m','pip','install','-q','-c',str(constraints),*missing],check=True,stdout=sys.stderr)
assert before=={name:m.version(name) for name in before}
import torch
assert torch.cuda.device_count()==8 and torch.version.cuda==q['cuda']
print(json.dumps(dict(core=before,packages={name:m.version(name) for name in q['external']},gpus=8,cuda=torch.version.cuda)))
"""
p=subprocess.run([python,'-c',code],input=json.dumps(q),text=True,capture_output=True)
if p.returncode:raise RuntimeError('runtime bootstrap failed: '+p.stderr[-1500:])
print(p.stdout.strip())
'''


def gate_ready(gates):
    """Missing evidence means unfinished; present contradictory evidence blocks."""
    for gate in gates:
        path = Path(gate["path"])
        for bound in ("minimum", "maximum"):
            constraints = gate.get(bound, {})
            if not isinstance(constraints, dict):
                raise ValueError(f"invalid qualification bound: {path.name}:{bound}")
            for key, expected in constraints.items():
                if (not isinstance(key, str) or not all(key.split("."))
                        or not _finite_gate_number(expected)):
                    raise ValueError(f"invalid qualification bound: {path.name}:{bound}:{key}")
        for key, lower in gate.get("minimum", {}).items():
            if key in gate.get("maximum", {}) and lower > gate["maximum"][key]:
                raise ValueError(f"inverted qualification bounds: {path.name}:{key}")
        if not path.exists():
            return False
        text = path.read_text()
        row = json.loads(text.splitlines()[-1] if gate.get("jsonl") else text)
        for key, expected in gate.get("equals", {}).items():
            actual = row
            for part in key.split("."):
                actual = actual[part]
            if actual != expected:
                raise ValueError(f"qualification differs: {path.name}:{key}")
        for bound in ("minimum", "maximum"):
            for key, expected in gate.get(bound, {}).items():
                actual = row
                for part in key.split("."):
                    if not isinstance(actual, dict) or part not in actual:
                        raise ValueError(f"qualification has missing numeric evidence: {path.name}:{key}")
                    actual = actual[part]
                if not _finite_gate_number(actual):
                    raise ValueError(f"qualification has invalid numeric evidence: {path.name}:{key}")
                if ((bound == "minimum" and actual < expected)
                        or (bound == "maximum" and actual > expected)):
                    raise ValueError(f"qualification outside {bound}: {path.name}:{key}")
        for key in gate.get("positive", []):
            if not isinstance(row.get(key), (int, float)) or not row[key] > 0:
                raise ValueError(f"qualification has no signal: {path.name}:{key}")
        if gate.get("contains"):
            records = [json.loads(line) for line in text.splitlines() if line]
            for required in gate["contains"]:
                if not any(all(record.get(key) == value for key, value in required.items())
                           for record in records):
                    raise ValueError(f"qualification trajectory is incomplete: {path.name}")
    return True


def _finite_gate_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value)))


def validate_plan(plan):
    if plan.get("format") != FORMAT or not re.fullmatch(r"[A-Za-z0-9_-]+", plan["alias"]):
        raise ValueError("invalid persistent RL queue plan")
    if not plan["stages"] or len({s["name"] for s in plan["stages"]}) != len(plan["stages"]):
        raise ValueError("stage names must be distinct")
    if Path(plan["source"]).joinpath("SOURCE_REVISION").read_text().strip() != plan["source_revision"]:
        raise ValueError("immutable source revision differs")
    for name, digest in plan["protected_files"].items():
        if sha256_file(name) != digest:
            raise ValueError("reviewed input changed: " + name)
    for stage in plan["stages"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", stage["name"]):
            raise ValueError("invalid RL stage geometry")
        if "pythonpath" in stage and (not isinstance(stage["pythonpath"], str) or "\0" in stage["pythonpath"]):
            raise ValueError("stage Python overlay must be a path string")
        extra_env = stage.get("env", {})
        protected_env = {"CUDA_VISIBLE_DEVICES", "PYTHONPATH", "ARCHLAB_SOURCE_REVISION", "ARCHLAB_CONTAINER_IMAGE"}
        if (not isinstance(extra_env, dict) or protected_env.intersection(extra_env)
                or any(not isinstance(key, str) or not isinstance(value, str) for key, value in extra_env.items())):
            raise ValueError("stage environment cannot override execution identity")
        if stage["kind"] == "math_queue":
            evaluation_template(plan, stage)
            continue
        if stage["kind"] not in ("qualification", "production", "train") or stage["world"] not in (1, 2, 8):
            raise ValueError("invalid RL stage geometry")
        if not stage["gates"] or not stage["module"].startswith("archlab."):
            raise ValueError("each stage requires a local executor and qualification evidence")


def evaluation_template(plan, stage):
    """Validate a frozen benchmark independently of the future checkpoint save."""
    template = pinned_json(stage["template_plan"], stage["template_plan_sha256"])
    phase = template["models"][0]
    gate = stage["checkpoint_gate"]
    if (phase["variant"] != gate["variant"] or phase["kind"] != "rl"
            or template["source"] != plan["source"]
            or template["source_git_revision"] != plan["source_revision"]):
        raise ValueError("post-RL evaluation executor or model contract differs")
    if gate["variant"] == "native" and (
            template["model"] != gate["publisher_snapshot"]
            or template["tokenizer"] != gate["publisher_snapshot"]):
        raise ValueError("native evaluation requires its verified publisher model and tokenizer")
    sampling = template["sampling"]
    expected = dict(samples_per_problem=4, temperature=0.6, top_p=0.95,
                    top_k=0, context_limit=131072, max_new_tokens=131072,
                    seed=20261002, budget_policy="native_context_minus_prompt", repetition_watchdog=False)
    if (template.get("evaluation_backend") != "eval_pipeline"
            or template.get("decode_mode") != "native_eager"
            or any(sampling.get(key) != value for key, value in expected.items())
            or sampling.get("eos_token_ids") != [151643, 151645]):
        raise ValueError("post-RL evaluation must retain the sealed full-context AIME26 protocol")
    # The final receipt cannot be pinned until training publishes it. All other
    # source, dataset and dependency identities are already independently sealed.
    checked = copy.deepcopy(template)
    checked["models"][0].pop("checkpoint", None)
    checked["models"][0].pop("checkpoint_receipt_sha256", None)
    validate_math_plan(checked)
    if not stage.get("plan_path") or not stage.get("checkpoint_cache"):
        raise ValueError("post-RL evaluation requires durable plan and local checkpoint cache")
    if not 1 <= stage.get("max_attempts", 2) <= 3:
        raise ValueError("evaluation retry bound must be between one and three")
    return template


class PersistentRLQueue:
    def __init__(self, plan, *, workers=None, setup=None):
        validate_plan(plan)
        self.plan = plan
        self.output = Path(plan["output"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.workers = workers or SSHWorkers(plan)
        self.setup = setup or self.setup_node
        self.state = (json.loads((self.output / "STATE.json").read_text())
                      if (self.output / "STATE.json").exists()
                      else dict(stage=0, status="waiting_for_allocation", workers={}))
        self.plan_sha = sha256_file(plan["plan_path"])
        if self.state.get("plan_sha256", self.plan_sha) != self.plan_sha:
            raise ValueError("queue plan changed during execution")
        self.state["plan_sha256"] = self.plan_sha
        self.evaluations = {}

    def publish(self, status, **fields):
        self.state.update(status=status, updated_at=time.time(), **fields)
        atomic_write_json(self.output / "STATE.json", self.state)
        return status

    def setup_node(self):
        plan = self.plan
        result = subprocess.run(
            [plan["cli"], "pai-dlc", "GetJob", "--region", "cn-zhongwei", "--endpoint",
             "pai-dlc.cn-zhongwei.aliyuncs.com", "--JobId", plan["job_id"], "--NeedDetail", "true"],
            capture_output=True, text=True, timeout=40,
        )
        result.check_returncode()
        job = json.loads(result.stdout)
        atomic_write_json(self.output / "JOB_STATUS.json", dict(
            job=job["JobId"], status=job["Status"], time=time.time(),
            reason=job.get("ReasonMessage"), max_runtime_minutes=job.get("JobMaxRunningTimeMinutes"),
        ))
        if job["Status"] in TERMINAL:
            raise ValueError("allocation ended; it will not be resubmitted: " + job["Status"])
        if job["Status"] != "Running":
            return False
        specs = job["JobSpecs"]
        pods = [p for p in job["Pods"] if p.get("Type", "").lower() != "aimaster"]
        if (len(specs) != 1 or specs[0]["PodCount"] != 1 or len(pods) != 1
                or specs[0]["Image"] != plan["container_image"]):
            raise ValueError("owned allocation geometry or container differs")
        pod = pods[0]
        config = Path(plan.get("ssh_config", "/root/.ssh/config"))
        text = config.read_text() if config.exists() else ""
        alias = plan["alias"]
        block = (f"Host {alias}\n  HostName {pod['PodIp']}\n  User root\n"
                 f"  HostKeyAlias {pod['PodId']}\n  StrictHostKeyChecking accept-new\n")
        pattern = rf"(?m)^Host {re.escape(alias)}\n(?:(?!Host ).*(?:\n|$))*"
        updated = re.sub(pattern, lambda _: block, text) if re.search(pattern, text) else text.rstrip()+"\n\n"+block
        if updated != text:
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(updated)
        prior = self.state.get("pod_uid")
        if prior and prior != pod["PodUid"] and self.state["workers"]:
            raise ValueError("pod replaced during an owned stage; explicit checkpoint recovery required")
        self.state.update(pod=pod["PodId"], pod_uid=pod["PodUid"])
        runtime_path = self.output / "RUNTIME.json"
        if not runtime_path.exists() or prior != pod["PodUid"]:
            command = "python3 -c " + shlex.quote(BOOTSTRAP)
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", alias, command],
                input=json.dumps(dict(python=plan["python"], **plan["runtime"])),
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode:
                raise RuntimeError("owned-node runtime setup failed: " + result.stderr[-1500:])
            atomic_write_json(runtime_path, json.loads(result.stdout))
        return True

    def tick(self):
        if self.state["status"] in ("blocked", "complete"):
            return self.state["status"]
        stage = self.plan["stages"][self.state["stage"]]
        # Dependent controllers must not SSH, bootstrap a runtime or claim GPUs
        # while the prior production controller is still completing its work.
        if not gate_ready(stage.get("preconditions", [])):
            return self.publish("waiting_for_preconditions", active_stage=stage["name"])
        evaluation = self.prepare_evaluation(stage) if stage["kind"] == "math_queue" else None
        if stage["kind"] == "math_queue" and evaluation is None:
            return self.publish("waiting_final_rl", active_stage=stage["name"])
        if not self.setup():
            return self.publish("waiting_for_allocation")
        if evaluation is not None:
            status = evaluation.tick()
            self.state["workers"][stage["name"]]["status"] = status
            if status in ("failed", "memory_overflow"):
                return self.publish("blocked", active_stage=stage["name"],
                                    error="post-RL evaluation " + status)
            if status != "complete":
                return self.publish(status, active_stage=stage["name"])
            return self.advance(stage)
        env = dict(self.plan["env"], **stage.get("env", {}))
        env.update(ARCHLAB_SOURCE_REVISION=self.plan["source_revision"],
                   ARCHLAB_CONTAINER_IMAGE=self.plan["container_image"],
                   CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(stage["world"])),
                   PYTHONPATH=self.plan["source"]+"/src:"+stage.get("pythonpath", self.plan["pythonpath"]))
        owner = self.state["workers"].get(stage["name"])
        request = dict(action="inspect" if owner else "launch",
                       slot=str(self.output / stage["name"]), marker=stage["marker"],
                       cwd=self.plan["source"], env=env, wait_for_modules=[], extra_logs=[],
                       argv=[self.plan["python"], "-m", "torch.distributed.run", "--nnodes=1",
                             f"--nproc_per_node={stage['world']}", "--node_rank=0",
                             "--master_addr=127.0.0.1", f"--master_port={stage['port']}",
                             "-m", stage["module"], *stage["args"]])
        if not owner:
            self.state["workers"][stage["name"]] = dict(status="launch_intent")
            self.publish("launching", active_stage=stage["name"])
        result = self.workers.request(self.plan["alias"], request)
        self.state["workers"][stage["name"]] = {k: v for k, v in result.items() if k != "log_tail"}
        if result["status"] == "running":
            return self.publish("running", active_stage=stage["name"])
        if not gate_ready(stage["gates"]):
            if result.get("marker"):
                # The worker's shared mount can observe completed receipts
                # before this controller's negative metadata cache expires.
                # Keep adopting the finished stage; never relaunch its work.
                return self.publish("waiting_for_marker_visibility", active_stage=stage["name"])
            raise ValueError("owned stage exited without complete qualification: " + stage["name"])
        return self.advance(stage)

    def advance(self, stage):
        self.state["stage"] += 1
        return self.publish("complete" if self.state["stage"] == len(self.plan["stages"]) else "ready",
                            completed_stage=stage["name"])

    def prepare_evaluation(self, stage):
        """Pin a verified final save, then reuse the existing sixteen-shard queue."""
        state = self.state["workers"].setdefault(stage["name"], {})
        proof = state.get("final_checkpoint")
        if proof is None:
            proof = final_checkpoint(stage["checkpoint_gate"])
            if proof is None:
                return None
            state["final_checkpoint"] = proof
            self.publish("verified_final_rl", active_stage=stage["name"])
        else:
            root = Path(stage["checkpoint_gate"]["rl_root"])
            pinned_json(root / "FINISHED.json", proof["finished_sha256"])
            pinned_json(Path(proof["checkpoint"]) / "COMPLETE.json", proof["receipt_sha256"])
            pinned_json(root / "MLFLOW.json", proof["tracking_sha256"])
        template = evaluation_template(self.plan, stage)
        receipt, finished = proof["receipt"], proof["finished"]
        phase = template["models"][0]
        phase.update(checkpoint=proof["checkpoint"], checkpoint_receipt_sha256=proof["receipt_sha256"],
                     step=finished["step"], applied_rl_updates=finished["applied_updates"],
                     current_rl_phase_start=receipt["phase_start"],
                     current_rl_phase_nominal_updates=finished["step"] - receipt["phase_start"])
        path = Path(stage["plan_path"])
        if path.exists():
            if json.loads(path.read_text()) != template:
                raise ValueError("materialized post-RL evaluation plan changed")
        else:
            atomic_write_json(path, template, allow_nan=False)
        digest = sha256_file(path)
        state.update(plan=str(path), plan_sha256=digest, mlflow_run_id=proof["mlflow_run_id"])
        validate_math_plan(template)
        config = dict(format=MATH_QUEUE_FORMAT, output=str(self.output / stage["name"]),
                      hosts=[self.plan["alias"]], python=self.plan["python"],
                      pythonpath=stage.get("pythonpath", self.plan["pythonpath"]),
                      env={**self.plan["env"], **stage.get("env", {}),
                           "ARCHLAB_CONTAINER_IMAGE": self.plan["container_image"]},
                      checkpoint_cache=stage["checkpoint_cache"], wait_for_training_exit=True,
                      max_attempts=stage.get("max_attempts", 2),
                      stages=[dict(name=stage["name"], plan=str(path), plan_sha256=digest)])
        if stage["name"] not in self.evaluations:
            self.evaluations[stage["name"]] = QueueController(config, workers=self.workers)
        self.publish("ready_evaluation", active_stage=stage["name"])
        return self.evaluations[stage["name"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    plan["plan_path"] = str(args.plan.resolve())
    queue = PersistentRLQueue(plan)
    with (queue.output / "LOCK").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_write_json(queue.output / "CONTROLLER.json", dict(pid=os.getpid(),
            start_ticks=Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
            job_id=plan["job_id"], plan_sha256=queue.plan_sha, source_revision=plan["source_revision"]))
        while not (queue.output / "STOP").exists():
            try:
                status = queue.tick()
                if status in ("complete", "blocked"):
                    return
            except (ValueError, AssertionError, KeyError) as error:
                queue.publish("blocked", error=str(error))
                return
            except Exception as error:
                queue.publish("retrying", error=str(error)[-1500:])
            time.sleep(30)


if __name__ == "__main__":
    main()
