"""Durable sequential, sixteen-shard evaluation of sealed Limite plans.

The CPU controller never imports a model or changes the benchmark executor.
Its SSH helper adopts exact argv/environment matches, including launches whose
acknowledgement was lost. Opt-in stale-heartbeat recovery signals only the exact
owned evaluation process after rechecking its unchanged heartbeat and identity.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.automodel.limite_adapter_queue import evaluation_ready
from archlab.evaluation.limite_math import benchmark_seed

FORMAT = "archlab-limite-math-queue-v1"
WORKER_MODULE = "archlab.automodel.limite_math_benchmark"
OOM = re.compile(r"CUDA out of memory|(?:torch\.)?OutOfMemoryError|CUDA error: out of memory", re.I)
RL_CONTRACT_KEYS = ("phase_start", "math_protocol", "curriculum_sha256", "rl_split_sha256")

# Executed by the configured Python on a worker host. stdin is structured data;
# shell text never contains a plan, output, environment value, or credential.
REMOTE = r'''
import ctypes, fcntl, hashlib, json, os, pathlib, platform, signal, subprocess, sys, tempfile, time
q = json.load(sys.stdin)
slot = pathlib.Path(q["slot"])
slot.mkdir(parents=True, exist_ok=True)
def publish(path, value):
    fd, name = tempfile.mkstemp(prefix=".owner-", dir=slot)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)
def identity(pid):
    root = pathlib.Path("/proc") / str(pid)
    try:
        argv = root.joinpath("cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        stat = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z": return None
        environ = dict(x.split("=", 1) for x in root.joinpath("environ").read_bytes()
                       .decode().split("\0") if "=" in x)
        return dict(pid=int(pid), start_ticks=int(stat[19]), argv=argv,
                    env={key: environ.get(key) for key in q["env"]})
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (PermissionError, UnicodeError):
        return dict(pid=int(pid), unreadable=True)
def pidfd_syscall(number, *args):
    # Some portable CPython builds omit the wrappers even on Linux. Use the
    # same kernel pidfd syscalls on the two supported Linux worker architectures.
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("owned stale recovery requires Linux pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result
with slot.joinpath("LOCK").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    matches, blockers = [], []
    for path in pathlib.Path("/proc").iterdir():
        if not path.name.isdigit(): continue
        candidate = identity(path.name)
        if candidate is None: continue
        if any(module in candidate.get("argv", []) for module in q.get("wait_for_modules", [])):
            blockers.append(candidate["pid"])
        if candidate.get("argv") != q["argv"]: continue
        if candidate["env"] != q["env"]:
            raise RuntimeError("matching worker argv has a different environment")
        matches.append(candidate)
    if len(matches) > 1:
        raise RuntimeError("duplicate workers; refusing any additional launch")
    if matches:
        answer = dict(matches[0], status="running", adopted=True)
        if q["action"] == "recover_stale":
            expected = q["expected_identity"]
            current = pathlib.Path(q["heartbeat"]["path"])
            def stale_owned():
                if q["heartbeat"]["minimum_age"] < 1800: return False
                observed = identity(expected["pid"])
                if observed != expected or observed != matches[0]: return False
                if pathlib.Path(q["marker"]).exists() or not current.exists(): return False
                raw = current.read_bytes()
                if hashlib.sha256(raw).hexdigest() != q["heartbeat"]["sha256"]: return False
                heartbeat = json.loads(raw)
                return (heartbeat.get("pid") == expected["pid"]
                        and heartbeat.get("status") == "generating"
                        and time.time() - heartbeat["time"] >= q["heartbeat"]["minimum_age"])
            if stale_owned():
                # pidfds prevent an exited PID being recycled between verification
                # and signaling. An unsupported host blocks recovery, never falls
                # back to signaling an unbound numeric PID.
                fd = (os.pidfd_open(expected["pid"]) if hasattr(os, "pidfd_open")
                      else pidfd_syscall(434, expected["pid"], 0))
                try:
                    if stale_owned():
                        if hasattr(signal, "pidfd_send_signal"):
                            signal.pidfd_send_signal(fd, signal.SIGTERM)
                        else:
                            pidfd_syscall(424, fd, signal.SIGTERM, ctypes.c_void_p(), 0)
                        answer.update(status="stop_requested", heartbeat_sha256=q["heartbeat"]["sha256"])
                    else:
                        answer["recovery_refused"] = "identity_or_progress_changed"
                finally:
                    os.close(fd)
            else:
                answer["recovery_refused"] = "identity_or_progress_changed"
        publish(slot / "OWNER.json", answer)
    elif blockers:
        answer = dict(status="waiting_for_training_exit", pids=blockers)
    elif q["action"] == "launch":
        if pathlib.Path(q["marker"]).exists():
            answer = dict(status="exited", marker=True)
        else:
            env = os.environ.copy(); env.update(q["env"])
            with slot.joinpath("worker.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(q["argv"], cwd=q["cwd"], env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, close_fds=True)
            answer = identity(process.pid)
            if answer is None:
                answer = dict(pid=process.pid, status="exited")
            else:
                answer.update(status="running", adopted=False)
            publish(slot / "OWNER.json", answer)
    else:
        answer = dict(status="exited")
    tails = []
    for log in [slot / "worker.log", *map(pathlib.Path, q.get("extra_logs", []))]:
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 65536))
                tails.append(stream.read().decode(errors="replace"))
    answer["log_tail"] = "\n".join(tails)
    answer["marker"] = pathlib.Path(q["marker"]).exists()
    print(json.dumps(answer))
'''


def read_json(path):
    return json.loads(Path(path).read_text())


def pinned_json(path, digest):
    if sha256_file(path) != digest:
        raise ValueError(f"reviewed file changed: {path}")
    return read_json(path)


def validate_queue(config):
    if config.get("format") != FORMAT:
        raise ValueError("unsupported queue format")
    hosts = config["hosts"]
    if len(hosts) not in (1, 2) or len(set(hosts)) != len(hosts) or any(not isinstance(h, str) or not h for h in hosts):
        raise ValueError("queue requires one or two distinct eight-GPU hosts")
    stages = config["stages"]
    if not stages or len({s["name"] for s in stages}) != len(stages):
        raise ValueError("stage names must be unique")
    if not 1 <= config.get("max_attempts", 2) <= 3:
        raise ValueError("worker retry bound must be between one and three")
    if not 1 <= config.get("poll_seconds", 15) <= 300:
        raise ValueError("invalid poll interval")
    if type(config.get("backfill_static_stages", False)) is not bool:
        raise ValueError("backfill_static_stages must be boolean")
    if type(config.get("workers_per_gpu", 2)) is not int or not 1 <= config.get("workers_per_gpu", 2) <= 2:
        raise ValueError("at most two benchmark workers may share a GPU")
    if "gpu_slots" in config:
        slots = config["gpu_slots"]
        if (not isinstance(slots, dict) or set(slots) != set(hosts)
                or any(not isinstance(gpus, list) or len(gpus) != len(set(gpus))
                       or any(type(gpu) is not int or not 0 <= gpu < 8 for gpu in gpus)
                       for gpus in slots.values()) or not any(slots.values())):
            raise ValueError("gpu_slots must list distinct allowed GPU IDs for each host")
        if not config.get("backfill_static_stages"):
            raise ValueError("explicit GPU slots require opt-in backfill scheduling")
    if "stale_heartbeat_seconds" in config:
        age = config["stale_heartbeat_seconds"]
        if type(age) not in (int, float) or not 1800 <= age <= 86400:
            raise ValueError("stale recovery requires at least 1800 seconds without progress")
    restarts = config.get("max_stale_restarts", 1)
    if type(restarts) is not int or not 1 <= restarts <= 2:
        raise ValueError("stale recovery must allow only one or two restarts")
    for stage in stages:
        if ("plan" in stage) == ("template_plan" in stage):
            raise ValueError("stage needs a frozen plan or a final-RL template")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", stage["name"]):
            raise ValueError("unsafe stage name")
        if "template_plan" in stage and stage["final_variant"] not in ("normal", "simplicial"):
            raise ValueError("invalid final RL variant")
    forbidden = {"CUDA_VISIBLE_DEVICES", "PYTHONPATH"}
    if forbidden.intersection(config.get("env", {})):
        raise ValueError("worker source and GPU environment are controller-owned")
    if config.get("final_rl_gate_mode", "both") not in ("both", "own_variant"):
        raise ValueError("unknown final RL gate mode")
    if ("wait_for_evaluations" in config) != ("wait_for_evaluations_sha256" in config):
        raise ValueError("evaluation gate path and SHA256 must be supplied together")
    if "wait_for_evaluations" in config:
        pinned_json(config["wait_for_evaluations"], config["wait_for_evaluations_sha256"])


def validate_plan(plan):
    if plan.get("format") != "archlab-limite-math-evaluation-v1" or len(plan["models"]) != 1:
        raise ValueError("each stage must evaluate exactly one model")
    bundle = Path(plan["bundle"])
    manifest = pinned_json(bundle / "MANIFEST.json", plan["manifest_sha256"])
    if sha256_file(bundle / "cases.jsonl") != manifest["cases_sha256"]:
        raise ValueError("sealed cases changed")
    cases = [json.loads(line) for line in (bundle / "cases.jsonl").read_text().splitlines() if line]
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("duplicate benchmark cases")
    source = Path(plan["source"])
    if source.joinpath("SOURCE_REVISION").read_text().strip() != plan["source_git_revision"]:
        raise ValueError("sealed executor revision changed")
    root = source / "src" / "archlab"
    actual = {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*.py"))}
    if actual != plan["source_file_sha256"]:
        raise ValueError("sealed executor files changed")
    if hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest() != plan["implementation_sha256"]:
        raise ValueError("executor aggregate identity differs")
    phase = plan["models"][0]
    if phase.get("checkpoint"):
        pinned_json(Path(phase["checkpoint"]) / "COMPLETE.json", phase["checkpoint_receipt_sha256"])
    if type(plan["sampling"]["samples_per_problem"]) is not int or plan["sampling"]["samples_per_problem"] < 1:
        raise ValueError("invalid sample count")
    if plan.get("evaluation_backend") == "eval_pipeline":
        from archlab.evaluation.limite_pipeline import validate_pipeline_contract

        validate_pipeline_contract(plan, cases)
    return cases


def verify_shard(plan, cases, shard):
    """Check every expected preserved/new response, not just a worker marker."""
    phase = plan["models"][0]
    output = Path(plan["output"])
    marker = read_json(output / f"SHARD-{shard}-COMPLETE.json")
    assigned = cases[shard::16]
    expected_marker = dict(shard=shard, models=[phase["name"]], problems=len(assigned),
                           samples_per_problem=plan["sampling"]["samples_per_problem"],
                           case_limit=None, implementation_sha256=plan["implementation_sha256"])
    if any(marker.get(key) != value for key, value in expected_marker.items()):
        raise ValueError(f"incomplete or different shard marker {shard}")
    expected = {(case["id"], sample): case for case in assigned
                for sample in range(plan["sampling"]["samples_per_problem"])}
    path = output / phase["name"] / f"shard-{shard}" / "records.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
    seen = set()
    for row in records:
        key = (row["problem_id"], row["sample_index"])
        if key not in expected or key in seen:
            raise ValueError(f"unexpected/duplicate response in shard {shard}")
        case = expected[key]
        accepted = set(plan.get("accepted_record_implementation_sha256", [])) | {plan["implementation_sha256"]}
        if row.get("source_implementation_sha256") not in accepted:
            raise ValueError(f"response executor identity differs in shard {shard}")
        checks = dict(model=phase["name"], task=case["task"], expected_answer=case["answer"],
                      seed=benchmark_seed(plan["sampling"]["seed"], *key))
        if any(row.get(k) != v for k, v in checks.items()):
            raise ValueError(f"response identity differs in shard {shard}")
        seen.add(key)
    if seen != set(expected):
        raise ValueError(f"missing responses in shard {shard}")
    return dict(shard=shard, records=len(records), marker_sha256=sha256_file(output / f"SHARD-{shard}-COMPLETE.json"),
                records_sha256=sha256_file(path) if path.exists() else None)


def final_checkpoint(gate):
    """Return None while training is unfinished; validate the exact final save."""
    root = Path(gate["rl_root"])
    finished_path = root / "FINISHED.json"
    if not finished_path.exists():
        return None
    finished_bytes = finished_path.read_bytes()
    finished = json.loads(finished_bytes)
    if finished.get("status") != "complete":
        return None
    if finished["step"] != gate["target_step"]:
        raise ValueError("RL finished at a different target step")
    checkpoint = root / "checkpoints" / f"step-{finished['step']:07d}"
    receipt_path = checkpoint / "COMPLETE.json"
    if not receipt_path.exists():
        return None
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    contract = gate["checkpoint_contract"]
    if set(contract) != set(RL_CONTRACT_KEYS) or any(receipt.get(key) != contract[key] for key in RL_CONTRACT_KEYS):
        raise ValueError("final RL protocol/curriculum/split/phase differs")
    if (receipt["step"] != finished["step"] or receipt.get("trainable_mode") != "full"
            or finished.get("trainable_mode") != "full"
            or receipt["source_revision"] != gate["source_revision"]):
        raise ValueError("final RL checkpoint lineage differs")
    native = gate["variant"] == "native"
    if native:
        if (receipt.get("model_kind") != "native" or finished.get("model_kind") != "native"
                or receipt.get("publisher_snapshot") != gate["publisher_snapshot"]
                or receipt.get("publisher_identity") != gate["publisher_identity"]
                or finished.get("publisher_identity") != gate["publisher_identity"]
                or "adapter" in receipt or "warmup_checkpoint" in receipt):
            raise ValueError("final native RL publisher lineage differs")
        if (gate["publisher_identity"].get("repo") not in {
                "paradigma-inc/limite-1b-base", "paradigma-inc/limite-1b-violetto"}
                or not gate["publisher_identity"].get("revision")
                or not gate["publisher_identity"].get("receipt_sha256")
                or type(finished.get("applied_updates")) is not int
                or finished["applied_updates"] <= 0):
            raise ValueError("final native RL publisher identity or update evidence is incomplete")
        required_files = {"model.pt", "optimizer.pt", "rng.pt", "rl_state.pt", "trainer_state.json"}
        if {"adapter.pt", "backbone.pt"}.intersection(receipt["files"]):
            raise ValueError("final native RL save contains adapter payloads")
    else:
        adapter = receipt["adapter"]
        if (adapter["variant"] != gate["variant"] or adapter["attention_backend"] != "tilelang"
                or receipt["warmup_checkpoint"] != gate["warmup_checkpoint"]):
            raise ValueError("final RL checkpoint lineage differs")
        parent = read_json(Path(gate["warmup_checkpoint"]) / "COMPLETE.json")
        if (parent.get("tokens") != 10_000_000_000 or parent.get("trainable_mode") != "full"
                or receipt.get("base_snapshot_sha256") != parent.get("base_snapshot_sha256")
                or finished.get("base_snapshot_sha256") != receipt.get("base_snapshot_sha256")):
            raise ValueError("final RL publisher/SFT lineage differs")
        required_files = {"adapter.pt", "backbone.pt", "optimizer.pt", "rng.pt", "rl_state.pt", "trainer_state.json"}
    tracking_path = root / "MLFLOW.json"
    tracking_bytes = tracking_path.read_bytes()
    tracking = json.loads(tracking_bytes)
    if tracking["run_id"] != gate["mlflow_run_id"]:
        raise ValueError("different RL experiment run")
    files = receipt["files"]
    if not required_files.issubset(files):
        raise ValueError("final full RL save lacks model/optimizer/RNG state")
    for name, identity in files.items():
        if Path(name).name != name or name in (".", "..", "COMPLETE.json", "VERIFIED.json"):
            raise ValueError("checkpoint payload must be a direct filename")
        path = checkpoint / name
        digest = identity["sha256"] if isinstance(identity, dict) else identity
        size = identity.get("bytes", identity.get("size")) if isinstance(identity, dict) else None
        if (size is not None and path.stat().st_size != size) or sha256_file(path) != digest:
            raise ValueError(f"final RL payload changed: {name}")
    if (receipt_path.read_bytes() != receipt_bytes or finished_path.read_bytes() != finished_bytes
            or tracking_path.read_bytes() != tracking_bytes):
        raise ValueError("final RL metadata changed during payload verification")
    return dict(checkpoint=str(checkpoint), receipt=receipt, finished=finished,
                receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                finished_sha256=hashlib.sha256(finished_bytes).hexdigest(),
                tracking_sha256=hashlib.sha256(tracking_bytes).hexdigest(), mlflow_run_id=tracking["run_id"])


class SSHWorkers:
    def __init__(self, config):
        self.config = config

    def request(self, host, request):
        command = " ".join(shlex.quote(x) for x in [self.config["python"], "-c", REMOTE])
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
                                input=json.dumps(request), capture_output=True, text=True, timeout=45)
        if result.returncode:
            raise RuntimeError(f"worker inspection/launch failed on {host}: {result.stderr[-4000:]}")
        return json.loads(result.stdout)


class QueueController:
    def __init__(self, config, workers=None):
        validate_queue(config)
        self.config = config
        self.workers = workers or SSHWorkers(config)
        self.output = Path(config["output"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        path = self.output / "STATE.json"
        self.state = read_json(path) if path.exists() else dict(format=FORMAT, config_sha256=self.config_sha,
                                                             stage=0, status="ready", stages={})
        if self.state["config_sha256"] != self.config_sha:
            raise ValueError("queue configuration changed during execution")
        self.cases = {}
        watched = self.state.setdefault("watched_logs", {})
        for filename in config.get("watch_logs", []):
            path = Path(filename)
            if filename not in watched:
                stat = path.stat() if path.exists() else None
                watched[filename] = dict(offset=stat.st_size if stat else 0,
                                         inode=[stat.st_dev, stat.st_ino] if stat else None, tail="")
        self.save()

    def save(self):
        atomic_write_json(self.output / "STATE.json", self.state, allow_nan=False)

    def event(self, kind, **fields):
        with (self.output / "EVENTS.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(time=time.time(), event=kind) | fields) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def memory_overflow(self, **fields):
        event = dict(time=time.time(), automatic_retry=False) | fields
        atomic_write_json(self.output / "OOM_EVENT.json", event, allow_nan=False)
        self.event("memory_overflow", **event)
        self.state["status"] = "memory_overflow"
        self.save()
        return "memory_overflow"

    def watch_memory(self):
        """Observe only new log bytes; do not change or signal either RL job."""
        for filename, cursor in self.state["watched_logs"].items():
            path = Path(filename)
            if not path.exists():
                continue
            stat = path.stat()
            inode = [stat.st_dev, stat.st_ino]
            if cursor["inode"] != inode or stat.st_size < cursor["offset"]:
                cursor.update(offset=0, inode=inode, tail="")
            with path.open("rb") as stream:
                stream.seek(cursor["offset"])
                while chunk := stream.read(65536):
                    text = cursor["tail"] + chunk.decode(errors="replace")
                    cursor["offset"] = stream.tell()
                    cursor["tail"] = text[-256:]
                    if OOM.search(text):
                        return self.memory_overflow(scope="watched_training_log", log=filename,
                                                    offset=cursor["offset"], excerpt=text[-8192:])
            cursor["inode"] = inode
        return None

    def plan_for(self, stage, state):
        if state.get("plan"):
            return pinned_json(state["plan"], state["plan_sha256"])
        if stage.get("wait_for_final_rl_variant"):
            variant = stage["wait_for_final_rl_variant"]
            gate = next(g for g in self.config["final_rl_gates"] if g["variant"] == variant)
            proof = final_checkpoint(gate)
            if proof is None:
                self.state["status"] = "waiting_final_rl"
                return None
            state["start_after_rl_proof"] = proof
        if "plan" in stage:
            path, digest = stage["plan"], stage["plan_sha256"]
            plan = pinned_json(path, digest)
        else:
            template = pinned_json(stage["template_plan"], stage["template_plan_sha256"])
            gates = self.config["final_rl_gates"]
            expected = ({stage["final_variant"]} if self.config.get("final_rl_gate_mode") == "own_variant"
                        else {"normal", "simplicial"})
            if {g["variant"] for g in gates} != expected or len(gates) != len(expected):
                raise ValueError("final RL gates differ from the explicit queue dependency contract")
            # Avoid reading/hashing one final save repeatedly while its peer is live.
            for gate in gates:
                path = Path(gate["rl_root"]) / "FINISHED.json"
                if not path.exists() or read_json(path).get("status") != "complete":
                    self.state["status"] = "waiting_final_rl"
                    return None
            proofs = self.state.get("final_rl_proofs")
            if proofs is None:
                proofs = {gate["variant"]: final_checkpoint(gate) for gate in gates}
            else:
                for gate in gates:
                    proof = proofs[gate["variant"]]
                    if (sha256_file(Path(proof["checkpoint"]) / "COMPLETE.json") != proof["receipt_sha256"]
                            or read_json(Path(gate["rl_root"]) / "FINISHED.json") != proof["finished"]):
                        raise ValueError("verified final RL metadata changed")
            if any(proof is None for proof in proofs.values()):
                self.state["status"] = "waiting_final_rl"
                return None
            self.state["final_rl_proofs"] = proofs
            self.save()
            proof = proofs[stage["final_variant"]]
            receipt, finished = proof["receipt"], proof["finished"]
            plan = copy.deepcopy(template)
            phase = plan["models"][0]
            if phase["variant"] != stage["final_variant"] or phase["kind"] != "rl":
                raise ValueError("final RL template variant differs")
            gate = next(g for g in gates if g["variant"] == stage["final_variant"])
            template_receipt = pinned_json(Path(phase["checkpoint"]) / "COMPLETE.json", phase["checkpoint_receipt_sha256"])
            if any(template_receipt.get(key) != gate["checkpoint_contract"][key] for key in RL_CONTRACT_KEYS):
                raise ValueError("final RL template protocol/curriculum/split/phase differs")
            name = f"rl-{stage['final_variant']}-final-step{finished['step']}"
            phase.update(name=name, step=finished["step"], checkpoint=proof["checkpoint"],
                         checkpoint_receipt_sha256=proof["receipt_sha256"],
                         applied_rl_updates=finished["applied_updates"],
                         current_rl_phase_start=receipt["phase_start"],
                         current_rl_phase_nominal_updates=finished["step"] - receipt["phase_start"])
            plan["output"] = f"{stage['output_prefix']}-step{finished['step']}"
            path = stage["plan_path"]
            if Path(path).exists():
                if read_json(path) != plan:
                    raise ValueError("materialized final plan differs")
            else:
                atomic_write_json(path, plan, allow_nan=False)
            digest = sha256_file(path)
            self.register_final(plan, path, proof["mlflow_run_id"])
            atomic_write_json(self.output / f"FINAL-{stage['final_variant']}.json", proofs, allow_nan=False)
        state.update(plan=str(path), plan_sha256=digest, slots={})
        self.cases[stage["name"]] = validate_plan(plan)
        self.save()
        return plan

    def register_final(self, plan, path, run_id):
        registry_path = self.config.get("report_registry")
        if not registry_path:
            return
        path_registry = Path(registry_path)
        with path_registry.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            registry = read_json(path_registry)
            entry = dict(name=plan["models"][0]["name"], plan=str(path), mlflow_run_id=run_id)
            previous = [row for row in registry["models"] if row["name"] == entry["name"]]
            if previous and previous != [entry]:
                raise ValueError("final registry name already denotes another plan")
            if not previous:
                registry["models"].append(entry)
                atomic_write_json(path_registry, registry, allow_nan=False)

    def worker_request(self, stage, plan, shard, action):
        source = str(Path(plan["source"]) / "src")
        overlay = self.config.get("pythonpath", "")
        env = {str(k): str(v) for k, v in self.config.get("env", {}).items()}
        env["ARCHLAB_SOURCE_REVISION"] = plan["source_git_revision"]
        slot = self.state["stages"][stage["name"]]["slots"].get(str(shard), {})
        env.update(CUDA_VISIBLE_DEVICES=str(slot.get("gpu", shard % 8)),
                   PYTHONPATH=source + (":" + overlay if overlay else ""))
        template = stage.get("worker_log_template")
        return dict(action=action, slot=str(self.output / "slots" / stage["name"] / str(shard)),
                    marker=str(Path(plan["output"]) / f"SHARD-{shard}-COMPLETE.json"),
                    wait_for_modules=(["archlab.automodel.limite_adapter_rl", "archlab.automodel.limite_full_rl"]
                                      if self.config.get("wait_for_training_exit") else []),
                    extra_logs=[template.format(shard=shard)] if template else [],
                    cwd=plan["source"], env=env,
                    argv=[self.config["python"], "-m",
                          ("archlab.automodel.limite_pipeline_benchmark"
                           if plan.get("evaluation_backend") == "eval_pipeline" else WORKER_MODULE), "--plan",
                          self.state["stages"][stage["name"]]["plan"], "--shard", str(shard),
                          "--shards", "16", "--checkpoint-cache", self.config["checkpoint_cache"]])

    def _available_gpu(self, slot):
        """Reserve capacity using durable launch intents as well as live workers."""
        allowed = self.config.get("gpu_slots", {host: list(range(8)) for host in self.config["hosts"]})
        used = {(host, gpu): 0 for host, gpus in allowed.items() for gpu in gpus}
        for stage_state in self.state["stages"].values():
            for other in stage_state.get("slots", {}).values():
                if other is slot or other.get("status") not in ("running", "launch_intent", "stop_requested"):
                    continue
                key = (other.get("host"), other.get("gpu"))
                if key in used:
                    used[key] += 1
        capacity = self.config.get("workers_per_gpu", 2)
        if "host" in slot:
            key = (slot["host"], slot["gpu"])
            return key if key in used and used[key] < capacity else None
        choices = [key for key, count in used.items() if count < capacity]
        return min(choices, key=lambda key: used[key]) if choices else None

    def _recover_stale(self, stage, plan, shard, slot, request, observed):
        timeout = self.config.get("stale_heartbeat_seconds")
        if timeout is None:
            return None
        previous = slot.get("stale_stop")
        if previous and previous["pid"] == observed["pid"] and previous["start_ticks"] == observed["start_ticks"]:
            if time.time() - previous["time"] > 180:
                slot["status"] = "stale_termination_failed"
                self.state["status"] = "failed"
                self.event("stale_termination_failed", stage=stage["name"], shard=shard)
                self.save()
                return "failed"
            slot["status"] = "stop_requested"
            return "stop_requested"
        current = Path(plan["output"]) / plan["models"][0]["name"] / f"shard-{shard}" / "CURRENT.json"
        if not current.exists():
            return None
        raw = current.read_bytes()
        heartbeat = json.loads(raw)
        timestamp = heartbeat.get("time")
        if (heartbeat.get("status") != "generating" or heartbeat.get("pid") != observed["pid"]
                or type(timestamp) not in (int, float) or not time.time() - timestamp >= timeout):
            return None
        if slot.get("stale_restarts", 0) >= self.config.get("max_stale_restarts", 1):
            slot["status"] = "stale_retry_limit"
            self.state["status"] = "failed"
            self.event("stale_retry_limit", stage=stage["name"], shard=shard)
            self.save()
            return "failed"
        recovery = dict(request, action="recover_stale",
                        expected_identity={key: observed[key] for key in ("pid", "start_ticks", "argv", "env")},
                        heartbeat=dict(path=str(current), sha256=hashlib.sha256(raw).hexdigest(), minimum_age=timeout))
        slot["stale_restarts"] = slot.get("stale_restarts", 0) + 1
        slot["stale_stop"] = dict(time=time.time(), pid=observed["pid"], start_ticks=observed["start_ticks"],
                                  heartbeat_sha256=recovery["heartbeat"]["sha256"])
        slot["status"] = "stop_requested"
        self.save()  # Lost SSH acknowledgement must not permit unbounded stop attempts.
        result = self.workers.request(slot["host"], recovery)
        slot["observed"] = result
        if result["status"] == "stop_requested":
            self.event("stale_worker_stopped", stage=stage["name"], shard=shard, **slot["stale_stop"])
            self.save()
            return "stop_requested"
        slot.pop("stale_stop")
        slot["stale_restarts"] -= 1
        slot["status"] = result["status"]
        return None

    def tick(self):
        if self.state["status"] in ("complete", "failed", "memory_overflow"):
            return self.state["status"]
        if self.watch_memory():
            return "memory_overflow"
        if "wait_for_evaluations" in self.config:
            gate = pinned_json(self.config["wait_for_evaluations"], self.config["wait_for_evaluations_sha256"])
            if not evaluation_ready(gate):
                self.state["status"] = "waiting_for_evaluations"
                self.save()
                return "waiting_for_evaluations"
        index = self.state["stage"]
        if index == len(self.config["stages"]):
            self.state["status"] = "complete"
            self.save()
            return "complete"
        result = self._tick_stage(index)
        if result != "running":
            self.save()
            return result
        if self.config.get("backfill_static_stages"):
            for later in range(index + 1, len(self.config["stages"])):
                stage = self.config["stages"][later]
                # Explicit dependencies remain barriers. Only already specified
                # independent benchmark plans may use the spare capacity.
                if "plan" not in stage or stage.get("wait_for_final_rl_variant"):
                    break
                if self.state["stages"].get(stage["name"], {}).get("status") == "complete":
                    continue
                result = self._tick_stage(later)
                if result != "running":
                    self.save()
                    return result
        while self.state["stage"] < len(self.config["stages"]):
            name = self.config["stages"][self.state["stage"]]["name"]
            if self.state["stages"].get(name, {}).get("status") != "complete":
                break
            self.state["stage"] += 1
        self.state["status"] = "running"
        self.save()
        return "running"

    def _tick_stage(self, index):
        stage = self.config["stages"][index]
        state = self.state["stages"].setdefault(stage["name"], {})
        plan = self.plan_for(stage, state)
        if plan is None:
            self.save()
            return self.state["status"]
        if stage["name"] not in self.cases:
            self.cases[stage["name"]] = validate_plan(plan)
        cases = self.cases[stage["name"]]
        if (Path(plan["output"]) / "STOP").exists():
            raise ValueError("stage output has a STOP request")
        all_done = True
        for shard in range(16):
            slot = state["slots"].setdefault(str(shard), dict(attempts=0))
            # Keep the sixteen sealed data shards and seeds unchanged when a
            # single host runs two independent batch-one workers per GPU.
            host = slot.get("host", self.config["hosts"][(shard // 8) % len(self.config["hosts"])])
            if host not in self.config["hosts"]:
                raise ValueError("existing slot host is absent from the queue")
            marker = Path(plan["output"]) / f"SHARD-{shard}-COMPLETE.json"
            if self.config.get("backfill_static_stages") and "host" not in slot:
                if marker.exists():
                    slot.update(status="complete", proof=verify_shard(plan, cases, shard))
                    continue
                available = self._available_gpu(slot)
                if available is None:
                    slot["status"] = "waiting_for_gpu"
                    all_done = False
                    continue
                host, gpu = available
                slot.update(host=host, gpu=gpu)
            request = self.worker_request(stage, plan, shard, "inspect")
            observed = self.workers.request(host, request)
            slot.update(host=host, gpu=int(request["env"]["CUDA_VISIBLE_DEVICES"]), argv=request["argv"], observed=observed)
            marker = Path(request["marker"]).exists()
            if observed["status"] == "running":
                slot["status"] = "running"
                all_done = False
                if self._recover_stale(stage, plan, shard, slot, request, observed) == "failed":
                    return "failed"
                continue
            if OOM.search(observed.get("log_tail", "")):
                return self.memory_overflow(scope="evaluation_worker", stage=stage["name"], shard=shard,
                                            host=host, gpu=slot["gpu"], plan=state["plan"],
                                            plan_sha256=state["plan_sha256"],
                                            worker_log=str(Path(request["slot"]) / "worker.log"),
                                            extra_logs=request["extra_logs"], excerpt=observed["log_tail"][-8192:])
            if marker:
                slot.update(status="complete", proof=verify_shard(plan, cases, shard))
                continue
            all_done = False
            if observed["status"] == "waiting_for_training_exit":
                slot["status"] = "waiting_for_training_exit"
                continue
            if observed.get("marker"):
                # Shared-storage negative metadata caches can lag the worker.
                # Wait for local verification, never restart completed work.
                slot["status"] = "waiting_for_marker_visibility"
                continue
            if self.config.get("backfill_static_stages") and self._available_gpu(slot) is None:
                slot["status"] = "waiting_for_gpu"
                continue
            if slot["attempts"] >= self.config.get("max_attempts", 2):
                self.state["status"] = "failed"
                slot["status"] = "retry_limit"
                self.event("worker_failed", stage=stage["name"], shard=shard)
                self.save()
                return "failed"
            slot["attempts"] += 1
            slot["status"] = "launch_intent"
            self.save()  # A lost SSH acknowledgement is adopted on the next tick.
            request["action"] = "launch"
            slot["observed"] = self.workers.request(host, request)
            slot["status"] = slot["observed"]["status"]
            self.save()
        if all_done:
            state["status"] = "complete"
            self.event("stage_complete", stage=stage["name"], plan_sha256=state["plan_sha256"])
        return "running"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--validate", action="store_true", help="metadata validation only; no SSH or launches")
    args = parser.parse_args()
    config = read_json(args.queue)
    validate_queue(config)
    if args.validate:
        for stage in config["stages"]:
            key = "plan" if "plan" in stage else "template_plan"
            validate_plan(pinned_json(stage[key], stage[key + "_sha256"]))
        return
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "CONTROLLER.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = QueueController(config)
        while True:
            try:
                status = controller.tick()
            except Exception as exc:
                controller.event("blocked", reason=repr(exc))
                controller.save()
                if args.once:
                    raise
            else:
                if status in ("complete", "failed", "memory_overflow") or args.once:
                    return
            time.sleep(config.get("poll_seconds", 15))


if __name__ == "__main__":
    main()
