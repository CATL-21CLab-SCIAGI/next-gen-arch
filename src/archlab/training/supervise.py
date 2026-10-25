"""Keep one RL arm training to its step budget without an LLM in the loop.

An arm spec (JSON, machine paths) names the GPU node, the resolved base config, the checkpoint directory, the Ray
head directory with its CPU agent pods, the immutable source snapshot and the training snapshot. Every 2 minutes:
- trainer running: if the spec's data or source differs from the last launch, stop the trainer right after its
  next saved checkpoint (verl writes the latest-iteration file last);
- trainer gone: resubmit evicted agent pods, wait for the GPUs to free, resume from the latest checkpoint.
A new data snapshot resumes the actor only (fresh dataloader, data.seed = 1000 + step); otherwise the full checkpoint
with its dataloader state resumes. Launches are recorded in <dir>/LAUNCHES.jsonl. Gives up after 3 launches without
a new checkpoint.
usage: python -m archlab.training.supervise <arm.json> [--once]
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from archlab.training import dlc_jobs

LIVE_POD_STATES = {"Running", "Creating", "Queuing", "EnvPreparing"}


def rotate_log(path: Path) -> Path | None:
    """Keep a non-empty previous log under a timestamped name; never overwrite an earlier rotation."""
    if not (path.exists() and path.stat().st_size):
        return None
    stem = path.with_name(f"{path.stem}-{time.strftime('%m%d-%H%M%S')}")
    target, n = stem.with_suffix(path.suffix), 1
    while target.exists():  # two relaunches within one second
        n += 1
        target = stem.with_name(f"{stem.name}-{n}").with_suffix(path.suffix)
    path.rename(target)
    return target


def log(message: str) -> None:
    print(f"{time.strftime('%F %T')} {message}", flush=True)


def last_launch(path: Path) -> dict:
    lines = path.read_text().splitlines() if path.exists() else []
    return json.loads(lines[-1]) if lines else {}


def base_digest(spec: dict) -> str:
    return hashlib.sha256(Path(spec["base"]).read_bytes()).hexdigest()[:16]


def plan_launch(spec: dict, step: int, last: dict) -> dict:
    """How to resume at `step`: a changed data snapshot (or a repeat at the same step after one) starts a fresh
    dataloader from the actor state; anything else resumes the full checkpoint with the recorded seed."""
    fresh = spec["data"] != last.get("data") or (step == last.get("step") and last.get("mode") == "fresh")
    return {"step": step, "data": spec["data"], "source": spec["source"], "base": base_digest(spec),
            "mode": "fresh" if fresh else "full", "seed": 1000 + step if fresh else last["seed"]}


def switch_due(spec: dict, last: dict) -> bool:
    """A running trainer must be stopped at its next checkpoint when the launch inputs changed: data, source, or
    the base config (a systems edit such as rollout concurrency; launches recorded before digests count as equal)."""
    return bool(last) and (spec["data"] != last.get("data") or spec["source"] != last.get("source")
                           or last.get("base", base_digest(spec)) != base_digest(spec))


def resolved_config(spec: dict, launch: dict, resume_from: str | None) -> Path:
    config = OmegaConf.load(spec["base"])
    resume = {"trainer.resume_mode": "resume_path", "trainer.resume_from_path": resume_from} if resume_from else \
        {"trainer.resume_mode": "disable"}
    for key, value in {"data.train_files": [launch["data"]], "data.seed": launch["seed"], **resume,
                       "trainer.save_freq": spec["save_freq"], "trainer.total_training_steps": spec["total_steps"]}.items():
        OmegaConf.update(config, key, value, merge=False, force_add=True)
    path = Path(spec["dir"]) / f"resolved-{spec['arm']}-s{launch['step']}.yaml"
    OmegaConf.save(config, path, resolve=True)
    return path


class Arm:
    def __init__(self, spec: dict):
        self.spec, self.dir = spec, Path(spec["dir"])
        self.launches = self.dir / "LAUNCHES.jsonl"
        self.checkpoints = Path(spec["checkpoints"])

    def ssh(self, command: str, check: bool = False) -> str:
        return subprocess.run(["ssh", self.spec["node"], command], capture_output=True, text=True, check=check).stdout

    def latest(self) -> int:
        path = self.checkpoints / "latest_checkpointed_iteration.txt"
        return int(path.read_text()) if path.exists() else 0  # 0: a new arm starts from its base model

    def trainer_pids(self) -> list[str]:
        pattern = f"resolved-{self.spec['arm']}-s"
        return self.ssh(f"ps -eo pid,args | awk '/{pattern}/ && /archlab.training.runtime|launch.py/ && !/awk/ "
                        "{print $1}'").split()

    def ensure_agent_pods(self) -> None:
        pods, config = self.spec["agent_pods"], dlc_jobs.site(self.spec["dlc_site"])
        record = Path(self.spec["head_dir"]) / "agent-pods-job.json"
        try:
            job = json.loads(record.read_text())["JobId"]
            current = dlc_jobs.describe(job, config)
            state = current["status"]
        except (OSError, KeyError, ValueError) as error:
            state = f"unrecorded ({type(error).__name__})"
        except RuntimeError as error:  # the API did not answer: never double-submit on an unknown state
            log(f"agent pod status unavailable ({str(error)[:120]}); not resubmitting")
            return
        if state in LIVE_POD_STATES:
            if current["pods"] >= pods["pods"]:
                return
            # Only reached between launches (no trainer running), so resizing cannot cut live sessions.
            log(f"agent pods {job}: {current['pods']} < {pods['pods']} pods; replacing")
            dlc_jobs.stop(job, config)
            state = "resized"
        log(f"agent pods {state}; resubmitting {pods['pods']} x {pods['cpu']} CPU")
        body = dlc_jobs.job_body(config, name=pods["name"], pods=pods["pods"], cpu=pods["cpu"],
                                 memory_gi=pods["memory"], shared_memory_gi=16,
                                 command=dlc_jobs.agent_pods_command(self.spec["head_dir"], pods["ray_cpus"]))
        record.write_text(json.dumps({"JobId": dlc_jobs.create(body, config)}) + "\n")
        for _ in range(80):  # up to 20 minutes for pods to join; the head's own CPUs can start the run
            joined = self.ssh("/opt/venv/bin/ray status 2>/dev/null | grep -c '^ 1 node_'").strip()
            if joined.isdigit() and int(joined) >= 5:
                break
            time.sleep(15)

    def wait_for_free_gpus(self) -> None:
        for _ in range(60):
            used = self.ssh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}'")
            if used.strip() and int(used) < 8000:
                return
            time.sleep(10)

    def launch(self, last: dict) -> None:
        step = self.latest()
        launch = plan_launch(self.spec, step, last)
        resume = self.checkpoints / f"global_step_{step}"
        if step == 0:  # first launch of a new arm
            resume = None
        elif launch["mode"] == "fresh":  # actor, optimizer and RNG only; the saved dataloader belongs to old data
            init = self.dir / f"init-step{step}" / f"global_step_{step}"
            init.mkdir(parents=True, exist_ok=True)
            link = init / "actor"
            if not link.is_symlink():
                link.symlink_to(resume / "actor")
            resume = init
        config = resolved_config(self.spec, launch, str(resume) if resume else None)
        train_log = self.dir / "train.log"
        rotate_log(train_log)
        source = self.spec["source"]
        paths = ":".join([f"{source}/src", f"{source}/third_party/verl", f"{source}/third_party/verl/third_party/uni_agent",
                          f"{source}/third_party/verl/third_party/mimoagent-osr/src", self.spec["packages"]])
        with self.launches.open("a") as out:
            out.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **launch, "config": str(config)}) + "\n")
        log(f"launch {self.spec['arm']} from step {step} ({launch['mode']}, seed {launch['seed']}, "
            f"{Path(launch['data']).name}, {Path(source).name})")
        self.ssh(f"cd /tmp && (PYTHONPATH={paths} ARCHLAB_MLFLOW_CONFIG={self.spec['mlflow_config']} NO_PROXY='*' "
                 f"no_proxy='*' setsid nohup /opt/venv/bin/python -u -m archlab.training.runtime {config} "
                 f"> {train_log} 2>&1 < /dev/null &)")

    def run(self, once: bool = False) -> int:
        stalls, switch_from = 0, None
        while True:
            last = last_launch(self.launches)
            if self.trainer_pids():
                if switch_due(self.spec, last):
                    step = self.latest()
                    if switch_from is None:
                        switch_from = step
                        log(f"switch to {Path(self.spec['data']).name} / {Path(self.spec['source']).name} pending: "
                            f"after checkpoint {step}")
                    elif step > switch_from:
                        pids = " ".join(self.trainer_pids())
                        log(f"checkpoint {step} saved; stopping trainer {pids} for the switch")
                        self.ssh(f"kill {pids}")
                time.sleep(120)
                continue
            step = self.latest()
            if step >= self.spec["total_steps"]:
                log(f"done at step {step}")
                return 0
            stalls = stalls + 1 if last and step <= last["step"] else 0
            if stalls >= 3:
                log(f"3 launches without a new checkpoint (latest {step}); giving up")
                return 1
            self.ensure_agent_pods()
            self.wait_for_free_gpus()
            self.launch(last)
            switch_from = None
            if once:
                return 0
            time.sleep(300)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    return Arm(json.loads(Path(args[0]).read_text())).run(once="--once" in args)


if __name__ == "__main__":
    raise SystemExit(main())
