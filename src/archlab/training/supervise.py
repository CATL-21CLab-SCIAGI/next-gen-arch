"""Own one qualified RL arm and recover only explicitly retriable failures.

Machine paths live in a JSON ArmSpec. Configuration changes require a new
qualification. An unresponsive PID is reported as stalled, never as progress.
The supervisor does not resize pools or change scientific settings.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock
from omegaconf import OmegaConf

from archlab.artifacts import atomic_write_json, sha256_file


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    node: str
    source: str
    qualification: str
    directory: str
    pythonpath: list[str]
    python: str = "/opt/venv/bin/python"
    mlflow_config: str | None = None
    max_attempts_without_progress: int = 3
    progress_timeout_seconds: int = 21600
    ssh_timeout_seconds: int = 60
    poll_seconds: int = 120

    def __post_init__(self):
        if not self.arm or not self.node or self.node.startswith("-"):
            raise ValueError("arm and SSH node must be explicit")
        for name in ("source", "qualification", "directory", "python"):
            if not Path(getattr(self, name)).is_absolute():
                raise ValueError(name + " must be an absolute path")
        if not self.pythonpath or any(not Path(path).is_absolute() for path in self.pythonpath):
            raise ValueError("pythonpath must contain absolute snapshot/runtime paths")
        if str(Path(self.source) / "src") != self.pythonpath[0]:
            raise ValueError("pythonpath must start with the qualified snapshot's src")
        for name in ("max_attempts_without_progress", "progress_timeout_seconds", "ssh_timeout_seconds", "poll_seconds"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(name + " must be a positive integer")


class Arm:
    def __init__(self, spec: ArmSpec):
        self.spec = spec
        self.directory = Path(spec.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "supervisor.json"
        self.qualification = Path(spec.qualification)
        report = json.loads((self.qualification / "qualification.json").read_text())
        if report.get("passed") is not True or report.get("source_root") != spec.source:
            raise ValueError("supervision requires a passed qualification for the selected source")
        self.config = OmegaConf.load(self.qualification / "resolved.yaml")
        self.checkpoints = Path(self.config.trainer.default_local_dir)
        self.identity = sha256_file(self.qualification / "qualification.json")

    def ssh(self, command: str) -> str:
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", self.spec.node, command],
            capture_output=True, text=True, check=True, timeout=self.spec.ssh_timeout_seconds,
        ).stdout

    def latest(self) -> int:
        marker = self.checkpoints / "latest_checkpointed_iteration.txt"
        return int(marker.read_text()) if marker.exists() else 0

    def alive(self, attempt: Path) -> bool:
        # Token matching excludes grep shells and other arms in the same node.
        for line in self.ssh("ps -eo args").splitlines():
            try:
                words = shlex.split(line)
            except ValueError:
                continue
            if ("archlab.training.runtime" in words and "--attempt" in words
                    and str(attempt) in words):
                return True
        return False

    def save(self, state: dict, status: str) -> None:
        state.update(status=status, checked_at=time.time(), qualification_sha256=self.identity)
        atomic_write_json(self.state_path, state)

    def gpus_idle(self) -> bool:
        values = self.ssh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").splitlines()
        if len(values) != self.config.trainer.n_gpus_per_node or any(not value.strip().isdigit() for value in values):
            raise subprocess.SubprocessError("GPU memory probe returned incomplete telemetry")
        return sum(int(value) for value in values) < 8000

    def launch(self, state: dict) -> None:
        attempt = self.directory / "attempts" / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        command = [self.spec.python, "-u", "-m", "archlab.training.runtime",
                   str(self.qualification / "resolved.yaml"), "--qualification", str(self.qualification),
                   "--attempt", str(attempt)]
        step = self.latest()
        if step:
            command += ["--resume", str(self.checkpoints / f"global_step_{step}")]
        variables = {"PYTHONPATH": ":".join(self.spec.pythonpath), "NO_PROXY": "*", "no_proxy": "*"}
        if self.spec.mlflow_config:
            variables["ARCHLAB_MLFLOW_CONFIG"] = self.spec.mlflow_config
        env = " ".join(shlex.quote(f"{key}={value}") for key, value in variables.items())
        state.update(attempt=str(attempt), started_at=time.time(), last_progress_at=time.time(),
                     step=step, attempts_without_progress=state.get("attempts_without_progress", 0) + 1)
        # Persist intent before SSH. An ambiguous launch is never blindly retried.
        self.save(state, "launching")
        self.ssh(f"cd /tmp && (setsid nohup env {env} {shlex.join(command)} "
                 f"> {shlex.quote(str(attempt / 'train.log'))} 2>&1 < /dev/null &)")

    def cycle(self, state: dict) -> str:
        step = self.latest()
        if step > state.get("step", 0):
            state.update(step=step, attempts_without_progress=0, last_progress_at=time.time())
        attempt = Path(state["attempt"]) if state.get("attempt") else None
        if attempt:
            progress = attempt / "progress.json"
            if progress.exists():
                heartbeat = json.loads(progress.read_text())
                if heartbeat["step"] > state.get("progress_step", -1):
                    state.update(progress_step=heartbeat["step"], last_progress_at=heartbeat["at"])
            if self.alive(attempt):
                return "stalled" if time.time() - state["last_progress_at"] > self.spec.progress_timeout_seconds else "running"
            receipt = attempt / "result.json"
            result = json.loads(receipt.read_text()) if receipt.exists() else {}
            if result.get("status") == "succeeded":
                return "complete" if step >= self.config.trainer.total_training_steps else "stopped_early"
            if result.get("status") != "failed" or not result.get("failure", {}).get("retriable"):
                return "needs_triage"
        if step >= self.config.trainer.total_training_steps:
            return "complete"
        if state.get("attempts_without_progress", 0) >= self.spec.max_attempts_without_progress:
            return "retry_limit"
        if not self.gpus_idle():
            return "gpu_busy"
        self.launch(state)
        return "running"

    def run(self, *, once: bool = False) -> int:
        with FileLock(self.directory / ".supervisor.lock", timeout=0):
            state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
            if state and state.get("qualification_sha256") != self.identity:
                raise ValueError("arm directory belongs to a different qualification")
            while True:
                try:
                    status = self.cycle(state)
                except subprocess.SubprocessError:
                    status = "probe_unavailable"
                self.save(state, status)
                print(f"{self.spec.arm}: {status}", flush=True)
                if status not in {"running", "probe_unavailable", "gpu_busy"}:
                    return 0 if status == "complete" else 1
                if once:
                    return 0 if status == "running" else 1
                time.sleep(self.spec.poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    return Arm(ArmSpec(**json.loads(args.spec.read_text()))).run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
