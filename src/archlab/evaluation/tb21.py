"""Terminal-Bench 2.1 ladder for RL checkpoints (Harbor + Terminus-2), logged to MLflow.

A JSON plan (machine paths stay in it) lists points in priority order. One point runs at a time: an in-flight Harbor
job for the label is adopted, otherwise a vLLM server for the checkpoint is started on the eval node, Harbor runs
all tasks with the official per-task timeouts and resources, and the score is logged next to the held-out curve.

Task verifiers install uv from GitHub and Python plus test wheels at run time. Over this site's slow direct egress
that cannot finish inside official timeouts, so a read-only toolchain mirror is mounted into every task container
and uv/pip are pointed at it (uv release via file://, python-build-standalone via file://, PyPI via an in-region
index). Test logic is unchanged; `oracle` runs the reference solutions to prove the verifier path before models.
usage: python -m archlab.evaluation.tb21 {run|oracle|score} <plan.json | job dir>
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from archlab.evaluation.ladder import Curves, candidates

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 80000, "max_turns": 500, "thinking": False}


def log(**fields):
    print(json.dumps({"at": time.strftime("%F %T"), **fields}), flush=True)


def harbor_command(plan: dict, model_name: str, label: str, agent: str = "terminus-2") -> list[str]:
    mirror, index = plan["mirror"]["path"], plan["mirror"]["index"]
    uv = plan["mirror"]["uv_version"]
    tool_env = {"UV_DOWNLOAD_URL": f"file:///opt/tb-mirror/astral-sh/uv/releases/download/{uv}",
                "UV_PYTHON_INSTALL_MIRROR": "file:///opt/tb-mirror/python-build-standalone",
                "UV_DEFAULT_INDEX": index, "PIP_INDEX_URL": index}
    command = [plan["harbor"], "run", "-p", plan["tasks"], "-a", agent, "-o", str(Path(plan["jobs"]) / label),
               "-n", str(plan["concurrency"]),
               # Image builds and agent setup run on our docker host; agent and verifier budgets stay official.
               "--environment-build-timeout-multiplier", "8", "--agent-setup-timeout-multiplier", "8",
               "--mounts", json.dumps([{"type": "bind", "source": mirror, "target": "/opt/tb-mirror", "read_only": True}])]
    for key, value in tool_env.items():
        command += ["--ve", f"{key}={value}", "--ae", f"{key}={value}"]
    if agent == "terminus-2":
        command += ["-m", f"openai/{model_name}",
                    "--ak", f"temperature={SAMPLING['temperature']}", "--ak", "parser_name=xml",
                    "--ak", f"max_turns={SAMPLING['max_turns']}",
                    # tmux is baked into the images; asciinema would be fetched from the network at setup.
                    "--ak", "record_terminal_session=false",
                    "--ak", "llm_call_kwargs=" + json.dumps({
                        "top_p": SAMPLING["top_p"], "top_k": SAMPLING["top_k"], "max_tokens": SAMPLING["max_tokens"],
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": SAMPLING["thinking"]}}}),
                    "--ak", "model_info=" + json.dumps({
                        "max_input_tokens": 262144, "max_output_tokens": SAMPLING["max_tokens"],
                        "input_cost_per_token": 0, "output_cost_per_token": 0})]
    return command


def complete_job(jobs: Path, label: str) -> Path | None:
    for path in sorted((jobs / label).glob("*/result.json")):
        stats = json.loads(path.read_text()).get("stats", {})
        if stats.get("n_running_trials") == 0 and stats.get("n_pending_trials") == 0:
            return path.parent
    return None


def score(job: Path, *, bootstrap: int = 2000) -> dict:
    """Mean reward over trials (an errored trial scores 0, as on the leaderboard) with a task-bootstrap 95% CI."""
    rows = []
    for trial in sorted(job.glob("*/result.json")):
        record = json.loads(trial.read_text())
        reward = ((record.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        error = (record.get("exception_info") or {}).get("exception_type")
        rows.append((record.get("task_name"), float(reward or 0.0), error))
    if not rows:
        raise ValueError(f"no trials in {job}")
    rewards = [reward for _, reward, _ in rows]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(rewards) for _ in rewards) / len(rewards) for _ in range(bootstrap))
    errors: dict[str, int] = {}
    for _, _, error in rows:
        if error:
            errors[error] = errors.get(error, 0) + 1
    return {"mean_reward": sum(rewards) / len(rewards), "ci95": [boots[int(0.025 * bootstrap)], boots[int(0.975 * bootstrap) - 1]],
            "trials": len(rows), "solved": sum(1 for reward in rewards if reward >= 1.0), "errors": errors,
            "solved_tasks": sorted(task for task, reward, _ in rows if reward >= 1.0)}


class EvalNode:
    """vLLM servers on the eval node over ssh; the playground is paused while the ladder owns the GPU."""

    def __init__(self, plan: dict):
        self.plan, self.serve = plan, plan["serve"]

    def ssh(self, command: str, check: bool = True) -> str:
        return subprocess.run(["ssh", self.serve["node"], command], capture_output=True, text=True, check=check).stdout

    def kill_matching(self, pattern: str) -> None:
        self.ssh(f"ps -eo pid,args | awk '/{pattern}/ && !/awk/ {{print $1}}' | xargs -r kill", check=False)

    def healthy(self, port: int) -> bool:
        url = f"http://{self.serve['host']}:{port}/health"
        return bool(self.ssh(f"curl -sf --noproxy '*' -m 3 {url} && echo up", check=False).strip())

    def start(self, model: str, name: str) -> int:
        for pattern in self.serve.get("pause", []):
            self.kill_matching(pattern)
        port, secrets = self.serve["port"], self.plan["secrets"]
        log_path = Path(self.plan["logs"]) / f"serve-{name}.log"
        pid = self.ssh(
            f"source {self.serve['env_script']}; "
            "export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas; "
            f"KEY=$(python -c \"import json;print(json.load(open('{secrets}'))['backend_api_key'])\"); "
            f"cd /tmp; VLLM_API_KEY=$KEY setsid nohup python -m vllm.entrypoints.openai.api_server --model {model} "
            f"--served-model-name {name} --host {self.serve['host']} --port {port} --dtype bfloat16 "
            f"--max-model-len 262144 --gpu-memory-utilization {self.serve['gpu_memory_utilization']} "
            "--language-model-only --enable-prefix-caching --max-num-seqs 16 --generation-config vllm "
            "--enable-auto-tool-choice --tool-call-parser qwen3_coder "
            f"--default-chat-template-kwargs '{{\"enable_thinking\": false}}' > {log_path} 2>&1 < /dev/null & echo $!").strip()
        for _ in range(120):
            if self.healthy(port):
                return int(pid)
            if not self.ssh(f"ps -p {pid} >/dev/null && echo alive", check=False).strip():
                raise RuntimeError(f"vLLM for {name} exited; see {log_path}")
            time.sleep(15)
        self.stop(int(pid))
        raise RuntimeError(f"vLLM for {name} not healthy after 30 min")

    def stop(self, pid: int) -> None:
        self.ssh(f"pkill -TERM -P {pid}; kill {pid}; sleep 20; pkill -KILL -P {pid}; kill -9 {pid}", check=False)


def harbor_running(plan: dict, label: str) -> bool:
    target = f" -o {Path(plan['jobs']) / label} "
    lines = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    return any("harbor" in line and " run " in line and target in f"{line} " for line in lines)


def run_harbor(plan: dict, model_name: str, port: int, label: str, agent: str = "terminus-2") -> None:
    secrets = json.loads(Path(plan["secrets"]).read_text())
    env = dict(os.environ, OPENAI_API_BASE=f"http://{plan['serve']['host']}:{port}/v1",
               OPENAI_API_KEY=secrets["backend_api_key"], NO_PROXY="*", no_proxy="*")
    with open(Path(plan["logs"]) / f"harbor-{label}.log", "a") as out:
        subprocess.run(harbor_command(plan, model_name, label, agent), stdout=out, stderr=subprocess.STDOUT, env=env)


def log_mlflow(plan: dict, point: dict, result: dict) -> None:
    metrics = {"tb21/mean_reward": result["mean_reward"], "tb21/ci95_low": result["ci95"][0],
               "tb21/ci95_high": result["ci95"][1], "tb21/solved": result["solved"]}
    Curves(plan["mlflow"]).log(point["curve"], f"tb21:{point['label']}", point["step"], metrics,
                               tags={"tb21_settings": json.dumps(plan["settings"])})


def run(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    ledger_path = Path(plan["ledger"])
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"settings": plan["settings"], "points": {}}
    node = EvalNode(plan)
    Path(plan["logs"]).mkdir(parents=True, exist_ok=True)
    while True:
        todo = candidates(plan, set(ledger["points"]))
        if not todo:
            time.sleep(900)
            continue
        point = todo[0]
        label, jobs = point["label"], Path(plan["jobs"])
        if complete_job(jobs, label) is None:
            pid = None
            try:
                if harbor_running(plan, label):
                    log(adopt=label)
                    while harbor_running(plan, label):
                        time.sleep(60)
                else:
                    log(start=label, model=point["model"])
                    pid = node.start(point["model"], label)
                    run_harbor(plan, label, plan["serve"]["port"], label)
            except (RuntimeError, subprocess.CalledProcessError) as error:
                log(label=label, error=str(error)[:300])
                time.sleep(600)
                continue
            finally:
                if pid:
                    node.stop(pid)
        job = complete_job(jobs, label)
        if job is None:
            log(label=label, error="Harbor finished without a complete result.json")
            time.sleep(600)
            continue
        result = score(job)
        ledger["points"][label] = {**point, **result, "job": str(job), "finished": time.strftime("%F %T"),
                                   "mlflow": False}
        log(done=label, mean_reward=result["mean_reward"], solved=result["solved"], errors=result["errors"])
        for done_label, done in ledger["points"].items():
            if not done["mlflow"]:
                try:
                    log_mlflow(plan, done, done)
                    done["mlflow"] = True
                except Exception as error:  # never block evaluation on tracking
                    log(label=done_label, mlflow_error=f"{type(error).__name__}: {error}"[:300])
        ledger_path.write_text(json.dumps(ledger, indent=1) + "\n")


def main(argv: list[str] | None = None) -> None:
    command, target = (argv or sys.argv[1:])[:2]
    if command == "run":
        run(Path(target))
    elif command == "oracle":  # reference solutions through the same verifier path; no model involved
        plan = json.loads(Path(target).read_text())
        Path(plan["logs"]).mkdir(parents=True, exist_ok=True)
        run_harbor(plan, "oracle", plan["serve"]["port"], "oracle", agent="oracle")
        print(json.dumps(score(complete_job(Path(plan["jobs"]), "oracle")), indent=1))
    elif command == "score":
        print(json.dumps(score(Path(target)), indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
