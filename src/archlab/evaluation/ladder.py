"""Evaluation ladder for RL checkpoints: export, held-out agentic curve, and MLflow curves per model.

Shared by the held-out tracker (`heldout` below) and the Terminal-Bench ladder (`archlab.evaluation.tb21`).
A JSON plan keeps machine paths. Checkpoints are exported with the upstream FSDP merger and verified tensor by
tensor; the held-out curve runs `archlab.evaluation.agentic_curve` (frozen protocol) per point; every completed point
is logged once to its MLflow curve(s) through an idempotent ledger.
usage: python -m archlab.evaluation.ladder heldout <plan.json>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def log(**fields):
    print(json.dumps({"at": time.strftime("%F %T"), **fields}), flush=True)


def exported_steps(exports: Path, prefix: str) -> list[int]:
    steps = []
    for path in exports.glob(f"{prefix}-step*"):
        tail = path.name[len(prefix) + len("-step"):]
        if tail.isdigit() and (path / "EXPORTED").exists():
            steps.append(int(tail))
    return sorted(steps)


def candidates(plan: dict, done: set[str]) -> list[dict]:
    """Points in priority order that are available and not yet scored.

    Entries: a fixed point {label, model, curve, step}; {"newest": spec} for the newest export once it is at least
    `every` steps past the newest scored one; {"every": spec} for every export at a multiple of `every`.
    """
    found, seen = [], set(done)
    for entry in plan["points"]:
        if "newest" in entry:
            spec = entry["newest"]
            steps = exported_steps(Path(spec["exports"]), spec["prefix"])
            ran = [s for s in steps if f"{spec['prefix']}-step{s}" in done]
            points = [(steps[-1], spec)] if steps and (not ran or steps[-1] - max(ran) >= spec["every"]) else []
        elif "every" in entry:
            spec = entry["every"]
            points = [(s, spec) for s in exported_steps(Path(spec["exports"]), spec["prefix"]) if s % spec["every"] == 0]
        else:
            if Path(entry["model"]).exists():
                found.append(entry)
            continue
        for step, spec in points:
            label = f"{spec['prefix']}-step{step}"
            if label not in seen:
                seen.add(label)
                found.append({"label": label, "model": str(Path(spec["exports"]) / label),
                              "curve": spec["curve"], "step": step})
    return [point for point in found if point["label"] not in done]


def export_checkpoint(checkpoint: Path, target: Path, log_path: Path, env: dict | None = None) -> Path:
    """Merge an FSDP actor checkpoint to HF weights with the upstream merger, then verify every tensor.

    Construction is CPU-only and eager (the stored flash-attention backend is rejected without CUDA); no weight
    changes and no forward pass. Idempotent: an EXPORTED receipt marks a verified export.
    """
    if (target / "EXPORTED").exists():
        return target
    for operation in ("merge", "test"):
        with open(log_path, "a") as out:
            code = subprocess.run([sys.executable, "-m", "archlab.evaluation.ladder", "merge", operation,
                                   str(checkpoint), str(target)], stdout=out, stderr=subprocess.STDOUT,
                                  env={**(env or os.environ), "CUDA_VISIBLE_DEVICES": ""}).returncode
        if code:
            raise RuntimeError(f"export {operation} failed for {checkpoint}")
    (target / "EXPORTED").write_text(json.dumps({"checkpoint": str(checkpoint), "validation": "upstream merger test"}))
    return target


def _merge(operation: str, checkpoint: Path, target: Path) -> None:
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    config = ModelMergerConfig(operation=operation, backend="fsdp", local_dir=str(checkpoint),
                               hf_model_config_path=str(checkpoint / "huggingface"), target_dir=str(target),
                               test_hf_dir=str(target), use_cpu_initialization=True)
    merger = FSDPModelMerger(config)
    for model_config in (merger.model_config, getattr(merger.model_config, "text_config", None),
                         getattr(merger.model_config, "vision_config", None)):
        if model_config is not None:
            model_config.attn_implementation = model_config._attn_implementation = "eager"
    merger.merge_and_save()
    merger.cleanup()


class Curves:
    """One MLflow run per model curve; each (curve, label) is logged once (ledger: {"runs", "logged"})."""

    def __init__(self, config: dict):
        self.config, self.ledger_path = config, Path(config["ledger"])

    def log(self, curve: str, label: str, step: int, metrics: dict, tags: dict | None = None) -> bool:
        import mlflow

        ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {"runs": {}, "logged": []}
        key = f"{curve}|{label}"
        if key in ledger["logged"]:
            return False
        secrets = json.loads(Path(self.config["secrets"]).read_text())
        os.environ["MLFLOW_TRACKING_TOKEN"] = secrets["token"]
        mlflow.set_tracking_uri(secrets["tracking_uri"])
        client = mlflow.tracking.MlflowClient()
        run_id = ledger["runs"].get(curve)
        if run_id is None and self.config.get("runs_from"):  # share runs with another ledger, never its file
            run_id = json.loads(Path(self.config["runs_from"]).read_text())["runs"].get(curve)
        if run_id is None:
            name = self.config["experiment"]
            experiment = client.get_experiment_by_name(name) or client.get_experiment(client.create_experiment(name))
            run_id = client.create_run(experiment.experiment_id, run_name=curve, tags=self.config.get("tags", {})).info.run_id
        ledger["runs"][curve] = run_id
        for metric, value in metrics.items():
            client.log_metric(run_id, metric, value, step=step)
        for tag, value in (tags or {}).items():
            client.set_tag(run_id, tag, value)
        ledger["logged"].append(key)
        self.ledger_path.write_text(json.dumps(ledger, indent=1))
        return True


def heldout(plan_path: Path) -> None:
    """Export and evaluate every `every`-th checkpoint of each arm on the fixed held-out set, after the baselines."""
    plan = json.loads(plan_path.read_text())
    runs, exports, logs = Path(plan["runs"]), Path(plan["exports"]), Path(plan["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    curves, protocol = Curves(plan["mlflow"]), plan["eval"]
    failed: set[str] = set()
    attempted: set[str] = set()  # one evaluation attempt per point per process; an incomplete run is reported

    def running(label: str) -> bool:  # adopt an evaluation already in flight (e.g. across a tracker restart)
        target = f"--output {runs / label} "
        lines = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
        return any(target in f"{line} " for line in lines if "--manifest" in line)

    def evaluate(label: str, model: Path, seeds: str | None = None, gpu_memory: float | None = None) -> dict | None:
        summary = runs / label / "summary.json"
        if running(label):
            return None
        if not (summary.exists() and json.loads(summary.read_text()).get("complete")) and label not in attempted:
            attempted.add(label)
            # The protocol pins the evaluator's import path, so every point runs the same code (frozen curve).
            env = {**os.environ, "PYTHONPATH": protocol["pythonpath"]} if protocol.get("pythonpath") else None
            with open(logs / f"eval-{label}.log", "a") as out:
                # A frozen curve names its evaluator file; the module is the same code for new curves.
                runner = [protocol["runner"]] if protocol.get("runner") else ["-m", "archlab.evaluation.agentic_curve"]
                subprocess.run([sys.executable, "-u", *runner, "--model", str(model),
                                "--manifest", protocol["manifest"], "--source", protocol["source"],
                                "--harness", protocol["harness"], "--pool", protocol["pool"],
                                "--output", str(runs / label), "--label", label, "--seeds", seeds or protocol["seeds"],
                                "--concurrency", str(protocol["concurrency"]),
                                "--gpu-memory", str(gpu_memory or protocol["gpu_memory"]),
                                "--attempts", str(protocol["attempts"])], stdout=out, stderr=subprocess.STDOUT, env=env)
        return json.loads(summary.read_text()) if summary.exists() else None

    while True:
        todo = [(p["label"], Path(p["model"]), p["curves"], p.get("seeds"), p.get("gpu_memory"))
                for p in plan["baselines"] if Path(p["model"]).exists()]
        for arm, spec in plan["arms"].items():
            root = Path(spec["checkpoints"])
            steps = sorted(int(p.name.split("_")[-1]) for p in root.glob("global_step_*")
                           if (p / "actor/huggingface").exists()) if root.exists() else []
            for step in (s for s in steps if s % spec["every"] == 0):
                label = f"{arm}-step{step}"
                if label in failed:
                    continue
                try:
                    model = export_checkpoint(root / f"global_step_{step}/actor", exports / label,
                                              logs / f"export-{label}.log")
                except RuntimeError as error:
                    log(label=label, error=str(error))
                    failed.add(label)
                    continue
                todo.append((label, model, [[spec["curve"], step]], None, spec.get("gpu_memory")))
        for label, model, point_curves, seeds, gpu_memory in todo:
            summary = evaluate(label, model, seeds, gpu_memory)
            if not (summary and summary.get("complete")):
                continue
            low, high = summary["task_bootstrap_95ci"]
            metrics = {"heldout43/pass_at_1": summary["pass_at_1"], "heldout43/ci95_low": low,
                       "heldout43/ci95_high": high, "heldout43/success_at_2": summary["success_at_k"],
                       "heldout43/generated_tokens": summary["generated_tokens"]}
            for curve, step in point_curves:
                try:
                    curves.log(curve, label, step, metrics)
                except Exception as error:  # never block evaluation on tracking
                    log(label=label, mlflow_error=f"{type(error).__name__}: {error}"[:300])
        render(runs, Path(plan["curve_file"]), plan["eval"]["set"], plan.get("pooled", {}))
        time.sleep(300)


def per_task_rates(summary: dict) -> dict[str, tuple[float, int]]:
    return {task: (row["successes"], row["scored"]) for task, row in summary["per_task"].items() if row["scored"]}


def pooled_difference(base: list[dict], arm: list[dict], *, bootstrap: int = 4000) -> dict:
    """Task-paired difference between pooled checkpoint samples and pooled baseline samples (task bootstrap CI).

    Single points of a 43-task curve carry about +/-9 points of noise; pooling checkpoints and baseline seeds per task
    is the more powerful test of whether training moved the policy."""
    import random

    def pool(summaries):
        rates: dict[str, list[float]] = {}
        for summary in summaries:
            for task, (successes, scored) in per_task_rates(summary).items():
                rates.setdefault(task, [0.0, 0])
                rates[task][0] += successes
                rates[task][1] += scored
        return {task: successes / scored for task, (successes, scored) in rates.items()}

    left, right = pool(base), pool(arm)
    tasks = sorted(set(left) & set(right))
    diff = [right[task] - left[task] for task in tasks]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diff) for _ in diff) / len(diff) for _ in range(bootstrap))
    return {"tasks": len(tasks), "baseline": sum(left[t] for t in tasks) / len(tasks),
            "trained": sum(right[t] for t in tasks) / len(tasks), "difference": sum(diff) / len(diff),
            "ci95": [boots[int(0.025 * bootstrap)], boots[int(0.975 * bootstrap) - 1]]}


def render(runs: Path, path: Path, eval_set: str, pooled: dict | None = None) -> None:
    keys = ("label", "complete", "scored_samples", "expected_samples", "pass_at_1", "success_at_k",
            "task_bootstrap_95ci", "generated_tokens", "wall_seconds", "terminations")
    summaries = {p.parent.name: json.loads(p.read_text())
                 for p in sorted(runs.glob("*/summary.json"), key=lambda p: p.stat().st_mtime)}
    points = [{k: s.get(k) for k in keys} for s in summaries.values()]
    windows = {}
    for name, spec in (pooled or {}).items():  # {"name": {"baseline": [labels], "trained": [labels]}}
        base = [summaries[label] for label in spec["baseline"] if summaries.get(label, {}).get("complete")]
        arm = [summaries[label] for label in spec["trained"] if summaries.get(label, {}).get("complete")]
        if base and arm:
            windows[name] = {**pooled_difference(base, arm), "baseline_points": len(base), "trained_points": len(arm)}
    path.write_text(json.dumps({"updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "set": eval_set, "points": points,
                                "pooled": windows}, indent=1))


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if args[0] == "heldout":
        heldout(Path(args[1]))
    elif args[0] == "merge":
        _merge(args[1], Path(args[2]), Path(args[3]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
