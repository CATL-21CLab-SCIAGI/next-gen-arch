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

    def evaluate(label: str, model: Path) -> dict | None:
        summary = runs / label / "summary.json"
        if not (summary.exists() and json.loads(summary.read_text()).get("complete")) and label not in attempted:
            attempted.add(label)
            with open(logs / f"eval-{label}.log", "a") as out:
                subprocess.run([sys.executable, "-u", "-m", "archlab.evaluation.agentic_curve", "--model", str(model),
                                "--manifest", protocol["manifest"], "--source", protocol["source"],
                                "--harness", protocol["harness"], "--pool", protocol["pool"],
                                "--output", str(runs / label), "--label", label, "--seeds", protocol["seeds"],
                                "--concurrency", str(protocol["concurrency"]), "--gpu-memory", str(protocol["gpu_memory"]),
                                "--attempts", str(protocol["attempts"])], stdout=out, stderr=subprocess.STDOUT)
        return json.loads(summary.read_text()) if summary.exists() else None

    while True:
        todo = [(p["label"], Path(p["model"]), p["curves"]) for p in plan["baselines"] if Path(p["model"]).exists()]
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
                todo.append((label, model, [[spec["curve"], step]]))
        for label, model, point_curves in todo:
            summary = evaluate(label, model)
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
        render(runs, Path(plan["curve_file"]), plan["eval"]["set"])
        time.sleep(300)


def render(runs: Path, path: Path, eval_set: str) -> None:
    keys = ("label", "complete", "scored_samples", "expected_samples", "pass_at_1", "success_at_k",
            "task_bootstrap_95ci", "generated_tokens", "wall_seconds", "terminations")
    points = [{k: s.get(k) for k in keys} for s in
              (json.loads(p.read_text()) for p in sorted(runs.glob("*/summary.json"), key=lambda p: p.stat().st_mtime))]
    path.write_text(json.dumps({"updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "set": eval_set, "points": points},
                               indent=1))


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
