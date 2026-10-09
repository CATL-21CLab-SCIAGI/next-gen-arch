"""Quiet MLflow/report synchronizer for the bounded matched evaluation queue."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from archlab.artifacts import atomic_write_json
from archlab.evaluation.reasoning import summarize
from archlab.tracking.mlflow_sync import configure_client


def monitor(plan_path, credentials):
    plan = json.loads(plan_path.read_text())
    root, bundle = Path(plan["output"]), Path(plan["bundle"])
    client = configure_client(credentials)
    experiment = client.get_experiment_by_name("DeepSeek V4.1 — Full fine-tuning")
    if experiment is None or experiment.lifecycle_stage != "active":
        raise ValueError("existing Full fine-tuning experiment is unavailable")
    state_path = root / "MLFLOW.json"
    if state_path.exists():
        run_id = json.loads(state_path.read_text())["run_id"]
    else:
        run = client.create_run(
            experiment.experiment_id,
            tags={
                "mlflow.runName": "Matched reasoning — full4537 + adapters307",
                "archlab.role": "evaluation",
                "archlab.output": str(root),
                "archlab.variants": "normal,simplicial",
                "archlab.paired_unit": "counterfactual twin",
                "archlab.inference": "native chat, greedy, uncached",
            },
        )
        run_id = run.info.run_id
        atomic_write_json(
            state_path,
            dict(
                run_id=run_id,
                experiment_id=experiment.experiment_id,
                experiment_name=experiment.name,
            ),
        )
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    for key, value in dict(
        sealed_cases=manifest["rows"],
        full_matched_tokens=756364650,
        adapter_matched_tokens=50119869,
    ).items():
        client.log_metric(run_id, key, value)
    client.log_artifact(run_id, str(plan_path), "protocol")
    client.log_artifact(run_id, str(bundle / "MANIFEST.json"), "protocol")
    client.log_artifact(run_id, str(bundle / "cases.json"), "protocol")
    client.log_artifact(run_id, str(bundle / "labels.json"), "protocol")
    seen, reports = {}, set()
    started = time.monotonic()
    deadline = plan["maximum_queue_hours"] * 3600 + 600
    while time.monotonic() - started < deadline:
        try:
            for path in [
                *root.glob("queue-node-*-HEALTH.json"),
                *root.glob("*/PROGRESS.json"),
                *root.glob("*/NUMERICAL_QUALIFICATION.json"),
                *root.glob("*/COMPLETE.json"),
            ]:
                if path.parent.name.startswith("tiny-"):
                    continue
                stamp = path.stat().st_mtime_ns
                if seen.get(str(path)) == stamp:
                    continue
                value = json.loads(path.read_text())
                if "HEALTH" in path.name:
                    for arm, ratio in value["slowdown_ratios"].items():
                        client.log_metric(run_id, f"training_slowdown.{arm}", ratio)
                    client.log_metric(run_id, "minimum_gpu_free_gib", value["minimum_gpu_free_gib"])
                    client.set_tag(run_id, "archlab.active_phase", value["phase"])
                elif path.name == "PROGRESS.json":
                    client.log_metric(
                        run_id, f"{path.parent.name}.completed_cases", value["completed"]
                    )
                elif path.name == "NUMERICAL_QUALIFICATION.json":
                    client.log_metric(
                        run_id, f"{path.parent.name}.qualification_ce_error", value["error"]
                    )
                    client.log_artifact(run_id, str(path), path.parent.name)
                else:
                    client.log_artifact(run_id, str(path), path.parent.name)
                seen[str(path)] = stamp
            for kind, indices in (("full", (0, 1)), ("adapter", (2, 3))):
                directories = [root / plan["phases"][index]["name"] for index in indices]
                if kind in reports or not all((p / "COMPLETE.json").exists() for p in directories):
                    continue
                report = summarize(bundle, *directories)
                report_path = root / f"{kind}-REPORT.json"
                atomic_write_json(report_path, report)
                for group, values in report["groups"].items():
                    for key in ("normal_accuracy", "simplicial_accuracy", "delta_pp"):
                        client.log_metric(
                            run_id, f"{kind}.{group.replace('/', '.')}.{key}", values[key]
                        )
                client.log_metric(
                    run_id,
                    f"{kind}.calibration_controls_pass",
                    int(report["calibration_controls_pass"]),
                )
                client.log_artifact(run_id, str(report_path), "results")
                for directory in directories:
                    client.log_artifact(
                        run_id, str(directory / "predictions.jsonl"), directory.name
                    )
                reports.add(kind)
            failures = list(root.glob("queue-node-*-FAILED.json"))
            if failures or (root / "STOP").exists():
                for path in failures:
                    client.log_artifact(run_id, str(path), "failures")
                client.set_terminated(run_id, "FAILED")
                atomic_write_json(
                    root / "TRACKING_STATUS.json", dict(status="FAILED", run_id=run_id)
                )
                return
            if len(reports) == 2 and all(
                (root / f"queue-node-{node}-COMPLETE.json").exists() for node in (0, 1)
            ):
                client.set_terminated(run_id, "FINISHED")
                atomic_write_json(
                    root / "TRACKING_STATUS.json", dict(status="FINISHED", run_id=run_id)
                )
                return
            atomic_write_json(
                root / "TRACKING_STATUS.json",
                dict(status="RUNNING", run_id=run_id, reports=sorted(reports), unix=time.time()),
            )
        except Exception as error:
            # Tracking must never interrupt either training or evaluation.
            atomic_write_json(
                root / "TRACKING_STATUS.json",
                dict(status="RETRYING", error=str(error), unix=time.time()),
            )
        time.sleep(60)
    client.set_terminated(run_id, "KILLED")
    atomic_write_json(root / "TRACKING_STATUS.json", dict(status="DEADLINE", run_id=run_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    args = parser.parse_args()
    monitor(args.plan, args.credentials)


if __name__ == "__main__":
    main()
