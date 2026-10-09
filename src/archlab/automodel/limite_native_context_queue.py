"""Prepare an inactive, sealed successor for Violetto's deferred RL phase.

This command writes new plan artifacts only. It never contacts a node, starts
a controller, changes an old plan, or creates an MLflow run. The caller supplies
the new run receipt after attaching the cap audit and qualification evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from archlab.automodel.limite_math_queue import sha256_file
from archlab.automodel.limite_rl_queue import validate_plan
from archlab.rl.limite_protocol import MathRolloutProtocol


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError("refusing to overwrite a prepared artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous-plan", "source", "curve-plan", "curve-config", "curve-completion",
                 "long-probe", "distributed-proof", "tracking-receipt", "destination", "production"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--oss", required=True)
    args = parser.parse_args()
    old = json.loads(args.previous_plan.read_text())
    source, destination, production = args.source.resolve(), args.destination.resolve(), args.production.resolve()
    if source.joinpath("SOURCE_REVISION").read_text().strip() != args.source_revision:
        raise ValueError("sealed source identity differs")
    if destination.exists() or production.exists():
        raise ValueError("successor queue and production roots must be fresh")
    recipe = source / "recipes/limite/violetto_math_rl_native_context.yaml"
    spec = yaml.safe_load(recipe.read_text())
    protocol = MathRolloutProtocol(**spec["rollout"]).contract()
    curve_config = json.loads(args.curve_config.read_text())
    curve_sha, curve_config_sha = sha256_file(args.curve_plan), canonical_sha(curve_config)
    if len(curve_config["stages"]) != 9:
        raise ValueError("expected all nine requested curve evaluations")
    tracking = json.loads(args.tracking_receipt.read_text())
    run_id = tracking["run_id"]
    old_run = old["stages"][-1]["checkpoint_gate"]["mlflow_run_id"]
    if run_id == old_run or not tracking.get("experiment_id"):
        raise ValueError("native-context phase needs a distinct audited MLflow run")
    plan = copy.deepcopy(old)
    plan.update(source=str(source), source_revision=args.source_revision, controller_source=str(source),
                controller_source_revision=args.source_revision, output=str(destination / "queue"),
                priority="Complete all requested full-context curve evaluations, then new native-context RL142–400 and AIME26")
    train, evaluation = plan["stages"]
    argv = train["args"]
    def set_arg(flag, value):
        argv[argv.index(flag) + 1] = str(value)
    for flag, value in {
        "--math-protocol": recipe, "--phase-start": 142, "--output": production,
        "--oss": args.oss, "--stop-file": destination / "STOP_REQUEST",
    }.items():
        set_arg(flag, value)
    if "--new-tracking-phase" not in argv:
        argv.append("--new-tracking-phase")
    train.update(name="violetto-native-context-rl142-400", marker=str(production / "FINISHED.json"), port=30607)
    train["gates"][0]["path"] = train["marker"]
    preconditions = [gate for gate in train["preconditions"]
                     if "limite-sft-curve-eval" not in gate["path"]]
    completion_equals = dict(passed=True, status="complete", stage=9, verified_models=9,
                             verified_responses=1080, verified_shards=144,
                             queue_plan_sha256=curve_sha, config_sha256=curve_config_sha)
    state_equals = dict(status="complete", stage=9, config_sha256=curve_config_sha)
    for stage in curve_config["stages"]:
        state_equals[f"stages.{stage['name']}.status"] = "complete"
        state_equals[f"stages.{stage['name']}.plan_sha256"] = stage["plan_sha256"]
    train["preconditions"] = [
        dict(path=str(args.curve_completion.resolve()), equals=completion_equals),
        dict(path=str(Path(curve_config["output"]) / "STATE.json"), equals=state_equals),
        dict(path=str(args.long_probe.resolve()), equals={
            "passed": True, "resident_actor": True, "resident_snapshot": True,
            "resident_graphs": 4, "qualified_context": 131072, "runtime.source_revision": args.source_revision,
        }),
        dict(path=str(args.distributed_proof.resolve()), equals={
            "passed": True, "world_size": 8, "source_revision": args.source_revision,
            "phase_start": 142, "checkpoint_step": 142, "completed_step": 143,
            "optimizer_scheduler_retained": True, "all_rank_rng_verified": True,
        }),
        *preconditions,
    ]
    template = json.loads(Path(evaluation["template_plan"]).read_text())
    source_files = {str(path.relative_to(source / "src/archlab")): sha256_file(path)
                    for path in sorted((source / "src/archlab").rglob("*.py"))}
    eval_recipe = source / "recipes/limite/eval_aime26_pipeline_v3.yaml"
    template.update(source=str(source), source_git_revision=args.source_revision,
                    source_file_sha256=source_files, implementation_sha256=canonical_sha(source_files),
                    recipe=str(eval_recipe), recipe_sha256=sha256_file(eval_recipe),
                    output=str(destination / "responses-violetto-native-context-rl400"))
    name = "violetto-native-context-rl400-full-context"
    template["models"][0].update(name=name, current_rl_phase_start=142, current_rl_phase_nominal_updates=258)
    template_path = destination / "TEMPLATE-violetto-native-context-rl400.json"
    write_new(template_path, template)
    evaluation.update(name=name, template_plan=str(template_path), template_plan_sha256=sha256_file(template_path),
                      plan_path=str(destination / "PLAN-violetto-native-context-rl400.json"))
    gate = evaluation["checkpoint_gate"]
    gate.update(rl_root=str(production), source_revision=args.source_revision, mlflow_run_id=run_id)
    gate["checkpoint_contract"].update(math_protocol=protocol, phase_start=142)
    evaluation["preconditions"] = [
        dict(path=train["marker"], equals=train["gates"][0]["equals"]),
        dict(path=str(production / "checkpoints/step-0000400/COMPLETE.json"), equals={
            "step": 400, "model_kind": "native", "trainable_mode": "full",
            "source_revision": args.source_revision, "phase_start": 142, "math_protocol": protocol,
        }),
    ]
    protected = {path: digest for path, digest in old["protected_files"].items()
                 if not path.startswith(old["source"] + "/")
                 and "limite-sft-curve-eval" not in path
                 and path != old["stages"][-1]["template_plan"]}
    for relative, digest in source_files.items():
        protected[str(source / "src/archlab" / relative)] = digest
    for path in (recipe, eval_recipe, template_path, args.curve_plan, args.curve_config, args.tracking_receipt):
        protected[str(path.resolve())] = sha256_file(path)
    plan["protected_files"] = protected
    tracking.update(math_protocol=protocol, phase_start=142,
                    curriculum_sha256=gate["checkpoint_contract"]["curriculum_sha256"], parent_run_id=old_run)
    write_new(production / "MLFLOW.json", tracking)
    # Tracking is intentionally not hash-pinned: the trainer refreshes its
    # metadata while preserving this new run ID, which the final gate checks.
    plan_path = destination / "QUEUE_PLAN.json"
    write_new(plan_path, plan)
    validate_plan(plan)
    write_new(destination / "MIGRATION_PREPARED.json", dict(
        passed=True, activated=False, old_plan=str(args.previous_plan.resolve()),
        old_plan_sha256=sha256_file(args.previous_plan), new_plan=str(plan_path),
        new_plan_sha256=sha256_file(plan_path), source_revision=args.source_revision,
        preserved_checkpoint=argv[argv.index("--resume") + 1], mlflow_run_id=run_id,
        required_activation_actions=[
            "Stop the owned old deferred CPU controller and verify it exited without live workers.",
            "Write STOP to every retired deferred queue root and STOP_REQUEST to old RL output roots.",
            "Verify all curve and GPU qualification gates before arming this fresh controller.",
            "Start only this successor queue; read back waiting/running state and controller ownership.",
        ],
    ))
    print(json.dumps(dict(plan=str(plan_path), production=str(production), activated=False)))


if __name__ == "__main__":
    main()
