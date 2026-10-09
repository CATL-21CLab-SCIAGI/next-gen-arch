import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from archlab.artifacts import sha256_file
from archlab.automodel.limite_rl_queue import (
    BOOTSTRAP,
    FORMAT,
    PersistentRLQueue,
    gate_ready,
    validate_plan,
)
from archlab.evaluation.limite_math import benchmark_seed


def test_nested_runtime_bootstrap_compiles_and_writes_real_constraint_lines():
    outer = ast.parse(BOOTSTRAP)
    code = next(ast.literal_eval(node.value) for node in outer.body
                if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "code" for x in node.targets))
    inner = ast.parse(code)
    statement = next(node for node in inner.body if isinstance(node, ast.Expr)
                     and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Attribute)
                     and node.value.func.attr == "write_text")
    result = []
    class Constraints:
        def write_text(self, value):
            result.append(value)
    exec(compile(ast.Module(body=[statement], type_ignores=[]), "bootstrap", "exec"),
         dict(constraints=Constraints(), before={"torch": "2.12", "triton": "3.6"}))
    assert result == ["torch==2.12\ntriton==3.6\n"]


def test_gate_rejects_false_evidence_and_waits_for_missing_peer(tmp_path):
    path = tmp_path / "proof.json"
    gate = dict(path=str(path), equals={"passed": True, "checkpoint.rng": True}, positive=["norm"])
    assert not gate_ready([gate])
    path.write_text(json.dumps(dict(passed=True, checkpoint=dict(rng=True), norm=1.)))
    assert gate_ready([gate])
    path.write_text(json.dumps(dict(passed=True, checkpoint=dict(rng=False), norm=1.)))
    with pytest.raises(ValueError, match="qualification differs"):
        gate_ready([gate])
    path.write_text(json.dumps(dict(passed=True, checkpoint=dict(rng=True), norm=0.)))
    with pytest.raises(ValueError, match="no signal"):
        gate_ready([gate])


def test_replay_gate_requires_both_training_and_evaluation(tmp_path):
    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps(dict(phase="train", batch_size=1))+"\n")
    gate = dict(path=str(path), jsonl=True,
                contains=[dict(phase="train", batch_size=1), dict(phase="eval", batch_size=4)])
    with pytest.raises(ValueError, match="trajectory is incomplete"):
        gate_ready([gate])
    with path.open("a") as stream:
        stream.write(json.dumps(dict(phase="eval", batch_size=4))+"\n")
    assert gate_ready([gate])


@pytest.mark.parametrize("targets", [1_000_000_000, 1_000_000_001, 1_000_131_071])
def test_completion_gate_accepts_bounded_last_update_overshoot(tmp_path, targets):
    completion = tmp_path / "COMPLETE.json"
    contract = tmp_path / "RUN_CONTRACT.json"
    completion.write_text(json.dumps(dict(passed=True, supervised_tokens=targets)))
    contract.write_text(json.dumps(dict(target_supervised_tokens=1_000_000_000)))
    gates = [dict(path=str(completion), equals={"passed": True},
                  minimum={"supervised_tokens": 1_000_000_000},
                  maximum={"supervised_tokens": 1_000_131_071}),
             dict(path=str(contract), equals={"target_supervised_tokens": 1_000_000_000})]
    assert gate_ready(gates)
    contract.write_text(json.dumps(dict(target_supervised_tokens=targets + 1)))
    with pytest.raises(ValueError, match="qualification differs"):
        gate_ready(gates)


@pytest.mark.parametrize("targets", [999_999_999, 1_000_131_072])
def test_completion_gate_rejects_incomplete_or_excessive_token_count(tmp_path, targets):
    path = tmp_path / "COMPLETE.json"
    path.write_text(json.dumps(dict(supervised_tokens=targets)))
    with pytest.raises(ValueError, match="outside"):
        gate_ready([dict(path=str(path), minimum={"supervised_tokens": 1_000_000_000},
                        maximum={"supervised_tokens": 1_000_131_071})])


def test_numeric_bounds_support_dotted_fields_and_wait_for_missing_receipt(tmp_path):
    path = tmp_path / "COMPLETE.json"
    gate = dict(path=str(path), minimum={"cursor.supervised_tokens": 3.5},
                maximum={"cursor.supervised_tokens": 4.5})
    assert not gate_ready([gate])
    path.write_text(json.dumps(dict(cursor=dict(supervised_tokens=4.25))))
    assert gate_ready([gate])


@pytest.mark.parametrize("value", [True, False, None, "1000000000", [], {},
                                   float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("bound", ["minimum", "maximum"])
def test_numeric_bounds_reject_invalid_receipt_values(tmp_path, value, bound):
    path = tmp_path / "COMPLETE.json"
    path.write_text(json.dumps(dict(cursor=dict(supervised_tokens=value))))
    with pytest.raises(ValueError, match="invalid numeric evidence"):
        gate_ready([dict(path=str(path), **{bound: {"cursor.supervised_tokens": 1_000_000_000}})])


@pytest.mark.parametrize("row", [{}, {"cursor": None}, {"cursor": []},
                                 {"cursor": {}}, [], None])
def test_numeric_bounds_reject_missing_or_corrupt_dotted_fields(tmp_path, row):
    path = tmp_path / "COMPLETE.json"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="missing numeric evidence"):
        gate_ready([dict(path=str(path), minimum={"cursor.supervised_tokens": 1})])


@pytest.mark.parametrize("value", [True, False, None, "1", float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("bound", ["minimum", "maximum"])
def test_numeric_bounds_reject_invalid_limits_even_without_receipt(tmp_path, value, bound):
    with pytest.raises(ValueError, match="invalid qualification bound"):
        gate_ready([dict(path=str(tmp_path / "missing.json"), **{bound: {"supervised_tokens": value}})])


@pytest.mark.parametrize("limits", [{"minimum": []}, {"maximum": None},
                                    {"minimum": {"": 1}}, {"maximum": {"cursor..tokens": 1}},
                                    {"minimum": {1: 1}}])
def test_numeric_bounds_reject_corrupt_constraint_maps(tmp_path, limits):
    with pytest.raises(ValueError, match="invalid qualification bound"):
        gate_ready([dict(path=str(tmp_path / "missing.json"), **limits)])


def test_numeric_bounds_reject_inverted_interval(tmp_path):
    with pytest.raises(ValueError, match="inverted qualification bounds"):
        gate_ready([dict(path=str(tmp_path / "missing.json"), minimum={"tokens": 2},
                        maximum={"tokens": 1})])


def make_plan(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "SOURCE_REVISION").write_text("reviewed")
    stage = dict(name="production", kind="production", world=8, port=30584,
                 module="archlab.automodel.limite_adapter_rl", args=["--variant", "native"],
                 marker=str(tmp_path / "FINISHED.json"),
                 gates=[dict(path=str(tmp_path / "FINISHED.json"), equals={"status": "complete", "step": 400})])
    plan = dict(format=FORMAT, alias="owned-native", source=str(source), source_revision="reviewed",
                protected_files={}, stages=[stage], output=str(tmp_path / "queue"),
                python="/opt/venv/bin/python", env={}, pythonpath="", container_image="reviewed-image")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return {**plan, "plan_path": str(path)}


def test_controller_adopts_worker_and_finishes_without_cloud_release(tmp_path):
    plan = make_plan(tmp_path)
    calls = []
    class Workers:
        def request(self, alias, request):
            calls.append(request)
            return dict(status="running" if len(calls) == 1 else "exited", adopted=True)
    queue = PersistentRLQueue(plan, workers=Workers(), setup=lambda: True)
    assert queue.tick() == "running"
    assert calls[0]["action"] == "launch"
    (tmp_path / "FINISHED.json").write_text(json.dumps(dict(status="complete", step=400)))
    assert queue.tick() == "complete"
    assert calls[1]["action"] == "inspect"
    assert queue.tick() == "complete"
    assert len(calls) == 2
    assert queue.state["plan_sha256"] == sha256_file(plan["plan_path"])


def test_exited_worker_without_qualification_is_not_relaunched(tmp_path):
    plan = make_plan(tmp_path)
    calls = []
    class Workers:
        def request(self, alias, request):
            calls.append(request)
            return dict(status="exited")
    queue = PersistentRLQueue(plan, workers=Workers(), setup=lambda: True)
    with pytest.raises(ValueError, match="exited without complete qualification"):
        queue.tick()
    assert queue.state["workers"]["production"]["status"] == "exited"
    queue.publish("blocked")
    assert queue.tick() == "blocked"
    assert len(calls) == 1


def test_remote_completion_waits_for_local_receipt_visibility_without_relaunch(tmp_path):
    plan = make_plan(tmp_path)
    calls = []
    class Workers:
        def request(self, alias, request):
            calls.append(copy.deepcopy(request))
            return dict(status="exited", marker=True)
    queue = PersistentRLQueue(plan, workers=Workers(), setup=lambda: True)
    assert queue.tick() == "waiting_for_marker_visibility"
    assert queue.tick() == "waiting_for_marker_visibility"
    assert [call["action"] for call in calls] == ["launch", "inspect"]
    (tmp_path / "FINISHED.json").write_text(json.dumps(dict(status="complete", step=400)))
    assert queue.tick() == "complete"
    assert [call["action"] for call in calls] == ["launch", "inspect", "inspect"]


def test_remote_marker_does_not_accept_contradictory_qualification(tmp_path):
    plan = make_plan(tmp_path)
    (tmp_path / "FINISHED.json").write_text(json.dumps(dict(status="complete", step=399)))
    class Workers:
        def request(self, alias, request):
            return dict(status="exited", marker=True)
    queue = PersistentRLQueue(plan, workers=Workers(), setup=lambda: True)
    with pytest.raises(ValueError, match="qualification differs"):
        queue.tick()


def test_exited_worker_without_remote_marker_blocks_when_receipt_is_missing(tmp_path):
    plan = make_plan(tmp_path)
    class Workers:
        def request(self, alias, request):
            return dict(status="exited", marker=False)
    queue = PersistentRLQueue(plan, workers=Workers(), setup=lambda: True)
    with pytest.raises(ValueError, match="exited without complete qualification"):
        queue.tick()


def make_evaluation_plan(tmp_path):
    plan = make_plan(tmp_path)
    source = Path(plan["source"])
    module = source / "src/archlab/automodel/limite_pipeline_benchmark.py"
    module.parent.mkdir(parents=True)
    module.write_text("# sealed native checkpoint executor\n")
    files = {"automodel/limite_pipeline_benchmark.py": sha256_file(module)}
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    cases = [dict(id=f"aime26:{i}", task="aime26", question=f"Q{i}", answer=str(i)) for i in range(30)]
    (bundle / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in cases))
    (bundle / "MANIFEST.json").write_text(json.dumps(dict(cases_sha256=sha256_file(bundle / "cases.jsonl"))))
    dependency = tmp_path / "pipeline/custom_eval/eval_aime26.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("# pinned AIME26 dependency\n")
    publisher = dict(repo="paradigma-inc/limite-1b-violetto", revision="publisher-revision", receipt_sha256="publisher-digest")
    root = tmp_path / "production"
    checkpoint = root / "checkpoints/step-0000400"
    checkpoint.mkdir(parents=True)
    payloads = ["model.pt", "optimizer.pt", "rng.pt", "rl_state.pt", "trainer_state.json"]
    for name in payloads:
        (checkpoint / name).write_bytes(name.encode())
    contract = dict(phase_start=0, math_protocol={"name": "qualified"}, curriculum_sha256="curriculum", rl_split_sha256="split")
    receipt = dict(step=400, model_kind="native", trainable_mode="full", publisher_snapshot=str(tmp_path / "publisher"),
                   publisher_identity=publisher, source_revision="training-source", **contract,
                   files={name: sha256_file(checkpoint / name) for name in payloads})
    (checkpoint / "COMPLETE.json").write_text(json.dumps(receipt))
    finished = dict(status="complete", step=400, applied_updates=350, model_kind="native", trainable_mode="full",
                    publisher_identity=publisher)
    (root / "MLFLOW.json").write_text(json.dumps(dict(run_id="existing-rl-run")))
    gate = dict(rl_root=str(root), target_step=400, variant="native", source_revision="training-source",
                publisher_snapshot=receipt["publisher_snapshot"], publisher_identity=publisher,
                checkpoint_contract=contract, mlflow_run_id="existing-rl-run")
    template = dict(format="archlab-limite-math-evaluation-v1", source=str(source), source_git_revision="reviewed",
                    source_file_sha256=files,
                    implementation_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                    bundle=str(bundle), manifest_sha256=sha256_file(bundle / "MANIFEST.json"),
                    model=receipt["publisher_snapshot"], tokenizer=receipt["publisher_snapshot"],
                    output=str(tmp_path / "responses"), decode_mode="native_eager", evaluation_backend="eval_pipeline",
                    models=[dict(name="violetto-native-rl400-full-context", variant="native", kind="rl")],
                    sampling=dict(seed=20261002, samples_per_problem=4, temperature=.6, top_p=.95, top_k=0,
                                  context_limit=131072, max_new_tokens=131072, budget_policy="native_context_minus_prompt",
                                  repetition_watchdog=False, eos_token_ids=[151643, 151645]),
                    eval_pipeline=dict(source=str(dependency.parents[1]), upstream_revision="pinned", integration_patch_sha256="patch",
                                       files_sha256={"custom_eval/eval_aime26.py": sha256_file(dependency)}))
    template_path = tmp_path / "TEMPLATE.json"
    template_path.write_text(json.dumps(template))
    stage = dict(name="post-rl-aime26", kind="math_queue", template_plan=str(template_path),
                 template_plan_sha256=sha256_file(template_path), plan_path=str(tmp_path / "FINAL_PLAN.json"),
                 checkpoint_gate=gate, checkpoint_cache="/tmp/node-local-checkpoint-cache",
                 preconditions=[dict(path=str(root / "FINISHED.json"), equals={"status": "complete", "step": 400})])
    plan["stages"] = [stage]
    Path(plan["plan_path"]).write_text(json.dumps({k: v for k, v in plan.items() if k != "plan_path"}))
    return plan, template, cases, finished


def finish_evaluation_shard(template, cases, shard):
    phase = template["models"][0]
    output = Path(template["output"])
    rows = []
    for case in cases[shard::16]:
        for sample in range(4):
            rows.append(dict(problem_id=case["id"], sample_index=sample, task="aime26", expected_answer=case["answer"],
                             model=phase["name"], source_implementation_sha256=template["implementation_sha256"],
                             seed=benchmark_seed(20261002, case["id"], sample)))
    records = output / phase["name"] / f"shard-{shard}/records.jsonl"
    records.parent.mkdir(parents=True)
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / f"SHARD-{shard}-COMPLETE.json").write_text(json.dumps(dict(
        shard=shard, models=[phase["name"]], problems=len(cases[shard::16]), samples_per_problem=4,
        case_limit=None, implementation_sha256=template["implementation_sha256"])))


def test_dependent_evaluation_waits_before_runtime_or_ssh_then_reuses_sealed_shards(tmp_path):
    plan, template, cases, finished = make_evaluation_plan(tmp_path)
    calls, setups = [], []
    class Workers:
        def request(self, alias, request):
            calls.append((alias, copy.deepcopy(request)))
            return dict(status="running" if request["action"] == "launch" else "exited")
    queue = PersistentRLQueue(plan, workers=Workers(), setup=lambda: setups.append(True) or True)
    assert queue.tick() == "waiting_for_preconditions"
    assert not calls and not setups
    Path(plan["stages"][0]["checkpoint_gate"]["rl_root"]).joinpath("FINISHED.json").write_text(json.dumps(finished))
    assert queue.tick() == "running"
    launched = [(host, request) for host, request in calls if request["action"] == "launch"]
    assert len(launched) == 16
    assert {host for host, _ in launched} == {plan["alias"]}
    assert [r["env"]["CUDA_VISIBLE_DEVICES"] for _, r in launched] == [str(i % 8) for i in range(16)]
    assert all("torch.distributed.run" not in r["argv"] and "archlab.automodel.limite_pipeline_benchmark" in r["argv"] for _, r in launched)
    assert all(r["wait_for_modules"] == ["archlab.automodel.limite_adapter_rl", "archlab.automodel.limite_full_rl"] for _, r in launched)
    materialized = json.loads(Path(plan["stages"][0]["plan_path"]).read_text())
    assert materialized["sampling"] == template["sampling"]
    assert materialized["models"][0]["checkpoint"].endswith("step-0000400")
    assert materialized["models"][0]["applied_rl_updates"] == 350
    for shard in range(16):
        finish_evaluation_shard(template, cases, shard)
    assert queue.tick() == "running"  # Existing math queue seals every response before closing.
    resumed = PersistentRLQueue(plan, workers=Workers(), setup=lambda: True)
    assert resumed.tick() == "complete"
    assert resumed.state["workers"]["post-rl-aime26"]["mlflow_run_id"] == "existing-rl-run"
    assert sum(slot["proof"]["records"] for slot in resumed.evaluations["post-rl-aime26"].state["stages"]["post-rl-aime26"]["slots"].values()) == 120


@pytest.mark.parametrize("change", ["cap", "draws", "eos", "variant", "tokenizer"])
def test_post_rl_template_rejects_changed_protocol_or_loader(tmp_path, change):
    plan, template, _, _ = make_evaluation_plan(tmp_path)
    if change == "cap":
        template["sampling"]["max_new_tokens"] = 16384
    elif change == "draws":
        template["sampling"]["samples_per_problem"] = 1
    elif change == "eos":
        template["sampling"]["eos_token_ids"] = [151645]
    elif change == "variant":
        template["models"][0]["variant"] = "base"
    else:
        template["tokenizer"] = "/different/tokenizer"
    stage = plan["stages"][0]
    Path(stage["template_plan"]).write_text(json.dumps(template))
    stage["template_plan_sha256"] = sha256_file(stage["template_plan"])
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_preconditions_wait_before_generic_training_and_allow_stages_after_production(tmp_path):
    plan = make_plan(tmp_path)
    first = plan["stages"][0]
    next_stage = copy.deepcopy(first)
    next_stage.update(name="scratch-qualification", kind="qualification", world=1,
                      preconditions=[dict(path=str(tmp_path / "PRIOR_EVAL.json"), equals={"status": "complete"})])
    plan["stages"].append(next_stage)
    validate_plan(plan)
    calls = []
    class Workers:
        def request(self, alias, request):
            calls.append(request)
            return dict(status="exited")
    queue = PersistentRLQueue(plan, workers=Workers(), setup=lambda: True)
    (tmp_path / "FINISHED.json").write_text(json.dumps(dict(status="complete", step=400)))
    assert queue.tick() == "ready"
    assert queue.tick() == "waiting_for_preconditions"
    assert len(calls) == 1


def test_evaluation_waits_for_final_checkpoint_even_without_explicit_preconditions(tmp_path):
    plan, _, _, _ = make_evaluation_plan(tmp_path)
    plan["stages"][0].pop("preconditions")
    queue = PersistentRLQueue(plan, setup=lambda: pytest.fail("unfinished RL must not bootstrap runtime"))
    assert queue.tick() == "waiting_final_rl"
    assert not Path(plan["stages"][0]["plan_path"]).exists()


def test_evaluation_rejects_changed_final_payload_before_contacting_node(tmp_path):
    plan, _, _, finished = make_evaluation_plan(tmp_path)
    root = Path(plan["stages"][0]["checkpoint_gate"]["rl_root"])
    (root / "FINISHED.json").write_text(json.dumps(finished))
    (root / "checkpoints/step-0000400/model.pt").write_bytes(b"corrupt model")
    queue = PersistentRLQueue(plan, setup=lambda: pytest.fail("invalid final save must not bootstrap runtime"))
    with pytest.raises(ValueError, match="payload changed"):
        queue.tick()


def test_resumed_evaluation_rejects_changed_finished_proof(tmp_path):
    plan, _, _, finished = make_evaluation_plan(tmp_path)
    root = Path(plan["stages"][0]["checkpoint_gate"]["rl_root"])
    (root / "FINISHED.json").write_text(json.dumps(finished))
    class Workers:
        def request(self, alias, request):
            return dict(status="running")
    queue = PersistentRLQueue(plan, workers=Workers(), setup=lambda: True)
    assert queue.tick() == "running"
    finished["applied_updates"] -= 1
    (root / "FINISHED.json").write_text(json.dumps(finished))
    resumed = PersistentRLQueue(plan, workers=Workers(), setup=lambda: pytest.fail("changed proof must block before runtime setup"))
    with pytest.raises(ValueError, match="reviewed file changed"):
        resumed.tick()


def test_two_gpu_stage_isolates_kernel_overlay_without_changing_common_runtime(tmp_path):
    plan = make_plan(tmp_path)
    plan["pythonpath"] = "/shared/native-runtime"
    stage = plan["stages"][0]
    stage.update(world=2, pythonpath="/isolated/cute/dsl_packages:/isolated/cute:/upstream/triton",
                 env={"NGA_KERNEL_BACKEND": "cute"})
    calls = []
    class Workers:
        def request(self, alias, request):
            calls.append(copy.deepcopy(request))
            return dict(status="running")
    queue = PersistentRLQueue(plan, workers=Workers(), setup=lambda: True)
    assert queue.tick() == "running"
    request = calls[0]
    assert "--nproc_per_node=2" in request["argv"]
    assert request["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert request["env"]["PYTHONPATH"] == plan["source"] + "/src:" + stage["pythonpath"]
    assert request["env"]["NGA_KERNEL_BACKEND"] == "cute"
    assert plan["pythonpath"] == "/shared/native-runtime"
    assert request["env"]["ARCHLAB_SOURCE_REVISION"] == "reviewed"


@pytest.mark.parametrize("key", ["CUDA_VISIBLE_DEVICES", "PYTHONPATH", "ARCHLAB_SOURCE_REVISION", "ARCHLAB_CONTAINER_IMAGE"])
def test_stage_environment_cannot_change_worker_identity(tmp_path, key):
    plan = make_plan(tmp_path)
    plan["stages"][0]["env"] = {key: "different"}
    with pytest.raises(ValueError, match="execution identity"):
        validate_plan(plan)
