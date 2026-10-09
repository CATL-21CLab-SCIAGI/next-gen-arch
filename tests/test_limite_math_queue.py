"""CPU queue barriers, ownership, failure and automatic final-plan proofs."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.automodel.limite_math_queue import (
    FORMAT,
    REMOTE,
    QueueController,
    final_checkpoint,
    read_json,
    validate_plan,
    validate_queue,
    verify_shard,
)
from archlab.evaluation.limite_math import benchmark_seed


@pytest.fixture
def setup_queue(tmp_path):
    source = tmp_path / "source"
    module = source / "src" / "archlab" / "automodel" / "limite_math_benchmark.py"
    module.parent.mkdir(parents=True)
    module.write_text("# sealed executor\n")
    (source / "SOURCE_REVISION").write_text("sealed-revision\n")
    files = {"automodel/limite_math_benchmark.py": sha256_file(module)}
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    cases = [dict(id=f"aime:{i}", task="aime", answer=str(i), question=f"Q{i}") for i in range(19)]
    (bundle / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    atomic_write_json(bundle / "MANIFEST.json", dict(cases_sha256=sha256_file(bundle / "cases.jsonl")))
    plan = dict(format="archlab-limite-math-evaluation-v1", source=str(source), source_git_revision="sealed-revision",
                source_file_sha256=files, implementation_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                bundle=str(bundle), manifest_sha256=sha256_file(bundle / "MANIFEST.json"),
                output=str(tmp_path / "responses"), models=[dict(name="base", variant="base", kind="base")],
                sampling=dict(seed=42, samples_per_problem=2))
    plan_path = tmp_path / "PLAN.json"
    atomic_write_json(plan_path, plan)
    config = dict(format=FORMAT, output=str(tmp_path / "queue"), hosts=["master", "worker"],
                  python=sys.executable, checkpoint_cache=str(tmp_path / "cache"),
                  stages=[dict(name="base", plan=str(plan_path), plan_sha256=sha256_file(plan_path))])
    return config, plan, cases


def complete(plan, cases, shard):
    output = Path(plan["output"])
    destination = output / plan["models"][0]["name"] / f"shard-{shard}"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases[shard::16]:
        for sample in range(plan["sampling"]["samples_per_problem"]):
            rows.append(dict(problem_id=case["id"], sample_index=sample, task=case["task"],
                             expected_answer=case["answer"], model=plan["models"][0]["name"],
                             source_implementation_sha256=plan["implementation_sha256"],
                             seed=benchmark_seed(plan["sampling"]["seed"], case["id"], sample)))
    (destination / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    atomic_write_json(output / f"SHARD-{shard}-COMPLETE.json",
                      dict(shard=shard, models=[plan["models"][0]["name"]], problems=len(cases[shard::16]),
                           samples_per_problem=plan["sampling"]["samples_per_problem"], case_limit=None,
                           implementation_sha256=plan["implementation_sha256"]))


class Workers:
    def __init__(self):
        self.running = set()
        self.launched = []
        self.oom = set()
        self.lost_ack = False

    def request(self, host, request):
        shard = int(request["argv"][request["argv"].index("--shard") + 1])
        if request["action"] == "launch":
            self.running.add(shard)
            self.launched.append((host, shard, request))
            if self.lost_ack:
                self.lost_ack = False
                raise RuntimeError("lost acknowledgement")
        return dict(status="running" if shard in self.running else "exited", pid=100 + shard,
                    start_ticks=200 + shard, log_tail="torch.OutOfMemoryError: CUDA out of memory" if shard in self.oom else "")


def test_single_host_preserves_shards_seeds_and_completed_responses(setup_queue):
    config, plan, cases = setup_queue
    config["hosts"] = ["single"]
    validate_queue(config)
    complete(plan, cases, 0)
    workers = Workers()
    controller = QueueController(config, workers=workers)
    assert controller.tick() == "running"
    assert {shard for _, shard, _ in workers.launched} == set(range(1, 16))
    for host, shard, request in workers.launched:
        assert host == "single"
        assert request["env"]["CUDA_VISIBLE_DEVICES"] == str(shard % 8)
        assert request["argv"][request["argv"].index("--shards") + 1] == "16"
    assert controller.state["stages"]["base"]["slots"]["0"]["status"] == "complete"


def install_evaluation_gate(config):
    root = Path(config["output"]).parent
    gates = []
    for index in range(4):
        state = root / f"evaluation-{index}.json"
        atomic_write_json(state, dict(status="complete", stages={
            "evaluation": dict(status="complete", plan_sha256=f"plan-{index}")
        }))
        gates.append(dict(state=str(state), stages={"evaluation": f"plan-{index}"}))
    path = root / "EVALUATION-GATE.json"
    atomic_write_json(path, dict(gates=gates))
    config.update(wait_for_evaluations=str(path), wait_for_evaluations_sha256=sha256_file(path))
    return gates, path


@pytest.mark.parametrize("blocked", ["running", "failed", "memory_overflow", "wrong_hash"])
def test_no_worker_or_stage_progress_before_all_evaluations_complete(setup_queue, blocked):
    config, _, _ = setup_queue
    gates, _ = install_evaluation_gate(config)
    path = Path(gates[3]["state"])
    state = json.loads(path.read_text())
    if blocked in ("failed", "memory_overflow"):
        state["status"] = blocked
    elif blocked == "wrong_hash":
        state["stages"]["evaluation"]["plan_sha256"] = "wrong"
    else:
        state["stages"]["evaluation"]["status"] = blocked
    atomic_write_json(path, state)

    class NoWorkBeforeReady(Workers):
        def request(self, host, request):
            raise AssertionError("waiting evaluation gate must not contact workers")

    controller = QueueController(config, NoWorkBeforeReady())
    assert controller.tick() == "waiting_for_evaluations"
    assert controller.state["stage"] == 0 and controller.state["stages"] == {}
    assert read_state(controller)["status"] == "waiting_for_evaluations"
    atomic_write_json(path, dict(status="complete", stages={
        "evaluation": dict(status="complete", plan_sha256="plan-3")
    }))
    workers = Workers()
    restarted = QueueController(config, workers)
    assert restarted.tick() == "running" and len(workers.launched) == 16


def read_state(controller):
    return json.loads((controller.output / "STATE.json").read_text())


def test_evaluation_gate_is_rehashed_before_every_tick(setup_queue):
    config, _, _ = setup_queue
    _, path = install_evaluation_gate(config)
    workers = Workers()
    controller = QueueController(config, workers)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="reviewed file changed"):
        controller.tick()
    assert not workers.launched and controller.state["stage"] == 0
    with pytest.raises(ValueError, match="reviewed file changed"):
        validate_queue(config)


@pytest.mark.parametrize("missing", ["wait_for_evaluations", "wait_for_evaluations_sha256"])
def test_evaluation_gate_path_and_hash_are_paired(setup_queue, missing):
    config, _, _ = setup_queue
    install_evaluation_gate(config)
    del config[missing]
    with pytest.raises(ValueError, match="must be supplied together"):
        validate_queue(config)


@pytest.mark.parametrize("terminal", ["complete", "failed", "memory_overflow"])
def test_evaluation_gate_does_not_reopen_terminal_queue(setup_queue, terminal):
    config, _, _ = setup_queue
    _, path = install_evaluation_gate(config)
    workers = Workers()
    controller = QueueController(config, workers)
    controller.state["status"] = terminal
    controller.save()
    path.unlink()  # Terminal queues need no new gate or worker operations.
    assert controller.tick() == terminal
    assert not workers.launched and read_state(controller)["status"] == terminal


def test_launch_geometry_and_adopt_existing(setup_queue):
    config, _, _ = setup_queue
    config['env'] = {'ARCHLAB_SOURCE_REVISION': 'older-controller-default'}
    workers = Workers()
    workers.running.add(0)
    controller = QueueController(config, workers)
    assert controller.tick() == "running"
    assert len(workers.launched) == 15
    for host, shard, request in workers.launched:
        assert host == config["hosts"][shard // 8]
        assert request["env"]["CUDA_VISIBLE_DEVICES"] == str(shard % 8)
        assert request["env"]["ARCHLAB_SOURCE_REVISION"] == 'sealed-revision'
        assert request["argv"][-3:] == ["16", "--checkpoint-cache", config["checkpoint_cache"]]
    assert controller.state["stages"]["base"]["slots"]["0"]["attempts"] == 0


def test_remote_completion_waits_for_shared_storage_visibility(setup_queue):
    config, plan, cases = setup_queue

    class CompletedRemote(Workers):
        def request(self, host, request):
            assert request["action"] == "inspect", "must not relaunch completed work"
            return dict(status="exited", marker=True, log_tail="")

    controller = QueueController(config, CompletedRemote())
    assert controller.tick() == "running"
    slots = controller.state["stages"]["base"]["slots"]
    assert all(s["attempts"] == 0 and s["status"] == "waiting_for_marker_visibility" for s in slots.values())
    for shard in range(16):
        complete(plan, cases, shard)
    controller.tick()
    assert controller.state["stage"] == 1


def test_final_save_does_not_overlap_live_training_processes(setup_queue):
    config, _, _ = setup_queue
    config['wait_for_training_exit'] = True

    class TrainingStillExiting(Workers):
        def request(self, host, request):
            assert request['action'] == 'inspect'
            assert 'archlab.automodel.limite_adapter_rl' in request['wait_for_modules']
            return dict(status='waiting_for_training_exit', pids=[42])

    controller = QueueController(config, TrainingStillExiting())
    controller.tick()
    assert all(slot['attempts'] == 0 and slot['status'] == 'waiting_for_training_exit'
               for slot in controller.state['stages']['base']['slots'].values())


def test_sft_can_wait_until_normal_rl_finishes(setup_queue, monkeypatch):
    config, _, _ = setup_queue
    config["stages"][0]["wait_for_final_rl_variant"] = "normal"
    config["final_rl_gates"] = [dict(variant="normal")]
    workers = Workers()
    monkeypatch.setattr("archlab.automodel.limite_math_queue.final_checkpoint", lambda gate: None)
    controller = QueueController(config, workers)
    assert controller.tick() == "waiting_final_rl" and not workers.launched
    monkeypatch.setattr("archlab.automodel.limite_math_queue.final_checkpoint", lambda gate: {"verified": True})
    assert controller.tick() == "running" and len(workers.launched) == 16


def test_lost_ack_is_adopted_after_restart(setup_queue):
    config, _, _ = setup_queue
    workers = Workers()
    workers.lost_ack = True
    with pytest.raises(RuntimeError, match="acknowledgement"):
        QueueController(config, workers).tick()
    controller = QueueController(config, workers)
    controller.tick()
    assert [shard for _, shard, _ in workers.launched].count(0) == 1
    assert controller.state["stages"]["base"]["slots"]["0"]["attempts"] == 1


def test_stage_barrier_requires_every_record_and_worker_exit(setup_queue):
    config, plan, cases = setup_queue
    second = copy.deepcopy(plan)
    second["output"] += "-second"
    second["models"][0]["name"] = "second"
    path = Path(config["output"]).parent / "SECOND.json"
    atomic_write_json(path, second)
    config["stages"].append(dict(name="second", plan=str(path), plan_sha256=sha256_file(path)))
    for shard in range(16):
        complete(plan, cases, shard)
    workers = Workers()
    workers.running.add(15)
    controller = QueueController(config, workers)
    controller.tick()
    assert controller.state["stage"] == 0
    assert not workers.launched
    workers.running.clear()
    controller.tick()
    assert controller.state["stage"] == 1
    assert not workers.launched  # next stage has its own tick
    controller.tick()
    assert len(workers.launched) == 16


@pytest.mark.parametrize("change", ["duplicate", "missing", "wrong_seed", "wrong_source"])
def test_marker_cannot_hide_bad_records(setup_queue, change):
    _, plan, cases = setup_queue
    complete(plan, cases, 0)
    path = Path(plan["output"]) / "base" / "shard-0" / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "missing":
        rows.pop()
    elif change == "wrong_seed":
        rows[0]["seed"] += 1
    else:
        rows[0]["source_implementation_sha256"] = "unreviewed"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        verify_shard(plan, cases, 0)


def test_preserved_sealed_rows_are_accepted_only_explicitly(setup_queue):
    _, plan, cases = setup_queue
    complete(plan, cases, 0)
    path = Path(plan["output"]) / "base" / "shard-0" / "records.jsonl"
    text = path.read_text().replace(plan["implementation_sha256"], "previous-sealed")
    path.write_text(text)
    plan["accepted_record_implementation_sha256"] = ["previous-sealed"]
    assert verify_shard(plan, cases, 0)["records"] == 4
    assert path.read_text() == text


def test_oom_publishes_event_without_retry(setup_queue):
    config, _, _ = setup_queue
    workers = Workers()
    workers.oom.add(0)
    controller = QueueController(config, workers)
    assert controller.tick() == "memory_overflow"
    assert not workers.launched
    event = json.loads((Path(config["output"]) / "OOM_EVENT.json").read_text())
    assert event["shard"] == 0 and event["automatic_retry"] is False
    assert controller.tick() == "memory_overflow"
    assert not workers.launched


def test_non_oom_retries_are_bounded(setup_queue):
    config, _, _ = setup_queue
    workers = Workers()
    controller = QueueController(config, workers)
    controller.tick()
    workers.running.clear()
    controller.tick()
    workers.running.clear()
    assert controller.tick() == "failed"
    assert [shard for _, shard, _ in workers.launched].count(0) == 2


def test_queue_config_and_plan_mutations_reject(setup_queue):
    config, _, _ = setup_queue
    controller = QueueController(config, Workers())
    controller.tick()
    changed = copy.deepcopy(config)
    changed["hosts"].reverse()
    with pytest.raises(ValueError, match="configuration changed"):
        QueueController(changed, Workers())
    Path(config["stages"][0]["plan"]).write_text("{}")
    with pytest.raises(ValueError, match="reviewed file changed"):
        controller.tick()


def test_sealed_executor_is_verified_without_importing(setup_queue):
    _, plan, cases = setup_queue
    assert validate_plan(plan) == cases
    path = Path(plan["source"]) / "src" / "archlab" / "automodel" / "limite_math_benchmark.py"
    path.write_text("raise RuntimeError('never import me')\n")
    with pytest.raises(ValueError, match="files changed"):
        validate_plan(plan)


def make_final(tmp_path, variant, complete_status=True):
    root = tmp_path / variant
    parent = tmp_path / (variant + "-sft")
    atomic_write_json(parent / "COMPLETE.json", dict(tokens=10_000_000_000, trainable_mode="full", base_snapshot_sha256="publisher"))
    gate = dict(rl_root=str(root), variant=variant, target_step=400,
                warmup_checkpoint=str(parent), source_revision="final-source", mlflow_run_id="run-" + variant,
                checkpoint_contract=dict(phase_start=77, math_protocol=dict(version="math-v1"),
                                         curriculum_sha256="curriculum", rl_split_sha256="split"))
    checkpoint = root / "checkpoints" / "step-0000400"
    checkpoint.mkdir(parents=True)
    files = {}
    for name in ("adapter.pt", "backbone.pt", "optimizer.pt", "rng.pt", "rl_state.pt", "trainer_state.json"):
        (checkpoint / name).write_bytes(name.encode())
        files[name] = sha256_file(checkpoint / name)
    receipt = dict(step=400, trainable_mode="full", adapter=dict(variant=variant, attention_backend="tilelang"),
                   warmup_checkpoint=str(parent), source_revision="final-source", base_snapshot_sha256="publisher",
                   files=files, **gate["checkpoint_contract"])
    atomic_write_json(checkpoint / "COMPLETE.json", receipt)
    atomic_write_json(root / "FINISHED.json", dict(status="complete" if complete_status else "stopped", step=400,
                                                  trainable_mode="full", base_snapshot_sha256="publisher", applied_updates=390))
    atomic_write_json(root / "MLFLOW.json", dict(run_id=gate["mlflow_run_id"]))
    return gate, checkpoint


def test_final_gate_verifies_every_payload_and_lineage(tmp_path):
    gate, checkpoint = make_final(tmp_path, "normal")
    assert final_checkpoint(gate)["checkpoint"] == str(checkpoint)
    (checkpoint / "optimizer.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="payload changed"):
        final_checkpoint(gate)


def make_native_final(tmp_path):
    gate, checkpoint = make_final(tmp_path, "native")
    publisher = dict(repo="paradigma-inc/limite-1b-violetto", revision="verified-revision", receipt_sha256="verified-download")
    gate.pop("warmup_checkpoint")
    gate.update(publisher_snapshot=str(tmp_path / "verified-violetto"), publisher_identity=publisher)
    receipt = read_json(checkpoint / "COMPLETE.json")
    for key in ("adapter", "warmup_checkpoint", "base_snapshot_sha256"):
        receipt.pop(key)
    for name in ("adapter.pt", "backbone.pt"):
        receipt["files"].pop(name)
        (checkpoint / name).unlink()
    (checkpoint / "model.pt").write_bytes(b"full native model")
    receipt["files"]["model.pt"] = sha256_file(checkpoint / "model.pt")
    receipt.update(model_kind="native", publisher_identity=publisher, publisher_snapshot=gate["publisher_snapshot"])
    atomic_write_json(checkpoint / "COMPLETE.json", receipt)
    root = Path(gate["rl_root"])
    finished = read_json(root / "FINISHED.json")
    finished.pop("base_snapshot_sha256")
    finished.update(model_kind="native", publisher_identity=publisher)
    atomic_write_json(root / "FINISHED.json", finished)
    return gate, checkpoint


def test_native_final_gate_requires_full_model_optimizer_rng_and_pins_receipts(tmp_path):
    gate, checkpoint = make_native_final(tmp_path)
    proof = final_checkpoint(gate)
    assert proof["checkpoint"] == str(checkpoint)
    root = Path(gate["rl_root"])
    assert proof["finished_sha256"] == sha256_file(root / "FINISHED.json")
    assert proof["tracking_sha256"] == sha256_file(root / "MLFLOW.json")
    (checkpoint / "model.pt").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="payload changed"):
        final_checkpoint(gate)


@pytest.mark.parametrize("mutation", ["publisher", "snapshot", "mode", "adapter", "missing_optimizer", "protocol", "zero_updates", "run"])
def test_native_final_gate_rejects_invalid_resume_and_lineage(tmp_path, mutation):
    gate, checkpoint = make_native_final(tmp_path)
    receipt = read_json(checkpoint / "COMPLETE.json")
    root = Path(gate["rl_root"])
    if mutation == "publisher":
        receipt["publisher_identity"]["revision"] = "different-revision"
    elif mutation == "snapshot":
        receipt["publisher_snapshot"] = "different-publisher"
    elif mutation == "mode":
        receipt["trainable_mode"] = "adapter"
    elif mutation == "adapter":
        receipt["files"]["adapter.pt"] = "unexpected"
    elif mutation == "missing_optimizer":
        receipt["files"].pop("optimizer.pt")
    elif mutation == "protocol":
        receipt["rl_split_sha256"] = "different-split"
    elif mutation == "zero_updates":
        finished = read_json(root / "FINISHED.json")
        finished["applied_updates"] = 0
        atomic_write_json(root / "FINISHED.json", finished)
    else:
        atomic_write_json(root / "MLFLOW.json", dict(run_id="different-run"))
    atomic_write_json(checkpoint / "COMPLETE.json", receipt)
    with pytest.raises(ValueError):
        final_checkpoint(gate)


def test_final_gate_rejects_metadata_replaced_while_payloads_are_verified(tmp_path, monkeypatch):
    import archlab.automodel.limite_math_queue as queue

    gate, _ = make_native_final(tmp_path)
    finished_path = Path(gate["rl_root"]) / "FINISHED.json"
    hash_file = queue.sha256_file
    def replacing_receipt(path):
        result = hash_file(path)
        if Path(path).name == "optimizer.pt":
            finished = read_json(finished_path)
            finished["applied_updates"] -= 1
            atomic_write_json(finished_path, finished)
        return result
    monkeypatch.setattr(queue, "sha256_file", replacing_receipt)
    with pytest.raises(ValueError, match="metadata changed during payload verification"):
        final_checkpoint(gate)


@pytest.mark.parametrize("field,value", [("trainable_mode", "adapter"), ("source_revision", "other"),
                                         ("warmup_checkpoint", "other"), ("base_snapshot_sha256", "other")])
def test_final_gate_rejects_other_training_lineage(tmp_path, field, value):
    gate, checkpoint = make_final(tmp_path, "normal")
    receipt = json.loads((checkpoint / "COMPLETE.json").read_text())
    receipt[field] = value
    atomic_write_json(checkpoint / "COMPLETE.json", receipt)
    with pytest.raises(ValueError):
        final_checkpoint(gate)


def test_final_plan_waits_for_both_then_materializes_and_preserves_registry(setup_queue):
    config, plan, _ = setup_queue
    root = Path(config["output"]).parent
    normal, normal_checkpoint = make_final(root, "normal")
    sim, _ = make_final(root, "simplicial", complete_status=False)
    template = copy.deepcopy(plan)
    template["models"][0].update(name="old-partial", variant="normal", kind="rl", step=157)
    template["models"][0].update(checkpoint=str(normal_checkpoint),
                                checkpoint_receipt_sha256=sha256_file(normal_checkpoint / "COMPLETE.json"))
    template_path = root / "TEMPLATE.json"
    atomic_write_json(template_path, template)
    config["stages"] = [dict(name="rl-normal", template_plan=str(template_path),
                             template_plan_sha256=sha256_file(template_path), final_variant="normal",
                             plan_path=str(root / "FINAL.json"), output_prefix=str(root / "final-responses"))]
    config["final_rl_gates"] = [normal, sim]
    registry = root / "REGISTRY.json"
    original = dict(models=[dict(name="old-partial", plan="original")], interpretation=["retain operational subset"])
    atomic_write_json(registry, original)
    config["report_registry"] = str(registry)
    workers = Workers()
    controller = QueueController(config, workers)
    assert controller.tick() == "waiting_final_rl"
    assert not workers.launched and not (root / "FINAL.json").exists()
    finished_path = Path(sim["rl_root"]) / "FINISHED.json"
    finished = json.loads(finished_path.read_text())
    finished["status"] = "complete"
    atomic_write_json(finished_path, finished)
    controller.tick()
    final = json.loads((root / "FINAL.json").read_text())
    assert final["sampling"] == template["sampling"]
    assert final["source_file_sha256"] == template["source_file_sha256"]
    assert final["models"][0]["step"] == 400
    assert final["models"][0]["name"] == "rl-normal-final-step400"
    result = json.loads(registry.read_text())
    assert result["models"][0] == original["models"][0]
    assert result["interpretation"] == original["interpretation"]
    assert len(workers.launched) == 16
    assert json.loads(template_path.read_text()) == template


@pytest.mark.parametrize('variant', ['normal', 'simplicial'])
def test_independent_final_queue_never_waits_for_peer(setup_queue, variant):
    config, plan, _ = setup_queue
    root = Path(config['output']).parent
    gate, checkpoint = make_final(root, variant)
    template = copy.deepcopy(plan)
    template['models'][0].update(name='template', variant=variant, kind='rl', step=174,
                                 checkpoint=str(checkpoint), checkpoint_receipt_sha256=sha256_file(checkpoint / 'COMPLETE.json'))
    path = root / 'TEMPLATE.json'
    atomic_write_json(path, template)
    config.update(final_rl_gate_mode='own_variant', final_rl_gates=[gate])
    config['stages'] = [dict(name='rl-' + variant, template_plan=str(path), template_plan_sha256=sha256_file(path),
                             final_variant=variant, plan_path=str(root / 'FINAL.json'), output_prefix=str(root / 'responses-final'))]
    workers = Workers()
    controller = QueueController(config, workers)
    assert controller.tick() == 'running'
    assert len(workers.launched) == 16
    assert set(controller.state['final_rl_proofs']) == {variant}


def remote(request):
    result = subprocess.run([sys.executable, "-c", REMOTE], input=json.dumps(request),
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def test_remote_exact_process_adoption_and_wrong_environment(tmp_path):
    # Inert CPU child: the real ownership helper scans argv, not substrings.
    argv = [sys.executable, "-c", "import time; time.sleep(20)", "unique-queue-ownership-token"]
    env = os.environ.copy() | {"QUEUE_TEST_GPU": "3"}
    process = subprocess.Popen(argv, env=env)
    request = dict(action="inspect", slot=str(tmp_path / "slot"), marker=str(tmp_path / "missing"),
                   env={"QUEUE_TEST_GPU": "3"}, argv=argv, cwd=str(tmp_path))
    try:
        observed = remote(request)
        assert observed["pid"] == process.pid and observed["status"] == "running"
        assert observed["start_ticks"] > 0
        request["env"]["QUEUE_TEST_GPU"] = "4"
        with pytest.raises(RuntimeError, match="different environment"):
            remote(request)
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_remote_substring_in_another_process_cannot_match(tmp_path):
    request = dict(action="inspect", slot=str(tmp_path / "slot"), marker=str(tmp_path / "missing"),
                   env={}, argv=[sys.executable, "-m", "nonexistent-unique-queue-worker"], cwd=str(tmp_path))
    observed = remote(request)
    assert observed["status"] == "exited"


def test_invalid_geometry_and_controlled_environment(setup_queue):
    config, _, _ = setup_queue
    config["hosts"] = ["same", "same"]
    with pytest.raises(ValueError, match="distinct"):
        validate_queue(config)
    config["hosts"] = ["master", "worker"]
    config["env"] = {"CUDA_VISIBLE_DEVICES": "all"}
    with pytest.raises(ValueError, match="controller-owned"):
        validate_queue(config)


def test_watch_logs_ignore_historical_oom_and_hold_on_new_oom(setup_queue):
    config, _, _ = setup_queue
    path = Path(config["output"]).parent / "rl.log"
    path.write_text("historical CUDA out of memory\n")
    config["watch_logs"] = [str(path)]
    workers = Workers()
    controller = QueueController(config, workers)
    assert controller.tick() == "running"
    assert len(workers.running) == 16
    restarted = QueueController(config, workers)
    assert restarted.tick() == "running"
    with path.open("a") as stream:
        stream.write("new CUDA out of memory\n")
    assert restarted.tick() == "memory_overflow"
    assert len(workers.running) == 16  # existing workers are retained
    event = json.loads((Path(config["output"]) / "OOM_EVENT.json").read_text())
    assert event["scope"] == "watched_training_log" and event["log"] == str(path)


def test_watch_log_oom_can_cross_read_boundary(setup_queue):
    config, _, _ = setup_queue
    path = Path(config["output"]).parent / "rl.log"
    path.write_text("")
    config["watch_logs"] = [str(path)]
    controller = QueueController(config, Workers())
    path.write_text("x" * (65536 - 6) + "CUDA out of memory\n")
    assert controller.tick() == "memory_overflow"


@pytest.mark.parametrize("field", ["math_protocol", "curriculum_sha256", "rl_split_sha256", "phase_start"])
def test_final_gate_verifies_protocol_and_phase(tmp_path, field):
    gate, checkpoint = make_final(tmp_path, "normal")
    receipt = json.loads((checkpoint / "COMPLETE.json").read_text())
    receipt[field] = "different"
    atomic_write_json(checkpoint / "COMPLETE.json", receipt)
    with pytest.raises(ValueError, match="protocol/curriculum/split/phase"):
        final_checkpoint(gate)


@pytest.mark.parametrize("mutation", [None, "new_heartbeat", "young", "pid", "start_ticks", "marker", "minimum_age"])
def test_remote_stale_recovery_revalidates_identity_and_progress(tmp_path, mutation):
    argv = [sys.executable, "-c", "import time; time.sleep(30)", str(tmp_path)]
    process = subprocess.Popen(argv, env=os.environ.copy() | {"QUEUE_TEST_GPU": "5"})
    request = dict(action="inspect", slot=str(tmp_path / "slot"), marker=str(tmp_path / "complete"),
                   env={"QUEUE_TEST_GPU": "5"}, argv=argv, cwd=str(tmp_path))
    try:
        observed = remote(request)
        current = tmp_path / "CURRENT.json"
        heartbeat = dict(pid=process.pid, status="generating", generated_tokens=1024, time=time.time() - 1900)
        if mutation == "young":
            heartbeat["time"] = time.time()
        if mutation == "pid":
            heartbeat["pid"] += 1
        atomic_write_json(current, heartbeat)
        request.update(action="recover_stale", expected_identity={
            key: observed[key] for key in ("pid", "start_ticks", "argv", "env")
        }, heartbeat=dict(path=str(current), sha256=sha256_file(current), minimum_age=1800))
        if mutation == "new_heartbeat":
            atomic_write_json(current, dict(heartbeat, generated_tokens=2048))
        if mutation == "start_ticks":
            request["expected_identity"]["start_ticks"] += 1
        if mutation == "marker":
            Path(request["marker"]).write_text("{}")
        if mutation == "minimum_age":
            request["heartbeat"]["minimum_age"] = 1
        result = remote(request)
        if mutation is None:
            assert result["status"] == "stop_requested"
            assert process.wait(timeout=5) == -15
        else:
            assert result["status"] == "running"
            assert result["recovery_refused"] == "identity_or_progress_changed"
            assert process.poll() is None
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


class LocatedWorkers:
    """Each real worker is distinct by stage, host, GPU and full sealed argv."""

    def __init__(self):
        self.running = {}
        self.launched = []
        self.stopped = []
        self.lost_recovery_ack = False

    @staticmethod
    def key(host, request):
        return host, tuple(request["argv"]), request["env"]["CUDA_VISIBLE_DEVICES"]

    def request(self, host, request):
        key = self.key(host, request)
        if request["action"] == "launch":
            self.launched.append((host, request))
            self.running[key] = dict(pid=1000 + len(self.launched), start_ticks=2000 + len(self.launched),
                                     argv=request["argv"], env=request["env"])
        if request["action"] == "recover_stale":
            self.stopped.append((host, request))
            previous = self.running.pop(key)
            if self.lost_recovery_ack:
                raise RuntimeError("lost recovery acknowledgement")
            return dict(previous, status="stop_requested")
        return dict(self.running[key], status="running") if key in self.running else dict(status="exited")


def add_static_stage(config, plan, name="second"):
    follow = copy.deepcopy(plan)
    follow["models"][0]["name"] = name
    follow["output"] += "-" + name
    path = Path(config["output"]).parent / f"PLAN-{name}.json"
    atomic_write_json(path, follow)
    config["stages"].append(dict(name=name, plan=str(path), plan_sha256=sha256_file(path)))
    return follow


def test_backfill_uses_idle_capacity_and_retains_live_slot_placement(setup_queue):
    config, plan, cases = setup_queue
    config.update(backfill_static_stages=True, gpu_slots={"master": [1], "worker": [4]}, workers_per_gpu=2)
    follow = add_static_stage(config, plan)
    validate_queue(config)
    for shard in range(16):
        if shard != 9:
            complete(plan, cases, shard)
    workers = LocatedWorkers()
    controller = QueueController(config, workers)
    state = controller.state["stages"].setdefault("base", {})
    controller.plan_for(config["stages"][0], state)
    state["slots"]["9"] = dict(attempts=0, host="master", gpu=1, status="running")
    request = controller.worker_request(config["stages"][0], plan, 9, "launch")
    workers.request("master", request)
    workers.launched.clear()
    assert controller.tick() == "running"
    assert len(workers.running) == 4 and len(workers.launched) == 3
    assert controller.state["stage"] == 0
    assert state["slots"]["9"]["gpu"] == 1 and state["slots"]["9"]["host"] == "master"
    slots = controller.state["stages"]["second"]["slots"]
    assert sum(slot["status"] == "running" for slot in slots.values()) == 3
    assert sum(slot["status"] == "waiting_for_gpu" for slot in slots.values()) == 13
    assert all(request["env"]["CUDA_VISIBLE_DEVICES"] in ("1", "4") for _, request in workers.launched)
    # A later fully finished stage cannot advance the dependency cursor or RL gate.
    for shard in range(16):
        complete(follow, cases, shard)
    workers.running = {key: value for key, value in workers.running.items() if key[1] == tuple(request["argv"])}
    assert controller.tick() == "running"
    assert controller.state["stage"] == 0
    assert controller.state["stages"]["second"]["status"] == "complete"


def test_stale_recovery_keeps_records_and_has_durable_bounded_intent(setup_queue):
    config, plan, cases = setup_queue
    config.update(stale_heartbeat_seconds=1800, max_stale_restarts=1)
    for shard in range(1, 16):
        complete(plan, cases, shard)
    workers = LocatedWorkers()
    controller = QueueController(config, workers)
    controller.tick()
    slot = controller.state["stages"]["base"]["slots"]["0"]
    current = Path(plan["output"]) / "base" / "shard-0" / "CURRENT.json"
    record = current.with_name("records.jsonl")
    record.parent.mkdir(parents=True)
    record.write_text("preserved response payload\n")
    atomic_write_json(current, dict(pid=slot["observed"]["pid"], status="generating", time=time.time() - 1900))
    workers.lost_recovery_ack = True
    with pytest.raises(RuntimeError, match="lost recovery acknowledgement"):
        controller.tick()
    persisted = read_state(controller)["stages"]["base"]["slots"]["0"]
    assert persisted["stale_restarts"] == 1 and persisted["status"] == "stop_requested"
    assert record.read_text() == "preserved response payload\n"
    workers.lost_recovery_ack = False
    controller = QueueController(config, workers)
    assert controller.tick() == "running"
    assert len(workers.launched) == 2 and len(workers.stopped) == 1
    slot = controller.state["stages"]["base"]["slots"]["0"]
    atomic_write_json(current, dict(pid=slot["observed"]["pid"], status="generating", time=time.time() - 1900))
    assert controller.tick() == "failed"
    assert len(workers.stopped) == 1 and record.read_text() == "preserved response payload\n"


@pytest.mark.parametrize("field,value", [("stale_heartbeat_seconds", 1799), ("stale_heartbeat_seconds", float("nan")),
                                        ("max_stale_restarts", 3), ("workers_per_gpu", 3),
                                        ("backfill_static_stages", 1)])
def test_invalid_recovery_and_backfill_settings_reject(setup_queue, field, value):
    config, _, _ = setup_queue
    config[field] = value
    with pytest.raises(ValueError):
        validate_queue(config)
