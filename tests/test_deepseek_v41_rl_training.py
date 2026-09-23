import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from archlab.artifacts import sha256_file
from archlab.automodel.deepseek_v41_rl_training import (
    admit_qualification,
    component_gate,
    configure_numerical_precision,
    digest,
    encode_examples,
    initial_cursor,
    length_bucket_order,
    load_data,
    make_contract,
    preflight_rl_checkpoint,
    rank_groups,
    read_recipe,
    replay_options,
    run_qualification,
    run_training_loop,
    save_qualification_rollout,
    validate_recipe,
    write_owned_failure,
)
from archlab.rl.nemotron_data import problem_key


def config():
    result = {
        "schema_version": 1, "family": "full", "world_size": 16,
        "assets": "assets", "weights": "weights", "container_kernel_packages": "packages",
        "data_manifest": "data.json", "data_manifest_sha256": "1" * 64,
        "component_qualification": "component.json", "component_qualification_sha256": "2" * 64,
        "prompt_template": "package:prompts/deepseek_v41_math_rl_v1.yaml",
        "parents": {name: {"checkpoint": f"/{name}/step-004537", "marker_sha256": "3" * 64}
                    for name in ("normal", "simplicial")},
        "seed": 314, "group_size": 4, "prompts_per_rank": 1,
        "learning_rate": 1e-6, "max_new_tokens": 3, "context_limit": 16,
        "max_rollout_updates": 3, "pilot_updates": 2, "eval_count": 2,
        "qualification_eval_count": 1,
    }
    validate_recipe(result)
    return result


def frozen_rows(size=48, offset=0):
    result = []
    for i in range(offset, offset + size):
        problem = f"Compute {i} + 1."
        key = problem_key(problem)
        result.append({"id": key, "problem_sha256": key, "uuid": f"u{i}",
                       "prompt": [{"role": "user", "content": problem}],
                       "expected_answer": "42", "canonical_answer": "42"})
    return result


def sealed_data(tmp_path):
    options = config()
    files = {}
    for split, rows in (("train", frozen_rows()), ("heldout", frozen_rows(4, 100))):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        files[split] = {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}
    exclusion = tmp_path / "excluded.json"
    exclusion.write_text(json.dumps({"problem_sha256": [], "uuid": []}))
    manifest = {"training_authorized": True, "solution_columns_read": [], "files": files,
                "format": "test", "exclusion_index": {"path": str(exclusion), "sha256": sha256_file(exclusion)}}
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps(manifest))
    options.update(data_manifest=str(path), data_manifest_sha256=sha256_file(path))
    return options, manifest


def test_data_seals_authorization_disjointness_and_exposure(tmp_path):
    options, manifest = sealed_data(tmp_path)
    splits, _ = load_data(options)
    assert len(splits["train"]) == 48 and len(splits["heldout"]) == 4
    # An overlap cannot be hidden by resealing just the JSONL file and manifest.
    heldout = Path(manifest["files"]["heldout"]["path"])
    heldout.write_text("".join(json.dumps(row) + "\n" for row in splits["train"][:4]))
    manifest["files"]["heldout"]["sha256"] = sha256_file(heldout)
    Path(options["data_manifest"]).write_text(json.dumps(manifest))
    options["data_manifest_sha256"] = sha256_file(options["data_manifest"])
    with pytest.raises(ValueError, match="overlap"):
        load_data(options)
    manifest["training_authorized"] = False
    Path(options["data_manifest"]).write_text(json.dumps(manifest))
    options["data_manifest_sha256"] = sha256_file(options["data_manifest"])
    with pytest.raises(ValueError, match="approved"):
        load_data(options)


def test_changed_data_and_previously_exposed_ids_fail_closed(tmp_path):
    options, manifest = sealed_data(tmp_path)
    exclusion = Path(manifest["exclusion_index"]["path"])
    exclusion.write_text(json.dumps({"problem_sha256": [frozen_rows(1)[0]["id"]], "uuid": []}))
    manifest["exclusion_index"]["sha256"] = sha256_file(exclusion)
    Path(options["data_manifest"]).write_text(json.dumps(manifest))
    options["data_manifest_sha256"] = sha256_file(options["data_manifest"])
    with pytest.raises(ValueError, match="exposed"):
        load_data(options)
    with Path(manifest["files"]["train"]["path"]).open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="Changed RL"):
        load_data(options)


def test_portable_recipe_and_scope_limits(tmp_path, monkeypatch):
    options = config()
    options["assets"] = "env:RL_TEST_ASSETS"
    monkeypatch.setenv("RL_TEST_ASSETS", "/sealed/assets")
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(options))
    loaded = read_recipe(path)
    assert loaded["assets"] == "/sealed/assets"
    assert loaded["prompt_template"].endswith("/archlab/prompts/deepseek_v41_math_rl_v1.yaml")
    for change in ({"temperature": .7}, {"max_rollout_updates": 129}, {"group_size": 1}, {"world_size": 8}):
        with pytest.raises(ValueError):
            validate_recipe({**options, **change})


def test_tokenization_uses_only_versioned_user_problem():
    observed = []

    class Encoder:
        def encode_messages(self, messages, **kwargs):
            observed.append((messages, kwargs))
            return messages[0]["content"]

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert "SECRET_REFERENCE" not in text and not add_special_tokens
            return [0, 4]

    rows = frozen_rows(1)
    rows[0]["expected_answer"] = "SECRET_REFERENCE"
    result = encode_examples(rows, tokenizer=Tokenizer(), renderer=SimpleNamespace(encoder=Encoder()),
                             user_template="Solve: {problem}", context_limit=8, max_new_tokens=3)
    assert result[0]["prompt_ids"] == [0, 4]
    assert observed == [([{"role": "user", "content": "Solve: Compute 0 + 1."}], {"thinking_mode": "chat"})]


def test_length_buckets_preserve_ids_and_rank_pairing_with_private_rng():
    rows = [{"problem_id": f"p{i:03}", "prompt_ids": [0] * (1 + i % 15)} for i in range(67)]
    random.seed(27)
    state = random.getstate()
    first, receipt = length_bucket_order(rows, global_batch_size=16, seed=100)
    second, other = length_bucket_order(list(reversed(rows)), global_batch_size=16, seed=100)
    assert first == second and receipt == other
    assert random.getstate() == state
    assert len({r["problem_id"] for r in first}) == 67
    assert receipt["partial_tail_retained_last"] == 3
    for offset in range(0, 64, 16):
        lengths = [len(r["prompt_ids"]) for r in first[offset:offset + 16]]
        assert max(lengths) - min(lengths) <= 4
    gathered = []
    for rank in range(16):
        groups, prompts = rank_groups(first, 16, rank=rank, world=16, prompts_per_rank=1, group_size=4)
        assert len(prompts) == 4 and all(p == groups[0]["prompt_ids"] for p in prompts)
        gathered.append(groups[0]["problem_id"])
    assert gathered == [r["problem_id"] for r in first[16:32]]
    with pytest.raises(ValueError, match="exhaust"):
        rank_groups(first, 64, rank=0, world=16, prompts_per_rank=1, group_size=4)


def test_qualification_is_contract_bound_and_never_synthetic_training():
    contract = {"a": 1}
    receipt = {"passed": True, "kind": "online-policy-numerical-v1",
               "contract_digest": digest(contract), "synthetic_optimizer_updates": 0}
    admit_qualification(receipt, contract)
    for change in ({"passed": False}, {"contract_digest": "wrong"}, {"synthetic_optimizer_updates": 1}):
        with pytest.raises(ValueError):
            admit_qualification({**receipt, **change}, contract)


class Tokenizer:
    def decode(self, ids, *, skip_special_tokens):
        assert not skip_special_tokens
        return "".join({4: "42", 6: "41"}[i] for i in ids)


def run_fake_loop(tmp_path, *, mixed=True, mode="train", stop=False):
    options = config()
    model = SimpleNamespace(lm_head=torch.nn.Linear(3, 8))
    train = [{"problem_id": f"p{i}", "prompt_ids": [0, i % 7], "expected_answer": "42"} for i in range(48)]
    encoded = {"train": train, "heldout": train[:2]}
    calls, saved, evaluations = [], [], []

    def sample(_model, prompts, **kwargs):
        calls.append(copy.deepcopy((prompts, kwargs)))
        tokens = [4, 6, 4, 6] if mixed else [6] * 4
        return SimpleNamespace(generated_ids=[[t, 1] for t in tokens], finish_reasons=["stop"] * 4,
                               receipt={**kwargs, "eos_token_ids": sorted(kwargs["eos_token_ids"])})

    def update(_model, _optimizer, _indexers, _rollout, rewards, **kwargs):
        assert rewards.tolist() == ([[1., 0., 1., 0.]] if mixed else [[0.] * 4])
        assert kwargs["lr"] == 1e-6
        assert kwargs["replay_mode"] == "sampled-prefix" and kwargs["replay_prefixes"] == 4
        assert kwargs["replay_seed"] == calls[-1][1]["seed"]
        return {"updated": mixed, "update_skipped": not mixed, "reward_mean": float(rewards.mean()),
                "policy_loss": .1 if mixed else 0., "reference_kl_available": False,
                "gradient_norm_before_clip": .2 if mixed else 0., "learning_rate": 1e-6}

    def evaluate(*_args, **kwargs):
        evaluations.append(kwargs["policy_version"])
        return {"pass_at_1": .5, "valid_answer_rate": 1., "truncation_rate": 0.,
                "count": 2, "policy_version": kwargs["policy_version"]}, []

    def checkpoint(path, _model, _optimizer, cursor, _contract):
        saved.append((path, copy.deepcopy(cursor)))

    report = run_training_loop(model, None, [], encoded, Tokenizer(), options, {"test": 1}, tmp_path,
        mode=mode, step_limit=2 if mode == "pilot" else 3, cursor=initial_cursor(), stops={1}, pad=2,
        stop_requested=lambda: stop, sample_fn=sample, update_fn=update,
        evaluate_fn=evaluate, checkpoint_fn=checkpoint)
    return report, calls, saved, evaluations


def test_true_reward_loop_advances_fresh_policy_and_exact_cursor(tmp_path):
    report, calls, saved, evaluations = run_fake_loop(tmp_path)
    assert report["rollout_step"] == report["optimizer_step"] == report["prompt_cursor"] == 3
    assert report["generated_tokens"] == 24
    assert report["stop_reason"] == "budget_reached"
    assert [args[0][0] for args in calls] == [[0, 0], [0, 1], [0, 2]]
    assert [args[1]["seed"] for args in calls] == [314, 315, 316]
    assert len({args[1]["policy_version"] for args in calls}) == 3
    assert len(saved) >= 2 and len(evaluations) == 3
    assert (tmp_path / "REAL_UPDATE_VERIFIED.json").is_file()
    assert (tmp_path / "COMPLETE.json").is_file()


def test_no_real_reward_variation_stops_after_pilot_without_false_completion(tmp_path):
    report, calls, _, _ = run_fake_loop(tmp_path, mixed=False)
    assert report["rollout_step"] == 2 and report["optimizer_step"] == 0
    assert report["stop_reason"] == "pilot_no_policy_update"
    assert calls[0][1]["policy_version"] == calls[1][1]["policy_version"]
    assert not (tmp_path / "COMPLETE.json").exists()
    assert not (tmp_path / "REAL_UPDATE_VERIFIED.json").exists()


def test_stop_request_checkpoints_without_sampling_or_update(tmp_path):
    report, calls, saved, _ = run_fake_loop(tmp_path, stop=True)
    assert report["stop_reason"] == "stop_request"
    assert not calls and saved[0][1]["optimizer_step"] == 0
    assert not (tmp_path / "COMPLETE.json").exists()


def test_pilot_is_not_full_budget_completion(tmp_path):
    report, _, _, _ = run_fake_loop(tmp_path, mode="pilot")
    assert report["optimizer_step"] == 2 and report["stop_reason"] == "step_limit"
    assert not (tmp_path / "COMPLETE.json").exists()


def test_actual_mlflow_metric_and_provenance_interfaces(tmp_path, monkeypatch):
    from archlab.automodel import deepseek_v41_rl_training as module
    from archlab.tracking.rl_mlflow import _metadata, _step, evaluation_values, train_values

    run_fake_loop(tmp_path)
    train = [json.loads(line) for line in (tmp_path / "rl-metrics.jsonl").read_text().splitlines()]
    evaluation = [json.loads(line) for line in (tmp_path / "evaluation.jsonl").read_text().splitlines()]
    assert [_step(row) for row in train] == [1, 2, 3]
    assert [_step(row) for row in evaluation] == [0, 2, 3]
    for row in train:
        metrics = train_values(row)
        assert "rl/reward_mean" in metrics and "perf/rollout_tokens_per_second" in metrics
        assert "rl/kl" not in metrics
    for row in evaluation:
        assert evaluation_values(row)["eval/pass_at_1"] == .5
    monkeypatch.setattr(module, "source_identity", lambda: {"unit_test": "sha"})
    contract = make_contract(config(), "normal", {"packages": {}}, {"cursor": {"step": 4537}}, {})
    metadata = _metadata({"variant": "normal"}, contract)
    for field in ("source_commit", "parent_checkpoint_path", "parent_checkpoint_sha256",
                  "data_manifest_path", "data_manifest_sha256", "algorithm", "group_size", "reward_backend"):
        assert metadata[field]
    assert metadata["algorithm"] == "RLOO"
    assert contract["gradient_estimator"]["replay_mode"] == "sampled-prefix"
    assert contract["gradient_estimator"]["scale"] == "global_forward_count/selected_prefix_count"
    assert contract["numerical_precision"]["cuda_matmul_allow_bf16_reduced_precision_reduction"] is False


def test_prefix_replay_settings_and_cross_actor_seed_agree():
    options = config()
    assert replay_options(options, rollout_step=3, world=16) == {
        "replay_mode": "sampled-prefix", "replay_prefixes": 4, "replay_seed": 314 + 48}
    assert replay_options(options, rollout_step=3, world=16, qualification=True)["replay_seed"] == 1000314
    for change in ({"replay_mode": "packed"}, {"replay_prefixes": 0},
                   {"numerical_precision": {"cuda_matmul_allow_tf32": True}}):
        with pytest.raises(ValueError):
            validate_recipe({**options, **change})


def test_numerical_precision_is_explicit_and_reapplied():
    previous = (torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        first = configure_numerical_precision()
        assert first == {"cuda_matmul_allow_tf32": False,
                         "cuda_matmul_allow_bf16_reduced_precision_reduction": False}
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        assert configure_numerical_precision() == first
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous[0]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous[1]


def diagnostic_rollout(version="test-v0"):
    from archlab.rl.rollout import RolloutBatch

    ids = torch.tensor([[0, 3, 4, 1]])
    labels = torch.tensor([[-100, 4, 1, -100]])
    return RolloutBatch(ids, torch.ones_like(ids, dtype=torch.bool), labels, labels != -100,
                        torch.tensor([[0., -.2, -.3, 0.]]), torch.tensor([[0., -.2, -.3, 0.]]),
                        [[4, 1]], [2], ["stop"], {"policy_version": version, "forward_shapes": [[1, 4], [1, 4]]})


def test_qualification_snapshot_keeps_exact_ids_masks_scores_without_model_state(tmp_path):
    rollout = diagnostic_rollout()
    receipt = save_qualification_rollout(rollout, tmp_path, rank=0,
                                         replay=replay_options(config(), rollout_step=0, world=16))
    assert sha256_file(receipt["path"]) == receipt["sha256"]
    saved = torch.load(receipt["path"], weights_only=True)
    assert torch.equal(saved["labels"], rollout.labels)
    assert torch.equal(saved["response_mask"], rollout.response_mask)
    assert torch.equal(saved["policy_log_probs"], rollout.policy_log_probs)
    assert saved["generated_ids"] == [[4, 1]] and saved["prompt_lengths"] == [2]
    assert saved["optimizer_updates_requested"] is False
    assert not {"model", "weights", "optimizer", "state_dict"} & saved.keys()
    with pytest.raises(FileExistsError):
        save_qualification_rollout(rollout, tmp_path, rank=0, replay={})


def test_failed_qualification_saves_replay_evidence_before_audit(tmp_path, monkeypatch):
    from archlab.automodel import deepseek_v41_rl_training as training
    from archlab.automodel import deepseek_v41_rl_update as update
    from archlab.rl import rollout as sampling

    restored = []
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 16)
    monkeypatch.setattr(training, "_snapshot_rng", lambda: "rng-before")
    monkeypatch.setattr(training, "_restore_rng", lambda value: restored.append(value))
    monkeypatch.setattr(training, "qualify_leaf_head", lambda *_: {"passed": True})
    monkeypatch.setattr(sampling, "sample_rollouts", lambda _model, _prompts, **kwargs:
                        diagnostic_rollout(kwargs["policy_version"]))

    def fail(_model, _optimizer, _indexers, _rollout, _rewards, **kwargs):
        assert (tmp_path / "rank-00-qualification-rollout.pt").is_file()
        assert kwargs["audit_only"] is True and kwargs["replay_mode"] == "sampled-prefix"
        assert kwargs["replay_prefixes"] == 4 and kwargs["replay_seed"] == 1000314
        raise ValueError("deliberate replay mismatch")

    monkeypatch.setattr(update, "policy_gradient_step", fail)
    optimizer = SimpleNamespace(state={}, zero_grad=lambda **_: None)
    model = SimpleNamespace(lm_head=torch.nn.Linear(3, 8))
    encoded = {"train": [{"problem_id": f"p{i}", "prompt_ids": [0, 3], "expected_answer": "42"}
                         for i in range(48)]}
    with pytest.raises(ValueError, match="deliberate replay mismatch"):
        run_qualification(model, optimizer, [], encoded, Tokenizer(), config(), {"test": 1}, tmp_path, {1}, 2)
    failure = json.loads((tmp_path / "rank-00-qualification-replay-failure.json").read_text())
    assert failure["reason"] == "deliberate replay mismatch"
    assert failure["optimizer_state_nonempty"] is False
    assert restored == ["rng-before"]
    assert not (tmp_path / "QUALIFIED.json").exists()


def test_component_gate_checks_rank_coverage_and_runtime(tmp_path):
    from archlab.automodel import deepseek_v41_rl_training as module

    path = tmp_path / "component.json"
    root = Path(module.__file__).resolve().parents[1]
    packet = {"passed": True, "world_size": 16, "container_image": "image", "cuda": "13.2",
              "weight_residency_qualified": True, "sampled_prefix_policy_gradient_qualified": True,
              "nccl": [2, 29, 7], "torch": "2.12.nv26.4.123",
              "implementation_sha256": {name: sha256_file(root / name) for name in (
                  "automodel/deepseek_v41_rl_head.py", "rl/rollout.py", "automodel/deepseek_v41_rl_fsdp_probe.py",
                  "rl/weight_residency.py", "automodel/deepseek_v41_rl_update.py")},
              "ranks": [{"rank": i, "passed": True, "signed_coefficients": True, "masked_targets": True,
                         "rollout_replay_max_abs_error": 0.} for i in range(16)]}
    path.write_text(json.dumps(packet))
    options = {"component_qualification": str(path), "component_qualification_sha256": sha256_file(path)}
    loading = {"container_image": "image", "cuda": "13.2", "nccl": [2, 29, 7],
               "packages": {"torch": "2.12.nv26.04.123"}}
    assert component_gate(options, loading)["passed"]
    with pytest.raises(ValueError, match="runtime"):
        component_gate(options, {**loading, "cuda": "12.0"})
    changed = copy.deepcopy(packet)
    changed["implementation_sha256"]["rl/rollout.py"] = "0" * 64
    path.write_text(json.dumps(changed))
    options["component_qualification_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="implementation hashes"):
        component_gate(options, loading)
    packet["ranks"].pop()
    path.write_text(json.dumps(packet))
    options["component_qualification_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="sixteen"):
        component_gate(options, loading)


def test_rejected_preexisting_output_is_not_modified(tmp_path):
    sentinel = tmp_path / "rank-00-failure.json"
    sentinel.write_text("previous evidence")
    write_owned_failure(tmp_path, 0, False, "new failure must not overwrite")
    assert sentinel.read_text() == "previous evidence"
    assert list(tmp_path.iterdir()) == [sentinel]
    write_owned_failure(tmp_path, 1, True, "failure in newly owned output")
    assert json.loads((tmp_path / "rank-01-failure.json").read_text())["traceback"].startswith("failure in newly")


def test_resume_rejects_mixed_rank_metadata_before_payload_load(tmp_path):
    contract = {"format": "rl", "variant": "normal"}
    sidecars = []
    for rank in range(2):
        path = tmp_path / f"rl-rank-{rank:02d}-rng.json"
        path.write_text("{}")
        sidecars.append({"file": path.name, "sha256": sha256_file(path)})
    cursor = {"rollout_step": 2, "rl_rng_sidecars": sidecars}
    marker = {"format": "archlab-v41-full-sharded-v1", "world_size": 2,
              "contract": contract, "cursor": cursor,
              "manifests": [f"rank-{rank:02d}/MANIFEST.json" for rank in range(2)]}
    (tmp_path / "COMPLETE.json").write_text(json.dumps(marker))
    folder = tmp_path / "rank-00"
    folder.mkdir()
    manifest = {"rank": 0, "world_size": 2, "contract": contract, "cursor": cursor}
    (folder / "MANIFEST.json").write_text(json.dumps(manifest))
    assert preflight_rl_checkpoint(tmp_path, contract, rank=0, world=2)["cursor"] == cursor
    for change in ({"rank": 1}, {"world_size": 3}, {"cursor": {"rollout_step": 99}},
                   {"contract": {**contract, "variant": "simplicial"}}):
        (folder / "MANIFEST.json").write_text(json.dumps({**manifest, **change}))
        with pytest.raises(ValueError, match="rank manifest"):
            preflight_rl_checkpoint(tmp_path, contract, rank=0, world=2)
