"""Admission must follow executed checks, never an old success receipt."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("omegaconf")
pytest.importorskip("hydra")

from omegaconf import OmegaConf

from archlab.training import qualification
from archlab.training.config import Experiment


def test_check_preserves_pending_and_reports_real_failure():
    pending = qualification._check("gpu", lambda: dict(status="pending", reason="no GPU test"))
    assert pending["status"] == "pending"

    def fail():
        raise ValueError("wrong data manifest")

    failed = qualification._check("data", fail)
    assert failed["status"] == "failed"
    assert failed["reason"] == "ValueError: wrong data manifest"


def test_undeclared_upstream_edits_cannot_reuse_an_admission(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setattr(qualification, "_git", lambda *args: b"changed runtime code")
    with pytest.raises(ValueError, match="declared patch set"):
        qualification._verify_patch_set(tmp_path, {"working_diff_sha256": hashlib.sha256(b"").hexdigest()})


def test_declared_patch_file_must_match_its_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    (tmp_path / "fix.patch").write_text("changed")
    with pytest.raises(ValueError, match="differs from provenance"):
        qualification._verify_patch_set(tmp_path, {"patches": [{"path": "fix.patch", "sha256": "wrong"}]})


def test_upstream_diff_fingerprint_is_independent_of_git_abbreviations(tmp_path):
    import hashlib
    import subprocess

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args])

    git("init", "-q")
    source = tmp_path / "model.py"
    source.write_text("value = 1\n")
    git("add", "model.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    source.write_text("value = 2\n")
    expected = hashlib.sha256(git("diff", "--full-index", "HEAD", "--")).hexdigest()
    abbreviated = []
    for length in (7, 12):
        git("config", "core.abbrev", str(length))
        abbreviated.append(git("diff", "HEAD", "--"))
        assert qualification._verify_patch_set(tmp_path, {"working_diff_sha256": expected}) == expected
    assert abbreviated[0] != abbreviated[1]
    source.write_text("value = 3\n")
    with pytest.raises(ValueError, match="declared patch set"):
        qualification._verify_patch_set(tmp_path, {"working_diff_sha256": expected})


@pytest.mark.parametrize("defect", [None, "rl_payload", "rl_file", "rl_metadata", "tokens", "full_weights"])
def test_fresh_rl_requires_actual_matched_sft_endpoint(tmp_path, defect):
    metadata = {"frozen_sha256": "a" * 64, "tokens": 2_000_158_720, "files": {"adapter.pt": "hash"}}
    if defect == "rl_payload":
        metadata["files"]["rl_state.pt"] = "hash"
    elif defect == "rl_file":
        (tmp_path / "rl_state.pt").touch()
    elif defect == "rl_metadata":
        metadata["math_protocol"] = {}
    elif defect == "tokens":
        metadata["tokens"] -= 1
    elif defect == "full_weights":
        metadata["trainable_mode"] = "full"
        metadata["files"]["backbone.pt"] = "hash"
    (tmp_path / "COMPLETE.json").write_text(json.dumps(metadata))
    experiment = Experiment(name="fresh", parent_checkpoint=str(tmp_path))
    if defect:
        with pytest.raises(ValueError, match="complete SFT endpoint"):
            qualification.check_sft_parent(experiment)
    else:
        assert qualification.check_sft_parent(experiment)["path"] == str(tmp_path)


def test_full_sft_requires_an_explicit_different_source_contract(tmp_path):
    metadata = {"trainable_mode": "full", "tokens": 10_000_000_000,
                "files": {"adapter.pt": "hash", "backbone.pt": "hash"}}
    (tmp_path / "COMPLETE.json").write_text(json.dumps(metadata))
    experiment = Experiment(name="other-trial", parent_checkpoint=str(tmp_path),
                            parent_stage="full_sft", parent_tokens=10_000_000_000)
    assert qualification.check_sft_parent(experiment)["path"] == str(tmp_path)


@pytest.mark.parametrize("change", ["edit", "add", "delete"])
def test_admission_detects_uncommitted_source_changes(tmp_path, monkeypatch, change):
    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    source = tmp_path / "src/archlab/training"
    source.mkdir(parents=True)
    module = source / "provider.py"
    module.write_text("version = 1\n")
    manifest = qualification.source_manifest()
    assert qualification.check_source_unchanged(manifest)["files"] == 1
    if change == "edit":
        module.write_text("version = 2\n")
    elif change == "add":
        (source / "new_provider.py").write_text("version = 1\n")
    else:
        module.unlink()
    with pytest.raises(RuntimeError, match="source changed"):
        qualification.check_source_unchanged(manifest)


@pytest.mark.parametrize("outcome", ["skipped", "failure", "error", "empty", "no_report"])
def test_pytest_requires_executed_passing_tests(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    (tmp_path / "test_real.py").write_text("def test_real(): pass\n")

    def run(command, **kwargs):
        report = command[-1].removeprefix("--junitxml=")
        if outcome == "no_report":
            return SimpleNamespace(returncode=0)
        xml = ("<testsuites/>" if outcome == "empty" else
               f'<testsuites><testsuite><testcase name="test_real"><{outcome}/></testcase></testsuite></testsuites>')
        from pathlib import Path

        Path(report).write_text(xml)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qualification.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="report|failed or skipped"):
        qualification.run_pytest(["test_real.py"], tmp_path / "output")


def test_missing_integration_is_pending_without_executing_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "ROOT", tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("missing tests must not run a process")

    monkeypatch.setattr(qualification.subprocess, "run", forbidden)
    result = qualification.run_pytest(["absent.py::test_gpu"], tmp_path / "output")
    assert result["status"] == "pending"


def test_serving_reload_requires_an_actual_optimizer_export(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    (tmp_path / "test_real.py").write_text("def test_real(): pass\n")
    with pytest.raises(ValueError, match="preceding package optimizer export"):
        qualification.run_pytest(["test_real.py"], tmp_path / "output",
                                  updated_model=tmp_path / "absent-export")


@pytest.mark.parametrize("contrast,grad_norm,expected", [(0, 0, "pending"), (1, 0, "pending"), (1, 0.2, "passed")])
def test_execution_alone_does_not_admit_a_zero_signal_pilot(tmp_path, contrast, grad_norm, expected):
    artifacts = tmp_path / "artifacts" / "actual-update"
    artifacts.mkdir(parents=True)
    (artifacts / "pipeline-evidence.json").write_text(json.dumps({
        "reward_contrast_groups": contrast, "grad_norm": grad_norm,
    }))
    result = qualification._check("signal", lambda: qualification.check_learning_signal(tmp_path))
    assert result["status"] == expected


def test_learning_signal_requires_current_finite_update_evidence(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        qualification.check_learning_signal(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "pipeline-evidence.json").write_text(json.dumps({
        "reward_contrast_groups": 1, "grad_norm": float("nan"),
    }))
    with pytest.raises(ValueError, match="finite gradient"):
        qualification.check_learning_signal(tmp_path)


def test_every_distributed_rank_must_execute_a_test(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    (tmp_path / "test_real.py").write_text("def test_real(): pass\n")
    output = tmp_path / "output"

    def run(command, **kwargs):
        assert "--nproc_per_node=2" in command
        (output / "pytest-rank-0.xml").write_text(
            '<testsuites><testsuite><testcase name="test_real"/></testsuite></testsuites>'
        )
        (output / "pytest-rank-1.xml").write_text("<testsuites/>")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qualification.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="empty rank report"):
        qualification.run_pytest(["test_real.py"], output, world_size=2)


def test_pytest_binds_actual_deployment_and_replaces_stale_xml(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    (tmp_path / "test_real.py").write_text("def test_real(): pass\n")
    output = tmp_path / "output"
    output.mkdir()
    (output / "pytest.xml").write_text("old report")
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    (deployment / "resolved.yaml").write_text("actor_rollout_ref:\n  model:\n    path: /actual-policy\n")
    (deployment / "experiment.json").write_text('{"name":"candidate"}')

    def run(command, **kwargs):
        assert not (output / "pytest.xml").exists()
        assert f"--basetemp={output / 'artifacts'}" in command
        assert kwargs["env"]["ARCHLAB_QUALIFICATION_CONFIG"] == str(output / "resolved.yaml")
        assert kwargs["env"]["ARCHLAB_QUALIFICATION_EXPERIMENT"] == str(output / "experiment.json")
        assert (output / "resolved.yaml").read_bytes() == (deployment / "resolved.yaml").read_bytes()
        (deployment / "resolved.yaml").write_text("changed: next-candidate\n")
        assert "/actual-policy" in (output / "resolved.yaml").read_text()
        (output / "pytest.xml").write_text('<testsuites><testsuite><testcase name="test_real"/></testsuite></testsuites>')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qualification.subprocess, "run", run)
    result = qualification.run_pytest(["test_real.py"], output, deployment=deployment)
    assert result["status"] == "passed" and result["tests"] == 1
    assert result["reports"]


def test_readiness_and_old_receipt_cannot_admit_unimplemented_gpu_pipeline(tmp_path, monkeypatch):
    from archlab.training import sandbox

    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    monkeypatch.setattr(sandbox, "check_sandbox", lambda *args: dict(verified=True))
    monkeypatch.setattr(qualification, "validate_science", lambda *args: None)
    for name in ("check_upstream", "check_package_config", "check_runtime", "check_data", "check_model", "check_execution"):
        monkeypatch.setattr(qualification, name, lambda *args: dict(inspected=True))
    original = qualification.run_pytest

    def fixture_or_real(nodes, output, **kwargs):
        if nodes == ["tests/test_training_limite.py"]:
            return dict(status="passed", tests=5)
        return original(nodes, output, **kwargs)

    monkeypatch.setattr(qualification, "run_pytest", fixture_or_real)
    output = tmp_path / "report"
    output.mkdir()
    (output / "qualification.json").write_text('{"passed":true,"production_admitted":true}')
    experiment = Experiment(name="test", parent_checkpoint="parent", data_manifest="manifest")
    result = qualification.qualify(experiment, OmegaConf.create({"name": "actual", "trainer": {"n_gpus_per_node": 8}}), tmp_path, output)
    assert not result["passed"]
    expected = list(qualification.INTEGRATION_TESTS)
    expected.insert(expected.index("learner_throughput") + 1, "throughput_regression")
    assert result["blockers"] == [*expected, "agentic_learning_signal"]
    pending = [row for row in result["checks"] if row["name"] in result["blockers"]]
    assert all(row["status"] == "pending" for row in pending)
    assert result["checks"][-1]["name"] == "source_unchanged"
    assert result["checks"][-1]["status"] == "passed"
    assert json.loads((output / "qualification.json").read_text()) == result


def test_unsupported_model_engine_does_not_become_supported_by_configuration():
    experiment = Experiment(name="test", engine="fsdp")
    with pytest.raises(ValueError, match="Megatron"):
        qualification.check_execution(experiment, OmegaConf.create({}))


def test_data_manifest_paths_cannot_be_bypassed_by_launch_override(tmp_path, monkeypatch):
    from archlab.data import mimo_code

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(mimo_code, "verify", lambda path: dict(verified=True), raising=False)
    config = OmegaConf.create(dict(data=dict(train_files=[str(tmp_path / "different.parquet")],
                                           val_files=[str(tmp_path / "validation.parquet")])))
    with pytest.raises(ValueError, match="train_files"):
        qualification.check_data(Experiment(name="test", data_manifest=str(manifest)), config)
