"""Regressions for the locally patched upstream MiMo SWE reward boundary."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def runner(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "third_party" / "verl"
    monkeypatch.syspath_prepend(str(root))
    return importlib.import_module("recipes.code.mimoagent_runner")


TASK = {"dataset_type": "opensource-code", "instance_id": "task"}


@pytest.mark.parametrize("code,reward", [(0, 1.0), (1, 0.0), (2, 0.0)])
def test_completed_pass_and_failed_tests_are_admissible(runner, code, reward):
    runner._require_code_verdict(TASK, reward, {"verifier_returncode": code, "resolved": code == 0})


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"model_patch": ""},
        {"verifier_returncode": None},
        {"verifier_returncode": -9},
        {"verifier_returncode": 124},
        {"verifier_returncode": 125},
        {"verifier_returncode": 126},
        {"verifier_returncode": 127},
        {"verifier_returncode": 137},
        {"verifier_returncode": True},
        {"verifier_returncode": 1, "resolved": False, "error_category": "reward/testbed_corrupted"},
        {"verifier_returncode": 1, "resolved": False, "transport_error": True},
        {"verifier_returncode": 1, "resolved": False, "model_patch_error": "cannot read git"},
    ],
)
def test_infrastructure_failures_cannot_become_zero_rewards(runner, extra):
    with pytest.raises(runner.RewardInfrastructureError):
        runner._require_code_verdict(TASK, 0.0, extra)


@pytest.mark.parametrize(
    "reward,extra",
    [
        (0.0, {"verifier_returncode": 0, "resolved": True}),
        (1.0, {"verifier_returncode": 1, "resolved": False}),
        (float("nan"), {"verifier_returncode": 0, "resolved": True}),
        (0.0, {"verifier_returncode": 1}),
    ],
)
def test_inconsistent_or_nonfinite_verdict_is_rejected(runner, reward, extra):
    with pytest.raises(runner.RewardInfrastructureError):
        runner._require_code_verdict(TASK, reward, extra)


def test_non_swe_reward_contract_is_unchanged(runner):
    runner._require_code_verdict({"dataset_type": "general_agent"}, 0.5, {})


def test_actual_runner_aborts_and_cleans_up_on_unverified_zero(runner, monkeypatch):
    import sys
    from types import SimpleNamespace

    cleaned = []
    environment = SimpleNamespace(
        env=object(),
        setup_environment=lambda: None,
        calculate_reward=lambda: (0.0, "sandbox timeout", {"model_patch": ""}),
        cleanup=lambda: cleaned.append(True),
    )

    class Agent:
        IDLE_STATUS = "Idle"

        def __init__(self, *args, **kwargs):
            pass

        def run(self, task):
            return self.IDLE_STATUS, "candidate patch"

    monkeypatch.setitem(
        sys.modules, "mimoagent.agents.factory", SimpleNamespace(get_agent_class=lambda _: Agent)
    )
    monkeypatch.setitem(
        sys.modules,
        "mimoagent.environments.utils",
        SimpleNamespace(make_dataset_env=lambda *a, **kw: environment),
    )
    monkeypatch.setattr(runner, "_build_model", lambda *a, **kw: object())
    monkeypatch.setattr(runner, "_prepare_swebench_import_path", lambda: None)
    with pytest.raises(runner.RewardInfrastructureError):
        runner._run_sync(
            raw_prompt="repair",
            instance=TASK,
            session=SimpleNamespace(base_url="http://unused.invalid"),
            config={},
            agent_overrides={},
            environment_overrides={},
        )
    assert cleaned == [True]


def test_single_harness_routes_every_sample_to_same_profile(runner, monkeypatch, tmp_path):
    profile = tmp_path / "bash.yaml"
    profile.write_text("agent: {type: bashonly-agent}\n")
    spec = tmp_path / "single.yaml"
    spec.write_text("harnesses:\n  - label: bash\n    config: bash.yaml\n")
    monkeypatch.setenv("MIXED_HARNESS_SPEC", str(spec))
    monkeypatch.setenv("MIXED_HARNESS_MODE", "step-hash")
    assert runner._mixed_harness_specs() == [("bash", str(profile))]
    for index in (0, 1, 17):
        assert runner._select_config_path(
            sample_index=index,
            tools_kwargs={"dataset_index": index, "harness_round": 3},
        ) == (str(profile), "bash")
    spec.write_text("harnesses: []\n")
    with pytest.raises(ValueError, match="at least one"):
        runner._mixed_harness_specs()
