"""Actual MiMo agent/tool/verifier admission; policy sampling is tested separately."""

import hashlib
import json
import os
from pathlib import Path

import pytest


def test_agent_tool_and_successful_verifier(monkeypatch, tmp_path):
    config_path = os.getenv("ARCHLAB_QUALIFICATION_CONFIG")
    if not config_path:
        pytest.skip("requires the exact deployment and an authorized task sandbox")
    from mimoagent.agents.bashonly.bashonly_agent import BashOnlyAgent
    from mimoagent.environments.utils import make_dataset_env
    from omegaconf import OmegaConf
    from recipes.code.mimoagent_runner import (
        _load_config,
        _mixed_harness_specs,
        _require_code_verdict,
    )

    config = OmegaConf.load(config_path)
    for key, value in config.ray_kwargs.ray_init.runtime_env.env_vars.items():
        monkeypatch.setenv(key, str(value))
    fixture_path = Path(os.environ["ARCHLAB_CODE_SMOKE_FIXTURE"])
    fixture = json.loads(fixture_path.read_text())
    payloads = {}
    for name in ("task", "reference_patch", "template_map"):
        path = Path(fixture[name]["path"])
        payloads[name] = path.read_bytes()
        assert hashlib.sha256(payloads[name]).hexdigest() == fixture[name]["sha256"]
    task = json.loads(payloads["task"])
    specs = _mixed_harness_specs()
    assert len(specs) == 1
    harness = _load_config(specs[0][1])
    assert harness["agent"]["type"] == "bashonly-agent"
    assert harness["environment"]["environment_class"] == "archlab.training.sandbox.E2BEnvironment"
    # The reference fixture can be outside the training split. Its explicit,
    # checksummed map is separate from production's all-task coverage gate.
    environment = dict(harness["environment"])
    environment["template_map"] = fixture["template_map"]["path"]
    env = make_dataset_env(task, **environment)
    evidence = {"fixture": task["instance_id"], "policy_sampling": False,
                "fixture_template_map_sha256": fixture["template_map"]["sha256"],
                "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest()}
    try:
        env.setup_environment()
        assert env._base_ref == fixture["runtime_base_commit"]
        assert not env.execute("git status --porcelain")["output"].strip()
        baseline, baseline_output, baseline_extra = env.calculate_reward()
        _require_code_verdict(task, baseline, baseline_extra)
        assert baseline == 0 and baseline_extra["verifier_returncode"] == 1
        assert fixture["baseline_assertion"] in baseline_output
        env.env.copy_to(fixture["reference_patch"]["path"], "/tmp/archlab-reference.patch")

        class RecordedAction:
            """Deterministic reference action, never a claim about model capability."""

            calls = 0

            def get_template_vars(self):
                return {}

            def query(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return {"content": "", "tool_calls": [{"id": "source-repair", "type": "function",
                            "function": {"name": "bash", "arguments": json.dumps({
                                "command": "git apply /tmp/archlab-reference.patch"})}}]}
                assert any(message["role"] == "tool" for message in messages)
                return {"content": "Applied the source repair."}

        agent_kwargs = {key: value for key, value in harness["agent"].items() if key != "type"}
        agent = BashOnlyAgent(RecordedAction(), env.env, **agent_kwargs)
        status, _ = agent.run(task["problem_statement"])
        assert status == agent.IDLE_STATUS
        assert not any(agent.tool_call_errors)
        # The verifier's hidden files are removed after baseline grading; they
        # never appear in the task prompt or the agent's tool observations.
        assert task["test_patch"] not in json.dumps(agent.messages)
        reward, output, extra = env.calculate_reward()
        _require_code_verdict(task, reward, extra)
        assert reward == 1 and extra["verifier_returncode"] == 0
        assert fixture["positive_assertion"] in output
        evidence.update(base_commit=env._base_ref, template_id=env.env.template_id,
                        baseline_returncode=1, repaired_returncode=0,
                        reference_patch_sha256=fixture["reference_patch"]["sha256"])
    finally:
        env.cleanup()
    (tmp_path / "agent-tool-verifier.json").write_text(json.dumps(evidence, indent=2) + "\n")
