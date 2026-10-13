"""Real package tools/DefaultAgent/runner must not grade failed transfers."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from archlab.training.sandbox import E2BEnvironment


@pytest.mark.parametrize("tool", ["edit", "write"])
@pytest.mark.parametrize("transfer_failure", [True, False], ids=["transport", "ordinary-tool-error"])
def test_actual_tool_agent_runner_preserves_infrastructure_failure(monkeypatch, tool, transfer_failure):
    checkout = Path(__file__).resolve().parents[1] / "third_party/verl"
    monkeypatch.syspath_prepend(str(checkout))
    monkeypatch.syspath_prepend(str(checkout / "third_party/mimoagent-osr/src"))
    runner = importlib.import_module("recipes.code.mimoagent_runner")
    events, feedback = [], []

    monkeypatch.setitem(sys.modules, "e2b.sandbox.commands.command_handle", SimpleNamespace(CommandExitException=type(
        "CommandExitException", (Exception,), {})))

    def fail_upload(*args, **kwargs):
        raise RuntimeError("deliberately-fake-private-sdk-value")

    env = E2BEnvironment(image="fake-image")
    env.sandbox = SimpleNamespace(
        files=SimpleNamespace(make_dir=fail_upload, write=fail_upload),
        commands=SimpleNamespace(run=lambda *a, **kw: SimpleNamespace(stdout="OK\n", stderr="", exit_code=0)),
    )
    params = {"path": "/tmp/example.py" if transfer_failure else "relative.py"}
    params.update({"old_str": "old", "new_str": "new"} if tool == "edit" else {"file_text": "new"})

    class Model:
        calls = 0

        def get_template_vars(self):
            return {}

        def query(self, messages, **kwargs):
            self.calls += 1
            feedback.extend(message for message in messages if message["role"] == "tool")
            if self.calls == 1:
                return {"content": "Exercise the requested tool.", "tool_calls": [{
                    "id": "call", "type": "function", "function": {"name": tool, "arguments": json.dumps(params)},
                }]}
            return {"content": "Finished."}

    def calculate_reward():
        events.append("verifier")
        return 0.0, "ordinary test failure", {"verifier_returncode": 1, "resolved": False}

    environment = SimpleNamespace(env=env, setup_environment=lambda: events.append("setup"),
                                  calculate_reward=calculate_reward, cleanup=lambda: events.append("cleanup"))
    monkeypatch.setitem(sys.modules, "mimoagent.environments.utils", SimpleNamespace(
        make_dataset_env=lambda *a, **kw: environment))
    monkeypatch.setattr(runner, "_prepare_swebench_import_path", lambda: None)
    monkeypatch.setattr(runner, "_build_model", lambda *a, **kw: Model())
    monkeypatch.delenv("UNI_AGENT_LOG_DIR", raising=False)

    def run():
        return runner._run_sync(
            raw_prompt="CPU-only tool-boundary fixture.",
            instance={"dataset_type": "opensource-code", "instance_id": "fixture"},
            session=SimpleNamespace(base_url="", reward_info_url=None),
            config={"agent": {"type": "default", "tools": [{"tool": tool}], "step_limit": 2}},
            agent_overrides={}, environment_overrides={},
        )

    if transfer_failure:
        with pytest.raises(RuntimeError, match="status=InfraError: E2B file transfer infrastructure failure: RuntimeError") as raised:
            run()
        assert "deliberately-fake-private-sdk-value" not in str(raised.value)
        assert events == ["setup", "cleanup"]
        assert feedback == []
    else:
        result = run()
        assert result["agent_status"] == "Idle" and result["finished"] is True
        assert result["reward"] == 0.0 and result["verifier_returncode"] == 1
        assert events == ["setup", "verifier", "cleanup"]
        assert len(feedback) == 1 and "Path must be absolute" in feedback[0]["content"]
