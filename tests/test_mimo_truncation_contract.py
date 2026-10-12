"""Native-context exhaustion must not become a successful agent completion."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def native_mimo(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "third_party/verl/third_party/mimoagent-osr/src"
    if not (source / "mimoagent/__init__.py").is_file():
        pytest.skip("requires the initialized pinned MiMo submodule")
    pytest.importorskip("openai")
    pytest.importorskip("rich")
    monkeypatch.syspath_prepend(str(source))
    from mimoagent.agents.bashonly.bashonly_agent import BashOnlyAgent
    from mimoagent.models.openai_chat import OpenAIChatModel

    return BashOnlyAgent, OpenAIChatModel


@pytest.mark.parametrize("finish_reason,expected", [("length", "LimitsExceeded"), ("stop", "Idle")])
@pytest.mark.parametrize("tool_calls", [None, []])
def test_nonempty_length_stop_preserves_response_and_is_not_idle(finish_reason, expected, tool_calls, native_mimo):
    BashOnlyAgent, OpenAIChatModel = native_mimo

    model = OpenAIChatModel(model_name="fixture", model_kwargs={
        "base_url": "http://unused.invalid/v1", "api_key": "unused-test-value"})
    response = SimpleNamespace(usage=None, choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content="partial candidate", tool_calls=tool_calls))])
    model.client.chat.completions.create = Mock(return_value=response)
    environment = SimpleNamespace(config=SimpleNamespace(cwd="/testbed"), get_template_vars=lambda: {})
    agent = BashOnlyAgent(model, environment, system_template="test", instance_template="{{task}}")
    status, _ = agent.run("repair the function")
    assert status == expected
    assistant = [message for message in agent.messages if message["role"] == "assistant"]
    assert assistant[-1]["content"] == "partial candidate"
    assert "finish_reason" not in assistant[-1]
    assert model.last_finish_reason == finish_reason


def test_length_stop_does_not_execute_a_truncated_tool_call(native_mimo):
    BashOnlyAgent, OpenAIChatModel = native_mimo

    model = OpenAIChatModel(model_name="fixture", model_kwargs={
        "base_url": "http://unused.invalid/v1", "api_key": "unused-test-value"})
    response = SimpleNamespace(usage=None, choices=[SimpleNamespace(
        finish_reason="length", message=SimpleNamespace(content="", tool_calls=[dict(
            id="call", type="function", function=dict(name="bash", arguments='{"command":"pwd"}'))]))])
    model.client.chat.completions.create = Mock(return_value=response)
    environment = SimpleNamespace(config=SimpleNamespace(cwd="/testbed"), get_template_vars=lambda: {})
    agent = BashOnlyAgent(model, environment, system_template="test", instance_template="{{task}}")
    agent.execute_action = Mock(side_effect=AssertionError("truncated tool call must not execute"))
    status, _ = agent.run("repair the function")
    assert status == "LimitsExceeded"
    agent.execute_action.assert_not_called()
