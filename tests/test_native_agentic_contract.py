"""Native tokenizer + upstream agent/gateway integration; run in the GPU runtime.

Set ARCHLAB_NATIVE_TOKENIZER to a local Limite Base tokenizer snapshot. This
never executes a command or loads model weights: only recorded token outputs.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

MODEL = os.environ.get("ARCHLAB_NATIVE_TOKENIZER")
pytestmark = pytest.mark.skipif(not MODEL, reason="needs qualified native tokenizer and upstream runtime")


def test_native_tools_context_and_training_mask(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    for directory in ("third_party/verl", "third_party/verl/third_party/uni_agent",
                      "third_party/verl/third_party/mimoagent-osr/src"):
        monkeypatch.syspath_prepend(str(root / directory))
    from mimoagent.agents.bashonly.bashonly_agent import BashOnlyAgentConfig
    from mimoagent.tools.registry import ToolRegistry
    from transformers import AutoTokenizer
    from uni_agent.gateway.session import GatewaySession, MessageCodec, SessionHandle

    model = Path(MODEL)
    native_files = [model / name for name in (
        "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
    )]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in native_files]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    tokenizer_eos = tokenizer.eos_token_id
    model_eos = json.loads((model / "generation_config.json").read_text())["eos_token_id"]
    assert isinstance(model_eos, int)
    tools = ToolRegistry.from_config(BashOnlyAgentConfig().tools).get_function_definitions()
    codec = MessageCodec(tokenizer, tool_parser_name="hermes", rollout_backend="verl",
                         generation_eos_token_id=model_eos)
    session = GatewaySession(SessionHandle("contract"), codec,
                             prompt_length=16384, response_length=114688)
    messages = [{"role": "system", "content": "You are an agent working in /testbed."},
                {"role": "user", "content": "Inspect the repository directory."}]
    command = json.dumps({"name": "bash", "arguments": {"command": "pwd"}})
    generated = [
        tokenizer.encode("<tool_call>\n" + command + "\n</tool_call>", add_special_tokens=False) + [model_eos],
        tokenizer.encode("Directory inspected.", add_special_tokens=False) + [model_eos],
    ]
    requests = []

    class Backend:
        async def generate(self, **kwargs):
            requests.append(kwargs)
            ids = generated[len(requests) - 1]
            return SimpleNamespace(token_ids=ids,
                                   extra_fields=dict(min_global_steps=len(requests), max_global_steps=len(requests)),
                                   log_probs=[-0.1] * len(ids),
                                   stop_reason="stop")

    async def run():
        backend = Backend()
        first = await session.run_generation(
            {"messages": messages, "tools": tools, "sampling_params": {"logprobs": True, "stop_token_ids": None}}, backend)
        call = first.assistant_msg["tool_calls"][0]
        assert first.finish_reason == "tool_calls"
        assert first.assistant_msg["content"] == ""
        assert call["function"]["name"] == "bash"
        assert json.loads(call["function"]["arguments"]) == {"command": "pwd"}
        continued = messages + [first.assistant_msg,
                                {"role": "tool", "tool_call_id": call["id"], "content": "/testbed"}]
        expected = codec.encode_full(continued, tools)
        await session.run_generation(
            {"messages": continued, "tools": tools, "sampling_params": {"logprobs": True, "stop_token_ids": None}}, backend)
        assert requests[1]["prompt_ids"] == expected
        trajectories = await session.finalize()
        assert len(trajectories) == 1
        return trajectories[0]

    trajectory = asyncio.run(run())
    from archlab.training.rollout import validate_trajectory

    trajectory.reward_info["termination_kind"] = "completed"
    metadata = validate_trajectory(trajectory, question_id="task", group_id="group", response_id="group_0_0",
                                   session_id=0, context=131072, eos_ids={model_eos}, rng_identity="recorded-fixture")
    assert (metadata.policy_version_first, metadata.policy_version_last) == (1, 2)
    assert metadata.native_eos_tokens == 2 and metadata.observation_tokens > 0
    assert sum(trajectory.response_mask) == sum(map(len, generated))
    assert len(trajectory.response_ids) == len(trajectory.response_mask) == len(trajectory.response_logprobs)
    assert 0 in trajectory.response_mask
    assert all(p == 0 for p, m in zip(trajectory.response_logprobs, trajectory.response_mask, strict=True) if m == 0)
    for request in requests:
        assert request["sampling_params"]["max_tokens"] == 131072 - len(request["prompt_ids"])
        assert request["sampling_params"]["stop_token_ids"] == [model_eos]
        assert request["sampling_params"]["ignore_eos"] is True
    assert requests[0]["sampling_params"]["max_tokens"] > 114688
    assert tokenizer.eos_token_id == tokenizer_eos
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in native_files]


@pytest.mark.parametrize("stop_reason,has_eos,expected_finish", [
    ("length", False, "length"),
    ("max_tokens", False, "length"),
    ("stop", True, "tool_calls"),
    ("eos", True, "tool_calls"),
])
def test_native_tool_calls_preserve_truncation_through_gateway_and_agent(
    monkeypatch, stop_reason, has_eos, expected_finish,
):
    from unittest.mock import Mock

    root = Path(__file__).resolve().parents[1]
    for directory in ("third_party/verl", "third_party/verl/third_party/uni_agent",
                      "third_party/verl/third_party/mimoagent-osr/src"):
        monkeypatch.syspath_prepend(str(root / directory))
    from mimoagent.agents.bashonly.bashonly_agent import BashOnlyAgent, BashOnlyAgentConfig
    from mimoagent.models.openai_chat import OpenAIChatModel
    from mimoagent.tools.registry import ToolRegistry
    from openai.types.chat import ChatCompletion
    from transformers import AutoTokenizer
    from uni_agent.gateway.adapters.openai import openai_build_response
    from uni_agent.gateway.session import GatewaySession, MessageCodec, SessionHandle

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    eos = json.loads((Path(MODEL) / "generation_config.json").read_text())["eos_token_id"]
    tools = ToolRegistry.from_config(BashOnlyAgentConfig().tools).get_function_definitions()
    codec = MessageCodec(tokenizer, tool_parser_name="hermes", rollout_backend="verl",
                         generation_eos_token_id=eos)
    session = GatewaySession(SessionHandle("truncation"), codec,
                             prompt_length=16384, response_length=114688)
    text = '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'
    if not has_eos:
        text += '\n<tool_call>{"name":"bash"'  # Earlier complete calls do not finish this turn.
    response_ids = tokenizer.encode(text, add_special_tokens=False) + ([eos] if has_eos else [])

    class Backend:
        async def generate(self, **kwargs):
            return SimpleNamespace(token_ids=response_ids, stop_reason=stop_reason,
                                   extra_fields={}, log_probs=None)

    async def run():
        outcome = await session.run_generation({
            "messages": [{"role": "user", "content": "Inspect the repository."}],
            "tools": tools, "sampling_params": {},
        }, Backend())
        trajectories = await session.finalize()
        assert trajectories[0].response_ids == response_ids
        assert all(trajectories[0].response_mask)
        return outcome

    outcome = asyncio.run(run())
    assert outcome.finish_reason == expected_finish
    assert len(outcome.assistant_msg["tool_calls"]) == 1
    assert outcome.assistant_msg["tool_calls"][0]["function"]["name"] == "bash"
    wire = openai_build_response(outcome, model="fixture")
    assert wire["choices"][0]["finish_reason"] == expected_finish
    model = OpenAIChatModel(model_name="fixture", model_kwargs={
        "base_url": "http://unused.invalid/v1", "api_key": "unused-test-value"})
    completed = SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason="stop",
        message=SimpleNamespace(content="Done.", tool_calls=[]))])
    model.client.chat.completions.create = Mock(side_effect=[ChatCompletion.model_validate(wire), completed])
    environment = SimpleNamespace(config=SimpleNamespace(cwd="/testbed"), get_template_vars=lambda: {})
    agent = BashOnlyAgent(model, environment, system_template="test", instance_template="{{task}}")
    agent.execute_action = Mock(return_value={"output": "/testbed", "metadata": {"returncode": 0}})
    status, _ = agent.run("repair the function")
    assert status == ("LimitsExceeded" if expected_finish == "length" else "Idle")
    assert agent.execute_action.call_count == (0 if expected_finish == "length" else 1)
    assert any(message.get("tool_calls") for message in agent.messages)
