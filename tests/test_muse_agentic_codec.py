"""Muse publisher tokenizer and stock vLLM parsing, without weights or a sandbox.

Set ARCHLAB_MUSE_TOKENIZER and use the qualified Muse package runtime. These
checks replay tokens; they do not establish model or executable task readiness.
"""

import asyncio
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

MODEL = os.environ.get("ARCHLAB_MUSE_TOKENIZER")
pytestmark = pytest.mark.skipif(not MODEL, reason="requires Muse tokenizer and native vLLM runtime")


@pytest.fixture
def native(monkeypatch, tokenizer):
    root = Path(__file__).resolve().parents[1]
    for directory in ("third_party/verl", "third_party/verl/third_party/uni_agent",
                      "third_party/verl/third_party/mimoagent-osr/src"):
        monkeypatch.syspath_prepend(str(root / directory))
    from mimoagent.agents.bashonly.bashonly_agent import BashOnlyAgentConfig
    from mimoagent.tools.registry import ToolRegistry
    from uni_agent.gateway.session import MessageCodec

    tools = ToolRegistry.from_config(BashOnlyAgentConfig().tools).get_function_definitions()
    eos = json.loads((Path(MODEL) / "generation_config.json").read_text())["eos_token_id"]
    codec = MessageCodec(tokenizer, tool_parser_name="muse_glimmer", rollout_backend="vllm",
                         generation_eos_token_id=eos,
                         apply_chat_template_kwargs={"current_date": "2026-10-13",
                                                     "reasoning_strength": "high"})
    return tokenizer, codec, tools


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)


def invoke(command="pwd", name="bash"):
    return (f'<atem:function_calls>\n<atem:invoke name="{name}">\n'
            f'<atem:parameter name="command">{command}</atem:parameter>\n'
            '</atem:invoke>\n</atem:function_calls>')


def ids(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def test_native_history_uses_nested_calls_and_preserves_links(native):
    tokenizer, codec, tools = native
    assert codec.stop_token_ids == [200001, 200008]
    assert 200007 not in codec.stop_token_ids
    assert codec.generation_prompt == ids(tokenizer, "<|start|>assistant")
    messages = [{"role": "user", "content": "Inspect the repository."},
                {"role": "assistant", "content": "", "reasoning_content": "Inspect first.",
                 "tool_calls": [{"id": "call_one", "type": "function",
                                 "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}]},
                {"role": "tool", "tool_call_id": "call_one", "content": "OBSERVATION_SENTINEL"}]
    before = copy.deepcopy(messages)
    canonical = copy.deepcopy(messages)
    canonical[1]["tool_calls"][0]["function"]["arguments"] = {"command": "pwd"}
    expected = tokenizer.apply_chat_template(canonical, tools=tools, tokenize=True,
        add_generation_prompt=True, current_date="2026-10-13", reasoning_strength="high", return_dict=False)
    assert codec.encode_full(messages, tools) == expected
    assert messages == before
    assert codec.model_messages_for(messages, tools)[1]["tool_calls"][0]["id"] == "call_one"
    delta = codec.encode_incremental(messages[-1:], history_messages=messages)
    text = tokenizer.decode(delta, skip_special_tokens=False)
    assert 'tool bash<|message|>' in text and 'tool call_one' not in text
    with pytest.raises(ValueError, match="matching function"):
        codec.encode_incremental(messages[-1:])


@pytest.mark.parametrize("eos", [200001, 200008])
def test_each_publisher_eos_finishes_without_leaking_framing(native, eos):
    tokenizer, codec, tools = native
    response = ids(tokenizer, " to=user<|message|>Done.") + [eos]
    message, finish = asyncio.run(codec.decode_response(response, tools=tools, stop_reason="stop"))
    assert message == {"role": "assistant", "content": "Done."}
    assert finish == "stop"


@pytest.mark.parametrize("channel", ["self", "user"])
def test_quoted_invokes_never_become_actions(native, channel):
    tokenizer, codec, tools = native
    end = "<|eom|><|start|>assistant to=user<|message|>Done.<|eot|>" if channel == "self" else "<|eot|>"
    raw = f" to={channel}<|message|>Quoted example: {invoke('fake')}{end}"
    message, finish = asyncio.run(codec.decode_response(ids(tokenizer, raw), tools=tools, stop_reason="stop"))
    assert finish == "stop" and "tool_calls" not in message
    assert "fake" in message["reasoning_content" if channel == "self" else "content"]


def test_multiple_reasoning_spans_and_bare_tool_name(native):
    tokenizer, codec, tools = native
    raw = (" to=self<|message|>First.<|eom|><|start|>assistant to=self<|message|>Second.<|eom|>"
           "<|start|>assistant to=bash.bash<|message|>" + invoke(name="bash.bash") + "<|eot|>")
    message, finish = asyncio.run(codec.decode_response(ids(tokenizer, raw), tools=tools, stop_reason="stop"))
    assert finish == "tool_calls"
    assert "First." in message["reasoning_content"] and "Second." in message["reasoning_content"]
    assert message["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"command": "pwd"}


@pytest.mark.parametrize("stop_reason", ["stop", "length"])
def test_incomplete_invoke_is_feedback_or_cap_not_bash(native, stop_reason):
    tokenizer, codec, tools = native
    raw = ' to=bash<|message|><atem:function_calls><atem:invoke name="bash"><atem:parameter name="command">pwd'
    if stop_reason == "stop":
        raw += "<|eot|>"
    message, finish = asyncio.run(codec.decode_response(ids(tokenizer, raw), tools=tools, stop_reason=stop_reason))
    assert finish == ("length" if stop_reason == "length" else "tool_calls")
    assert message["tool_calls"][0]["function"]["name"] == "__malformed_tool_call__"
    assert "bash" != message["tool_calls"][0]["function"]["name"]


@pytest.mark.parametrize("multiple_reasoning", [False, True])
def test_two_turn_incremental_keeps_raw_generations_and_masks_observation(native, multiple_reasoning):
    tokenizer, codec, tools = native
    from uni_agent.gateway.session import GatewaySession, SessionHandle

    second_reasoning = "<|start|>assistant to=self<|message|>Second.<|eom|>" if multiple_reasoning else ""
    first = ids(tokenizer, " to=self<|message|>First.<|eom|>" + second_reasoning +
                "<|start|>assistant to=bash<|message|>" + invoke() + "<|eot|>")
    final = ids(tokenizer, " to=user<|message|>Done.<|eot|>")
    contexts = []

    class Backend:
        async def generate(self, **kwargs):
            contexts.append(list(kwargs["prompt_ids"]))
            assert kwargs["sampling_params"]["stop_token_ids"] == [200001, 200008]
            assert kwargs["sampling_params"]["ignore_eos"] is True
            response = first if len(contexts) == 1 else final
            return SimpleNamespace(token_ids=response, stop_reason="stop", extra_fields={}, log_probs=None)

    async def run():
        session = GatewaySession(SessionHandle("muse-native"), codec,
                                 prompt_length=16384, response_length=114688)
        messages = [{"role": "user", "content": "Inspect the repository."}]
        initial = await session.run_generation({"messages": messages, "tools": tools,
                                               "sampling_params": {}}, Backend())
        assert initial.finish_reason == "tool_calls"
        observation = {"role": "tool", "tool_call_id": initial.assistant_msg["tool_calls"][0]["id"],
                       "content": "OBSERVATION_SENTINEL\nTool metadata: {'returncode': 0}"}
        complete_history = [*messages, initial.assistant_msg, observation]
        outcome = await session.run_generation({"messages": complete_history, "tools": tools,
                                               "sampling_params": {}}, Backend())
        assert outcome.finish_reason == "stop"
        trajectories = await session.finalize()
        assert len(trajectories) == 1
        row = trajectories[0]
        assert contexts[1][:len(contexts[0]) + len(first)] == contexts[0] + first
        if not multiple_reasoning:
            # A single reasoning+call pair is representable by the semantic
            # dictionary. Interleaved channels above are intentionally checked
            # by their original generated tokens, never reconstructed storage.
            assert contexts[1] == codec.encode_full(complete_history, tools)
        assert row.response_ids[:len(first)] == first
        assert row.response_ids[-len(final):] == final
        assert [token for token, mask in zip(row.response_ids, row.response_mask, strict=True) if mask] == first + final
        observation_text = tokenizer.decode([token for token, mask in zip(row.response_ids, row.response_mask, strict=True)
                                            if not mask], skip_special_tokens=False)
        assert "OBSERVATION_SENTINEL" in observation_text
        assert 'tool bash<|message|>' in observation_text
        assert 'tool call_' not in observation_text
    asyncio.run(run())


@pytest.mark.parametrize("capped", [False, True])
def test_native_wire_obeys_mimo_length_guard_before_dispatch(native, capped):
    from unittest.mock import Mock

    from mimoagent.agents.bashonly.bashonly_agent import BashOnlyAgent
    from mimoagent.models.openai_chat import OpenAIChatModel
    from openai.types.chat import ChatCompletion
    from uni_agent.gateway.adapters.openai import openai_build_response
    from uni_agent.gateway.session.session import GenerationOutcome

    tokenizer, codec, tools = native
    raw = (" to=self<|message|>Plan.<|eom|><|start|>assistant to=bash<|message|>"
           + invoke() + ("" if capped else "<|eot|>"))
    response = ids(tokenizer, raw)
    message, finish = asyncio.run(codec.decode_response(response, tools=tools,
                                                       stop_reason="length" if capped else "stop"))
    outcome = GenerationOutcome(message, finish, 10, len(response))
    wire = openai_build_response(outcome, model="muse-fixture")
    model = OpenAIChatModel(model_name="muse-fixture", model_kwargs={
        "base_url": "http://unused.invalid/v1", "api_key": "unused-test-value"})
    completed = SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason="stop",
        message=SimpleNamespace(content="Done.", tool_calls=[]))])
    model.client.chat.completions.create = Mock(side_effect=[ChatCompletion.model_validate(wire), completed])
    environment = SimpleNamespace(config=SimpleNamespace(cwd="/testbed"), get_template_vars=lambda: {})
    agent = BashOnlyAgent(model, environment, system_template="test", instance_template="{{task}}")
    agent.execute_action = Mock(return_value={"output": "/testbed", "metadata": {"returncode": 0}})
    status, _ = agent.run("inspect the repository")
    assert status == ("LimitsExceeded" if capped else "Idle")
    assert agent.execute_action.call_count == (0 if capped else 1)
    assert any(message.get("reasoning_content") == "Plan." for message in agent.messages)
