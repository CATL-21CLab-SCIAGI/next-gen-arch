"""Public bash conversion and the upstream pretokenized SFT data contract."""

import asyncio
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from archlab.data.agentic_sft import (
    PreparedAgenticSFTDataset,
    convert_mini_swe_messages,
    encode_assistant_targets,
    normalize_bash_messages,
)


def public_messages():
    return [
        {"role": "system", "content": "Source fenced-action instructions must be discarded."},
        {"role": "user", "content": "<pr_description>\nConsider the following PR description:\n"
                                    "Fix the parser.\n</pr_description>\nSource action instructions."},
        {"role": "assistant", "content": "THOUGHT: Inspect the repository.\n\n"
                                         "```bash\nls /home/nonroot/testbed\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\n"
                                    "OBSERVATION_SENTINEL /home/nonroot/testbed\n</output>"},
        {"role": "assistant", "content": "THOUGHT: Finished.\n\n```bash\n"
                                         "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                                         " && git add -A && git diff --cached\n```"},
    ]


def convert(messages, formatter=lambda output: output["output"]):
    return convert_mini_swe_messages(
        messages, system="Runtime system.", instance_template="Fix the following issue:\n\n{{task}}",
        format_observation=formatter, source_cwd="/home/nonroot/testbed", runtime_cwd="/testbed",
    )


def public_excerpt(elided_chars=123):
    warning = (
        "The output of your last command was too long.\n"
        "Please try a different command that produces less output.\n"
        "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\n"
        "If you're using grep or find and it produced too much output, you can use a more selective search pattern.\n"
        "If you really need to see something from the full command's output, you can redirect output to a file and then search in that file."
    )
    head = "OBSERVATION_SENTINEL /home/nonroot/testbed".ljust(5000, "h")
    return (f"<returncode>1</returncode>\n<warning>\n{warning}\n</warning>"
            f"<output_head>\n{head}\n</output_head>\n<elided_chars>\n{elided_chars} characters elided\n"
            f"</elided_chars>\n<output_tail>\n{'t' * 5000}\n</output_tail>")


def convert_excerpt(messages, formatter=lambda output: output["output"]):
    return convert_mini_swe_messages(
        messages, system="Runtime system.", instance_template="Fix the following issue:\n\n{{task}}",
        format_observation=formatter, source_cwd="/home/nonroot/testbed", runtime_cwd="/testbed",
        allow_observation_excerpts=True,
    )


@pytest.mark.parametrize("elided_chars", [0, 123])
def test_public_excerpt_requires_opt_in_and_preserves_delivered_body(elided_chars):
    messages = public_messages()
    messages[3]["content"] = public_excerpt(elided_chars)
    with pytest.raises(ValueError, match="truncated"):
        convert(messages)
    observations = []

    def formatter(observation):
        observations.append(observation)
        return observation["output"]

    converted = convert_excerpt(messages, formatter)
    body = public_excerpt(elided_chars).split("\n", 1)[1].replace("/home/nonroot/testbed", "/testbed")
    assert observations == [{"output": body, "metadata": {"returncode": 1}}]
    assert converted[3]["content"] == body
    assert len(converted) == 5
    assert f"{elided_chars} characters elided" in converted[3]["content"]
    assert converted[2] == convert(public_messages())[2]


@pytest.mark.parametrize("mutation", [
    lambda text: text.removesuffix("</output_tail>"),
    lambda text: text.replace("</output_head>", ""),
    lambda text: text.replace("123 characters elided", "-1 characters elided"),
    lambda text: text.replace("123 characters elided", "0123 characters elided"),
    lambda text: text.replace("123 characters elided", "unknown characters elided"),
    lambda text: text.replace("123 characters elided", "١٢٣ characters elided"),
    lambda text: text.replace("command was too long", "command may have been too long"),
    lambda text: text.replace("h\n</output_head>", "\n</output_head>"),
    lambda text: text.replace("t\n</output_tail>", "\n</output_tail>"),
])
def test_public_excerpt_rejects_unclosed_or_nonofficial_structure(mutation):
    messages = public_messages()
    messages[3]["content"] = mutation(public_excerpt())
    with pytest.raises(ValueError, match="truncated"):
        convert_excerpt(messages)


def test_public_conversion_retains_commands_and_uses_runtime_formatter():
    observations = []

    def formatter(output):
        observations.append(output)
        return output["output"] + f"\n\nTool metadata: {output['metadata']}"

    messages = convert(public_messages(), formatter)
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[1]["content"] == "Fix the following issue:\n\nFix the parser."
    call = messages[2]["tool_calls"][0]
    assert call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {"command": "ls /testbed"}
    assert messages[3]["tool_call_id"] == call["id"]
    assert observations == [{"output": "OBSERVATION_SENTINEL /testbed\n", "metadata": {"returncode": 0}}]
    assert messages[3]["content"].endswith("Tool metadata: {'returncode': 0}")
    assert messages[-1] == {"role": "assistant", "content": "THOUGHT: Finished."}
    assert "Source" not in json.dumps(messages)
    assert "COMPLETE_TASK" not in json.dumps(messages)


@pytest.mark.parametrize("output", ["", "edited", "edited\n", "\n", "edited\n\n"])
def test_public_conversion_preserves_complete_output_body_verbatim(output):
    messages = public_messages()
    messages[3]["content"] = f"<returncode>0</returncode>\n<output>\n{output}</output>"
    observations = []

    def formatter(observation):
        observations.append(observation)
        return observation["output"]

    converted = convert(messages, formatter)
    assert observations == [{"output": output, "metadata": {"returncode": 0}}]
    assert converted[3]["content"] == output


@pytest.mark.parametrize("mutation,match", [
    (lambda messages: messages[3].update(content="<returncode>0</returncode>\n<output_head>clipped</output_head>"), "truncated"),
    (lambda messages: messages[3].update(content="<returncode>0</returncode>\n<output>\nincomplete"), "truncated"),
    (lambda messages: messages[2].update(content="```bash\npwd\n```\n```bash\nls\n```"), "exactly one"),
    (lambda messages: messages.pop(3), "no complete observation"),
    (lambda messages: messages[4].update(content="THOUGHT: Done.\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && rm -rf x\n```"), "Unsupported submission"),
    (lambda messages: messages[1].update(content="Missing original problem."), "PR description"),
])
@pytest.mark.parametrize("converter", [convert, convert_excerpt])
def test_public_conversion_rejects_ambiguous_or_incomplete_records(mutation, match, converter):
    messages = public_messages()
    mutation(messages)
    with pytest.raises(ValueError, match=match):
        converter(messages)


def test_structured_conversion_rejects_other_tools_and_broken_links():
    messages = convert(public_messages())
    messages[2]["tool_calls"][0]["function"]["name"] = "str_replace_editor"
    with pytest.raises(ValueError, match="non-bash"):
        normalize_bash_messages(messages, system="Runtime")
    messages = convert(public_messages())
    messages[3]["tool_call_id"] = "unrelated"
    with pytest.raises(ValueError, match="no matching"):
        normalize_bash_messages(messages, system="Runtime")


@pytest.fixture(params=[False, True], ids=["complete-stdout", "delivered-excerpt"])
def native(monkeypatch, request):
    model = os.environ.get("ARCHLAB_NATIVE_TOKENIZER")
    if not model:
        pytest.skip("needs qualified native tokenizer and upstream runtime")
    root = Path(__file__).resolve().parents[1]
    for directory in ("third_party/verl", "third_party/verl/third_party/uni_agent",
                      "third_party/verl/third_party/mimoagent-osr/src"):
        monkeypatch.syspath_prepend(str(root / directory))
    from mimoagent.agents.bashonly.bashonly_agent import BashOnlyAgentConfig
    from mimoagent.agents.default import DefaultAgent
    from mimoagent.tools.registry import ToolRegistry
    from transformers import AutoTokenizer
    from uni_agent.gateway.session import MessageCodec

    config = BashOnlyAgentConfig()
    agent = object.__new__(DefaultAgent)
    agent.config = config
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    codec = MessageCodec(tokenizer, tool_parser_name="hermes", rollout_backend="verl",
                         generation_eos_token_id=151645)
    tools = ToolRegistry.from_config(config.tools).get_function_definitions()
    source_messages = public_messages()
    if request.param:
        source_messages[3]["content"] = public_excerpt()
    messages = convert_mini_swe_messages(
        source_messages, system=config.system_template.replace("{{cwd}}", "/testbed"),
        instance_template=config.instance_template, format_observation=agent._format_observation,
        source_cwd="/home/nonroot/testbed", runtime_cwd="/testbed",
        allow_observation_excerpts=request.param,
    )
    return tokenizer, codec, tools, messages


def test_native_masks_observations_out_and_parses_supervised_bash(native):
    tokenizer, codec, tools, messages = native
    row = encode_assistant_targets(tokenizer, codec, messages, tools, eos_token_id=151645, max_length=16384)
    targets = [token for token, target in zip(row["input_ids"], row["loss_mask"], strict=True) if target]
    target_text = tokenizer.decode(targets, skip_special_tokens=False)
    assert "OBSERVATION_SENTINEL" in tokenizer.decode(row["input_ids"], skip_special_tokens=False)
    assert "OBSERVATION_SENTINEL" not in target_text
    assert "Tool metadata:" not in target_text
    assert "Fix the parser" not in target_text
    assert "<tool_call>" in target_text and "ls /testbed" in target_text
    assert "THOUGHT: Finished." in target_text
    assert targets.count(151645) == 2
    assert row["input_ids"][0] == tokenizer.eos_token_id == 151643
    assert row["loss_mask"][0] == 0
    first_response = targets[:targets.index(151645) + 1]
    parsed, finish = asyncio.run(codec.decode_response(first_response, tools=tools, stop_reason="stop"))
    assert finish == "tool_calls"
    function = parsed["tool_calls"][0]["function"]
    assert function["name"] == "bash"
    assert json.loads(function["arguments"]) == {"command": "ls /testbed"}
    with pytest.raises(ValueError, match="exceeds context"):
        encode_assistant_targets(tokenizer, codec, messages, tools, eos_token_id=151645,
                                 max_length=len(row["input_ids"]) - 1)
    messages[-1]["content"] += "<|im_end|>"
    with pytest.raises(ValueError, match="exactly one native EOS"):
        encode_assistant_targets(tokenizer, codec, messages, tools, eos_token_id=151645, max_length=16384)


def prepared_row():
    return {"input_ids": [151643, 42, 151645], "attention_mask": [1, 1, 1],
            "loss_mask": [0, 1, 1], "position_ids": [0, 1, 2]}


def test_prepared_dataset_preserves_unshifted_eos_mask(tmp_path):
    import torch

    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([prepared_row(), prepared_row()]), path)
    dataset = PreparedAgenticSFTDataset(str(path), None, {"max_length": 3}, max_samples=1)
    assert len(dataset) == 1
    assert set(dataset[0]) == {"input_ids", "attention_mask", "loss_mask", "position_ids"}
    assert all(tensor.dtype == torch.int64 for tensor in dataset[0].values())
    assert dataset[0]["loss_mask"].tolist() == [0, 1, 1]


def multi_row(length):
    return {"input_ids": [151643] + [7] * (length - 2) + [151645], "attention_mask": [1] * length,
            "loss_mask": [0] * (length - 1) + [1], "position_ids": list(range(length))}


def test_prepared_dataset_rows_span_files_and_row_boundaries(tmp_path):
    rows = [multi_row(length) for length in (2, 5, 3, 4)]
    paths = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    pq.write_table(pa.Table.from_pylist(rows[:1]), paths[0])
    pq.write_table(pa.Table.from_pylist(rows[1:]), paths[1])
    dataset = PreparedAgenticSFTDataset([str(path) for path in paths], None, {"max_length": 5})
    assert [{key: value.tolist() for key, value in dataset[i].items()} for i in range(len(dataset))] == rows
    assert len(PreparedAgenticSFTDataset([str(path) for path in paths], None, {}, max_samples=3)) == 3


@pytest.mark.parametrize("updates,match", [
    ({"position_ids": [4, 5, 6]}, "contiguous"),
    ({"position_ids": [0, 2, 3]}, "contiguous"),
    ({"attention_mask": [1, 1, 0]}, "padding"),
    ({"loss_mask": [1, 0, 1]}, "token/mask"),
    ({"loss_mask": [0, 0, 0]}, "token/mask"),
    ({"input_ids": [151643, 7, 8]}, "no supervised native EOS"),
])
def test_prepared_dataset_checks_every_row(tmp_path, updates, match):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist([multi_row(4), multi_row(3) | updates, multi_row(2)]), path)
    with pytest.raises(ValueError, match=match):
        PreparedAgenticSFTDataset(str(path), None, {})


@pytest.mark.parametrize("updates,config,match", [
    ({"loss_mask": [1, 1, 1]}, {}, "token/mask"),
    ({"loss_mask": [0, 1, 0]}, {}, "no supervised native EOS"),
    ({"attention_mask": [1, 0, 1]}, {}, "padding"),
    ({"position_ids": [0, 1, 3]}, {}, "contiguous"),
    ({}, {"max_length": 2}, "token/mask"),
])
def test_prepared_dataset_fails_closed(tmp_path, updates, config, match):
    row = prepared_row() | updates
    path = tmp_path / "invalid.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    with pytest.raises(ValueError, match=match):
        PreparedAgenticSFTDataset(str(path), None, config)
