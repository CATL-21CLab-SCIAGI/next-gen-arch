"""Use the revision-pinned upstream V4.1 encoder without copying its logic.

The small upstream assets are supplied at launch, independently of model weights.
No model loading, network access, package changes, or tool execution occurs here.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from bisect import bisect_left
from pathlib import Path

MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "df42c109f1defefcbfcedbe7d905718a12266e40"
ENCODER_SHA256 = "b4bd8a06f94a06a61564156c7a543d649ed86ecdd72a4dfb94b064e16d9c572f"
ASSISTANT_MESSAGE_POLICY = "lossless-assistant-sequences-and-terminal-calls-v2"


def coalesce_reasoning_fragments(row):
    """Join split analysis before one answer/call, never merge completed answers.

    The accepted v1 domain (no adjacent assistant messages) is unchanged. Retain
    every reasoning character, insert a documented separator, and keep the final
    message's content/call intact. Source indices refer to the original row.
    """
    messages = row["messages"]
    merged, repairs = [], []
    index = 0
    while index < len(messages):
        end = index + 1
        if messages[index].get("role") == "assistant":
            while end < len(messages) and messages[end].get("role") == "assistant":
                end += 1
        if end == index + 1:
            merged.append(messages[index])
        else:
            group = messages[index:end]
            safe_prefixes = all(
                not fragment.get("content") and not fragment.get("tool_calls")
                and isinstance(fragment.get("reasoning_content"), str) and fragment["reasoning_content"]
                for fragment in group[:-1]
            )
            if not safe_prefixes:
                # Native encoding accepts adjacent assistants. Keep each original
                # message/EOS/call in order, never fabricate a missing tool result
                # or turn multiple calls into a single parallel call batch.
                merged.extend(group)
                repairs.append(dict(
                    policy="preserve-native-consecutive-assistants-v1",
                    source_message_indices=list(range(index, end)),
                    messages_merged=False,
                ))
                index = end
                continue
            for fragment in group[:-1]:
                for key, value in fragment.items():
                    if key not in {"role", "content", "reasoning_content", "tool_calls"} and value not in (None, "", [], {}):
                        raise ValueError(f"Cannot discard assistant fragment field {key!r}")
            reasoning = [message.get("reasoning_content") or "" for message in group]
            if any(not isinstance(value, str) for value in reasoning):
                raise ValueError("Assistant reasoning must be text")
            final = copy.deepcopy(group[-1])
            final["reasoning_content"] = "\n\n".join(reasoning)
            merged.append(final)
            repairs.append(dict(
                policy="merge-consecutive-reasoning-only-prefixes-v1",
                source_message_indices=list(range(index, end)),
                reasoning_character_lengths=[len(value) for value in reasoning],
                reasoning_sha256=[hashlib.sha256(value.encode()).hexdigest() for value in reasoning],
                separator="\n\n",
            ))
        index = end
    return ({**row, "messages": merged} if repairs else row), repairs


def prepare_messages(row):
    from archlab.preprocessing.nemotron_math import normalize_row

    messages, has_tools = normalize_row(row)
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported source message role")
        if not isinstance(message.get("content", ""), str):
            raise ValueError("This corpus contract is text-only")
        if message.get("wo_eos") or message.get("task"):
            raise ValueError("Unexpected incomplete/internal-task message")
    tools = row.get("tools") or []
    if tools:
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        if messages[0].get("tools") and messages[0]["tools"] != tools:
            raise ValueError("Conflicting source tool definitions")
        messages[0]["tools"] = copy.deepcopy(tools)
    return messages, has_tools


class DeepSeekV41Renderer:
    def __init__(self, asset_root, reasoning_effort=75):
        if type(reasoning_effort) is not int or not 1 <= reasoning_effort <= 100:
            raise ValueError("reasoning_effort must be an integer in 1..100")
        self.reasoning_effort = reasoning_effort
        path = Path(asset_root) / "encoding/encoding.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != ENCODER_SHA256:
            raise ValueError("Official encoder does not match the reviewed V4.1 revision")
        spec = importlib.util.spec_from_file_location("archlab_upstream_deepseek_v41_encoding", path)
        self.encoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.encoder)

    def encode(self, messages):
        text, media = self.encoder.encode_messages(
            messages, thinking_mode="thinking", drop_thinking=False,
            reasoning_effort=self.reasoning_effort, return_multi_modal_data=True,
        )
        if media.get("images"):
            raise ValueError("Unexpected image in text-only math corpus")
        return text

    def render(self, row):
        source_last_index = len(row["messages"]) - 1
        row, repairs = coalesce_reasoning_fragments(row)
        messages, has_tools = prepare_messages(row)
        if messages[-1].get("tool_calls"):
            repairs.append(dict(
                policy="preserve-terminal-assistant-tool-call-v1",
                source_message_indices=[source_last_index],
                complete_answer=False,
                tool_result_fabricated=False,
            ))
        if messages[-1]["role"] != "assistant":
            repairs.append(dict(
                policy="preserve-native-nonassistant-ending-v1",
                source_message_indices=[source_last_index],
                terminal_role=messages[-1]["role"],
                complete_answer=False,
                assistant_answer_fabricated=False,
            ))
        text = self.encode(messages)
        if not text.startswith(self.encoder.bos_token):
            raise ValueError("Official encoding lost the native conversation boundaries")
        if messages[-1]["role"] == "assistant" and not text.endswith(self.encoder.eos_token):
            raise ValueError("Official encoding lost the native assistant ending")
        spans = []
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            if index == 0:
                raise ValueError("Assistant-only source requires explicit policy")
            prefix = self.encode(messages[:index])
            through_answer = self.encode(messages[:index + 1])
            if not text.startswith(prefix) or not text.startswith(through_answer):
                raise ValueError("Native encoding is not prefix-stable at assistant boundaries")
            spans.append((len(prefix), len(through_answer)))
            if len(prefix) >= len(through_answer):
                raise ValueError("Empty assistant supervision span")
        metadata = {"assistant_character_spans": spans}
        if repairs:
            metadata["message_repairs"] = repairs
        return text, has_tools, messages, metadata


def token_spans(offsets, character_spans):
    """Translate native assistant boundaries without training on prompt/tool text.

    Fail closed if BPE merges across a supervision boundary. Metadata uses target
    token indices before next-token shifting; the future trainer must shift once.
    """
    starts = [start for start, _ in offsets]
    ends = [end for _, end in offsets]
    result = []
    for left, right in character_spans:
        begin = bisect_left(starts, left)
        stop = bisect_left(starts, right)
        if begin == stop or starts[begin] != left or ends[stop - 1] != right:
            raise ValueError("Tokenizer crosses an assistant supervision boundary")
        if begin and ends[begin - 1] > left:
            raise ValueError("Tokenizer overlaps assistant supervision boundary")
        result.append([begin, stop])
    return result
