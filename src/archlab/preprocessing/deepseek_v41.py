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


def prepare_messages(row):
    from archlab.preprocessing.nemotron_math import normalize_row

    messages, has_tools = normalize_row(row)
    if messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls"):
        raise ValueError("Expected a completed assistant answer, not an unfinished tool call")
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
        messages, has_tools = prepare_messages(row)
        text = self.encode(messages)
        if not text.startswith(self.encoder.bos_token) or not text.endswith(self.encoder.eos_token):
            raise ValueError("Official encoding lost the native conversation boundaries")
        spans = []
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            if index == 0 or messages[index - 1]["role"] == "assistant":
                raise ValueError("Assistant-only/consecutive-assistant source requires explicit policy")
            prefix = self.encode(messages[:index])
            through_answer = self.encode(messages[:index + 1])
            if not text.startswith(prefix) or not text.startswith(through_answer):
                raise ValueError("Native encoding is not prefix-stable at assistant boundaries")
            spans.append((len(prefix), len(through_answer)))
            if len(prefix) >= len(through_answer):
                raise ValueError("Empty assistant supervision span")
        return text, has_tools, messages, {"assistant_character_spans": spans}


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
