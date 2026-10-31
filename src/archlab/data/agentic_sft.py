"""Prepare assistant-only SFT targets against the agent gateway's real template.

Prepared rows contain target-aligned (unshifted) masks. The upstream SFT engine
owns next-token shifting, batching and optimization; observations are context.
No commands or task verifiers are executed during preparation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

_MINI_SWE_OUTPUT_WARNING = (
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\n"
    "If you're using grep or find and it produced too much output, you can use a more selective search pattern.\n"
    "If you really need to see something from the full command's output, you can redirect output to a file and then search in that file."
)
_MINI_SWE_OUTPUT_EXCERPT = re.compile(
    r"<returncode>(?P<returncode>-?[0-9]+)</returncode>\n"
    r"(?P<body><warning>\n(?P<warning>.*?)\n</warning><output_head>\n"
    r"(?P<head>.*)\n</output_head>\n<elided_chars>\n"
    r"(?P<elided_chars>0|[1-9][0-9]*) characters elided\n</elided_chars>\n"
    r"<output_tail>\n(?P<tail>.*)\n</output_tail>)\s*",
    re.DOTALL,
)


def text_content(content: str | list[dict] | None) -> str:
    """Accept text-only public chat records without discarding other modalities."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or any(part.get("type") != "text" for part in content):
        raise ValueError("Only text message content is supported")
    return "\n".join(part["text"] for part in content)


def convert_mini_swe_messages(
    messages: list[dict], *, system: str, instance_template: str,
    format_observation: Callable[[dict], str], source_cwd: str, runtime_cwd: str,
    allow_observation_excerpts: bool = False,
) -> list[dict]:
    """Project strict mini-swe bash records onto the deployed agent's protocol.

    Complete stdout is required by default. The explicit excerpt contract also
    accepts the official, fully closed warning/head/omission-count/tail body,
    preserving that delivered body verbatim rather than inventing missing stdout.
    Commands become structured bash calls and observations use the runtime's
    formatter. The documented submission sentinel becomes an idle assistant
    response. Paths are rebased explicitly, never inferred or executed.
    """
    source = [message for message in messages if message["role"] != "system"]
    if len(source) < 3 or source[0]["role"] != "user":
        raise ValueError("Missing mini-swe issue prompt")
    problem = re.search(r"<pr_description>\n(.*?)\n</pr_description>",
                        text_content(source[0]["content"]), re.DOTALL)
    if problem is None:
        raise ValueError("Missing mini-swe PR description")
    problem = problem.group(1).removeprefix("Consider the following PR description:\n")

    def rebase(text):
        return text.replace(source_cwd, runtime_cwd)

    result = [{"role": "user", "content": instance_template.replace("{{task}}", rebase(problem))}]
    pending = None
    for index, message in enumerate(source[1:], 1):
        original_content = text_content(message.get("content"))
        content = rebase(original_content)
        if message["role"] == "assistant":
            if pending is not None:
                raise ValueError("mini-swe action has no complete observation")
            action = re.fullmatch(r"(.*?)```bash\n(.*?)\n```\s*", content, re.DOTALL)
            if action is None or "```" in action.group(1) or "```" in action.group(2):
                raise ValueError("Expected exactly one fenced bash action")
            thought, command = action.group(1).strip(), action.group(2)
            if not thought or not command.strip():
                raise ValueError("Missing mini-swe thought or bash command")
            terminal = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"
            if command.strip() == terminal:
                if index != len(source) - 1:
                    raise ValueError("Submission must be the terminal source turn")
                result.append({"role": "assistant", "content": thought})
                continue
            if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
                raise ValueError("Unsupported submission command")
            pending = f"public-bash-{index}"
            result.append({"role": "assistant", "content": thought, "tool_calls": [{
                "id": pending, "type": "function", "function": {
                    "name": "bash", "arguments": json.dumps({"command": command}, ensure_ascii=False),
                },
            }]})
        elif message["role"] == "user" and pending is not None:
            observation = re.fullmatch(
                r"<returncode>(-?\d+)</returncode>\n<output>\n(.*)</output>\s*",
                content, re.DOTALL,
            )
            if observation is not None:
                body, returncode = observation.group(2), int(observation.group(1))
            else:
                excerpt = _MINI_SWE_OUTPUT_EXCERPT.fullmatch(original_content) if allow_observation_excerpts else None
                if (excerpt is None or excerpt["warning"] != _MINI_SWE_OUTPUT_WARNING
                        or len(excerpt["head"]) != 5000 or len(excerpt["tail"]) != 5000):
                    raise ValueError("Incomplete, truncated or nonstandard mini-swe observation")
                body, returncode = rebase(excerpt["body"]), int(excerpt["returncode"])
            output = format_observation({"output": body, "metadata": {"returncode": returncode}})
            result.append({"role": "tool", "tool_call_id": pending, "content": output})
            pending = None
        else:
            raise ValueError("Unexpected mini-swe role or turn order")
    return normalize_bash_messages(result, system=system)


def normalize_bash_messages(messages: list[dict], *, system: str) -> list[dict]:
    """Keep bash commands/observations exactly; reject other tools and bad links.

    The system instruction is supplied by the deployed BashOnlyAgent. Public
    source instructions are retained separately in provenance, never mixed into
    the runtime tool declaration. Assistant tool arguments remain structured so
    MessageCodec and the native template own their wire representation.
    """
    result = [{"role": "system", "content": system}]
    pending: set[str] = set()
    calls = 0
    for message in messages:
        role = message["role"]
        if role == "system":
            continue
        if role not in {"user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {role}")
        item = {"role": role, "content": text_content(message.get("content"))}
        if role == "assistant":
            if pending:
                raise ValueError("Assistant turn precedes its tool observations")
            tool_calls = message.get("tool_calls") or []
            if len(tool_calls) > 1:
                raise ValueError("Bootstrap requires one bash call per assistant turn")
            if tool_calls:
                call = tool_calls[0]
                function = call["function"]
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if function["name"] != "bash":
                    raise ValueError("Trajectory uses a non-bash tool")
                if (not isinstance(arguments, dict) or not arguments.get("command")
                        or set(arguments) - {"command", "timeout"}):
                    raise ValueError("Bash arguments differ from the runtime tool contract")
                if "timeout" in arguments and (type(arguments["timeout"]) is not int
                                                or not 0 < arguments["timeout"] <= 300):
                    raise ValueError("Bash timeout is outside the runtime contract")
                call_id = call["id"]
                if not call_id:
                    raise ValueError("Missing tool-call identity")
                item["tool_calls"] = [{"id": call_id, "type": "function", "function": {
                    "name": "bash", "arguments": json.dumps(arguments, ensure_ascii=False)}}]
                pending.add(call_id)
                calls += 1
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("Tool observation has no matching bash call")
            item["tool_call_id"] = call_id
            pending.remove(call_id)
        result.append(item)
    if pending or not calls or len(result) < 4 or result[1]["role"] != "user":
        raise ValueError("Incomplete bash trajectory")
    if result[-1]["role"] != "assistant" or result[-1].get("tool_calls"):
        raise ValueError("Trajectory must finish with an idle assistant response")
    return result


def encode_assistant_targets(tokenizer, codec, messages: list[dict], tools: list[dict], *,
                             eos_token_id: int, max_length: int) -> dict:
    """Render the native conversation and prove exact assistant/EOS mask spans.

    Every assistant prefix must equal the gateway's generation request. This
    rejects templates whose historical serialization cannot safely reproduce
    sampled turns. No clipping or tokenizer/template mutation is permitted.
    """
    def render(history):
        prompt = tokenizer.apply_chat_template(
            codec.model_messages_for(history, tools), tools=codec.model_tools_for(tools),
            tokenize=False, add_generation_prompt=False,
        )
        # Match MessageCodec.encode_full exactly. In Transformers 5 the native
        # template's tokenize=True result is a BatchEncoding, not token IDs.
        return tokenizer.encode(prompt, add_special_tokens=False)

    ids = render(messages)
    if len(ids) > max_length:
        raise ValueError(f"Trajectory exceeds context: {len(ids)} > {max_length}")
    mask = [0] * len(ids)
    turns = 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before = codec.encode_full(messages[:index], tools)
        after = render(messages[:index + 1])
        if after[:len(before)] != before or ids[:len(after)] != after:
            raise ValueError("Native assistant history is not prefix-stable")
        ends = [i for i in range(len(before), len(after)) if after[i] == eos_token_id]
        if len(ends) != 1 or ends[0] == len(before):
            raise ValueError("Assistant turn must contain content and exactly one native EOS")
        end = ends[0]
        if tokenizer.decode(after[end + 1:], skip_special_tokens=False).strip():
            raise ValueError("Unexpected template tokens after the assistant EOS")
        mask[len(before):end + 1] = [1] * (end + 1 - len(before))
        turns += 1
    if not turns or mask[0] or not sum(mask):
        raise ValueError("Missing assistant targets")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "loss_mask": mask,
            "position_ids": list(range(len(ids)))}


class PreparedAgenticSFTDataset:
    """Small pretokenized Parquet adapter for upstream verl's SFT trainer."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        import pyarrow.parquet as pq

        if processor is not None:
            raise ValueError("Prepared agentic SFT is text-only")
        files = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)
        self.rows = [row for path in files for row in pq.read_table(path).to_pylist()]
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.max_length = int(config.get("max_length", 131072))
        self.eos_token_id = int(config.get("generation_eos_token_id", 151645))
        for row in self.rows:
            ids, mask = row["input_ids"], row["loss_mask"]
            if (not ids or len(ids) > self.max_length or len(ids) != len(mask)
                    or mask[0] != 0 or not any(mask) or set(mask) - {0, 1}):
                raise ValueError("Invalid prepared SFT token/mask contract")
            if config.get("require_supervised_eos", True) and not any(
                token == self.eos_token_id and target for token, target in zip(ids, mask, strict=True)
            ):
                raise ValueError("Prepared trajectory has no supervised native EOS")
            if row["attention_mask"] != [1] * len(ids):
                raise ValueError("Prepared trajectory contains padding")
            if row["position_ids"] != list(range(len(ids))):
                raise ValueError("Prepared position IDs must be contiguous")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        return {key: torch.tensor(self.rows[index][key], dtype=torch.int64)
                for key in ("input_ids", "attention_mask", "loss_mask", "position_ids")}
