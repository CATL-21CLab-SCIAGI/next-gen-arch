"""Load versioned prompt sets shared by training and evaluation entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PROMPT_PATH = Path(__file__).with_name("prompts") / "smoke.yaml"


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    tags: tuple[str, ...] = ()


def load_prompts(path: str | Path | None = None) -> tuple[Prompt, ...]:
    prompt_path = DEFAULT_PROMPT_PATH if path is None else Path(path).expanduser().resolve()
    value = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"unsupported prompt schema: {prompt_path}")
    rows = value.get("prompts")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"prompt set is empty: {prompt_path}")
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError(f"invalid prompt row in {prompt_path}")
        prompt_id = row["id"]
        text = row.get("text")
        if prompt_id in seen or not isinstance(text, str) or not text:
            raise ValueError(f"invalid or duplicate prompt {prompt_id!r} in {prompt_path}")
        tags = row.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"invalid tags for prompt {prompt_id!r}")
        seen.add(prompt_id)
        prompts.append(Prompt(prompt_id, text, tuple(tags)))
    return tuple(prompts)


def load_prompt_texts(path: str | Path | None = None) -> list[str]:
    return [prompt.text for prompt in load_prompts(path)]
