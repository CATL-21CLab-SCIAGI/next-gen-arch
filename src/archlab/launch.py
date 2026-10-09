"""Shared backend launch-plan primitives."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_ENV_TOKEN = re.compile(r"^env:([A-Z][A-Z0-9_]*)$")


def _shell_token(token: str) -> str:
    match = _ENV_TOKEN.fullmatch(token)
    if match:
        return f'"${{{match.group(1)}}}"'
    return shlex.quote(token)


@dataclass(frozen=True)
class LaunchPlan:
    """An inspectable command, environment, and provenance bundle."""

    backend: str
    argv: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def shell(self) -> str:
        prefix = [f"{key}={shlex.quote(value)}" for key, value in sorted(self.env.items())]
        return " ".join([*prefix, *(_shell_token(token) for token in self.argv)])

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "argv": list(self.argv),
            "env": dict(self.env),
            "metadata": dict(self.metadata),
        }

    def json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"


def get_backend(name: str):
    """Construct one of the two intentionally supported execution backends."""
    if name == "speedrun":
        from archlab.speedrun.backend import SpeedrunBackend

        return SpeedrunBackend()
    if name == "megatron":
        from archlab.megatron.backend import MegatronBackend

        return MegatronBackend()
    raise ValueError(f"unknown backend: {name}")
