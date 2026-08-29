"""Portable, layered experiment specifications.

Committed specifications never contain machine-specific absolute paths. Runtime
locations are supplied through ``env:NAME`` references or explicit CLI overrides.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from archlab.contracts import ArtifactPolicy, ComparisonContract, ContractError

PACKAGE_ROOT = Path(__file__).resolve().parent
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


class SpecError(ValueError):
    """Raised when an experiment specification is incomplete or unsafe."""


def deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and sequence values."""
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SpecError(f"experiment configuration must be a mapping: {path}")
    return value


def _load_layered(path: Path, stack: tuple[Path, ...] = ()) -> tuple[dict[str, Any], list[Path]]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise SpecError(f"cyclic experiment extends chain: {chain}")
    value = _read_mapping(path)
    extends = value.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    if not isinstance(extends, list) or not all(isinstance(item, str) for item in extends):
        raise SpecError(f"extends must be a string or list of strings: {path}")

    merged: dict[str, Any] = {}
    sources: list[Path] = []
    for reference in extends:
        parent_value, parent_sources = _load_layered(path.parent / reference, (*stack, path))
        deep_merge(merged, parent_value)
        sources.extend(parent_sources)
    deep_merge(merged, value)
    sources.append(path)
    return merged, sources


def _validate_portable_reference(value: str, label: str) -> None:
    if value.startswith("env:"):
        name = value.removeprefix("env:")
        if not _ENV_NAME.fullmatch(name):
            raise SpecError(f"invalid environment reference for {label}: {value!r}")
        return
    if value.startswith("package:"):
        relative = Path(value.removeprefix("package:"))
        if relative.is_absolute() or ".." in relative.parts:
            raise SpecError(f"invalid package reference for {label}: {value!r}")
        return
    if Path(value).is_absolute():
        raise SpecError(
            f"committed path {label} must be portable, got absolute path {value!r}; "
            "use env:NAME or a CLI override"
        )


@dataclass(frozen=True)
class ExperimentSpec:
    """A backend-neutral experiment contract plus its composition provenance."""

    source: Path
    source_files: tuple[Path, ...]
    config: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.config["name"])

    @property
    def backend(self) -> str:
        return str(self.config["backend"])

    @property
    def variant(self) -> str:
        return str(self.config["variant"])

    @property
    def seed(self) -> int:
        return int(self.config["seed"])

    @property
    def prompts(self) -> str:
        return str(self.config.get("prompts", "package:prompts/smoke.yaml"))

    @property
    def model(self) -> dict[str, Any]:
        return dict(self.config.get("model", {}))

    @property
    def training(self) -> dict[str, Any]:
        return dict(self.config.get("training", {}))

    @property
    def parallelism(self) -> dict[str, Any]:
        return dict(self.config.get("parallelism", {}))

    @property
    def comparison(self) -> ComparisonContract | None:
        value = self.config.get("comparison")
        return None if value is None else ComparisonContract.from_mapping(value)

    @property
    def artifacts(self) -> ArtifactPolicy | None:
        value = self.config.get("artifacts")
        return None if value is None else ArtifactPolicy.from_mapping(value)

    def resolve_reference(
        self,
        value: str,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> str:
        """Resolve one portable path/asset reference at runtime."""
        environ = os.environ if environ is None else environ
        if value.startswith("env:"):
            name = value.removeprefix("env:")
            resolved = environ.get(name)
            if not resolved:
                raise SpecError(f"required environment variable {name} is not set")
            return str(Path(resolved).expanduser().resolve())
        if value.startswith("package:"):
            relative = value.removeprefix("package:")
            return str((PACKAGE_ROOT / relative).resolve())
        return str((self.source.parent / value).resolve())

    def resolve_paths(
        self,
        overrides: Mapping[str, str] | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Resolve backend paths, with explicit overrides taking precedence."""
        overrides = {} if overrides is None else dict(overrides)
        configured = self.config.get("paths", {})
        if not isinstance(configured, Mapping):
            raise SpecError("paths must be a mapping")
        unknown = set(overrides) - set(configured)
        if unknown:
            raise SpecError(f"unknown path override(s): {', '.join(sorted(unknown))}")
        return {
            key: str(Path(overrides[key]).expanduser().resolve())
            if key in overrides
            else self.resolve_reference(str(value), environ=environ)
            for key, value in configured.items()
        }


def load_experiment(path: str | Path) -> ExperimentSpec:
    """Load, compose, and validate a portable experiment specification."""
    source = Path(path).expanduser().resolve()
    config, source_files = _load_layered(source)
    required = {"schema_version", "name", "backend", "variant", "seed"}
    missing = required - set(config)
    if missing:
        raise SpecError(f"missing experiment key(s): {', '.join(sorted(missing))}")
    if config["schema_version"] != 1:
        raise SpecError(f"unsupported schema_version: {config['schema_version']!r}")
    if config["backend"] not in {"speedrun", "megatron"}:
        raise SpecError(f"unknown backend: {config['backend']!r}")
    if int(config["seed"]) < 0:
        raise SpecError("seed must be non-negative")
    paths = config.get("paths", {})
    if not isinstance(paths, Mapping):
        raise SpecError("paths must be a mapping")
    for key, value in paths.items():
        if not isinstance(value, str):
            raise SpecError(f"path {key} must be a string")
        _validate_portable_reference(value, f"paths.{key}")
    prompts = config.get("prompts", "package:prompts/smoke.yaml")
    if not isinstance(prompts, str):
        raise SpecError("prompts must be a portable string reference")
    _validate_portable_reference(prompts, "prompts")
    research_keys = {key for key in ("comparison", "artifacts") if key in config}
    if research_keys and research_keys != {"comparison", "artifacts"}:
        raise SpecError("new research specs must define both comparison and artifacts")
    if research_keys:
        if not isinstance(config["comparison"], Mapping):
            raise SpecError("comparison must be a mapping")
        if not isinstance(config["artifacts"], Mapping):
            raise SpecError("artifacts must be a mapping")
        try:
            ComparisonContract.from_mapping(config["comparison"])
            ArtifactPolicy.from_mapping(config["artifacts"])
        except ContractError as error:
            raise SpecError(str(error)) from error
    return ExperimentSpec(source, tuple(source_files), config)
