"""Read standalone YAML contracts or an explicit profile from a recipe catalog."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def split_recipe_reference(reference: str | Path) -> tuple[Path, str | None]:
    path, marker, profile = str(reference).partition("#")
    if not path or (marker and not profile):
        raise ValueError("recipe reference needs a path and a nonempty profile after #")
    return Path(path), profile if marker else None


def read_recipe(reference: str | Path) -> dict[str, Any]:
    """Select a profile explicitly; never infer a variant from a path or output name."""
    path, profile = split_recipe_reference(reference)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"recipe must be a mapping: {reference}")
    if profile is not None:
        profiles = value.get("profiles")
        if not isinstance(profiles, dict) or profile not in profiles:
            raise ValueError(f"unknown recipe profile {profile!r} in {path}")
        value = profiles[profile]
        if not isinstance(value, dict):
            raise ValueError(f"recipe profile must be a mapping: {reference}")
    elif "profiles" in value:
        raise ValueError(f"catalog requires an explicit #PROFILE: {path}")
    return copy.deepcopy(value)


def recipe_sha256(reference: str | Path) -> str:
    """Bind a launch receipt to its selected contract; retain legacy raw-file hashes."""
    path, profile = split_recipe_reference(reference)
    if profile is None:
        payload = path.read_bytes()
    else:
        payload = json.dumps(read_recipe(reference), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
