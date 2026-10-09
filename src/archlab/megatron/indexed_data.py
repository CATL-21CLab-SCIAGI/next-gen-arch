"""One strict DATA_READY membership validator for all indexed-data trainers."""

from __future__ import annotations

import json
from pathlib import Path


def data_prefixes(data_root: Path, split: str) -> list[Path]:
    prefixes = sorted(path.with_suffix("") for path in (data_root / split).glob("part-*.bin"))
    if not prefixes:
        raise FileNotFoundError(f"no {split} part-*.bin files under {data_root}")
    for prefix in prefixes:
        for suffix in (".bin", ".idx", ".json"):
            artifact = Path(f"{prefix}{suffix}")
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                raise FileNotFoundError(f"missing or empty indexed-data artifact: {artifact}")
    return prefixes


def validated_data_prefixes(data_root: Path) -> tuple[list[Path], list[Path]]:
    """Return artifacts only when DATA_READY declares their exact membership."""
    data_root = data_root.expanduser().resolve()
    ready_path = data_root / "DATA_READY.json"
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid indexed-data manifest: {ready_path}") from error
    if not isinstance(ready, dict):
        raise RuntimeError(f"indexed-data manifest must contain an object: {ready_path}")

    validated = []
    for split, manifest_key in (("train", "train_parts"), ("val", "valid_parts")):
        declared_values = ready.get(manifest_key)
        if not isinstance(declared_values, list) or not declared_values:
            raise RuntimeError(f"indexed-data manifest lacks nonempty {manifest_key}")
        if not all(isinstance(value, str) and value for value in declared_values):
            raise RuntimeError(f"indexed-data manifest has invalid {manifest_key}")
        declared = []
        for value in declared_values:
            prefix = Path(value).expanduser()
            if not prefix.is_absolute():
                prefix = data_root / prefix
            declared.append(prefix.resolve())
        if len(set(declared)) != len(declared):
            raise RuntimeError(f"indexed-data manifest has duplicate {manifest_key}")
        discovered = [prefix.resolve() for prefix in data_prefixes(data_root, split)]
        if set(declared) != set(discovered):
            missing = sorted(str(path) for path in set(declared) - set(discovered))
            undeclared = sorted(str(path) for path in set(discovered) - set(declared))
            raise RuntimeError(
                f"indexed-data manifest membership changed for {split}: "
                f"missing={missing}, undeclared={undeclared}"
            )
        validated.append(discovered)
    return validated[0], validated[1]
