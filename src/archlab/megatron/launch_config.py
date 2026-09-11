"""Resolve explicit DLC recipe defaults without inspecting an output path.

This module only renders configuration; it never launches a process. Machine
paths remain launch-time environment overrides owned by the shell entrypoint.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

import yaml

from archlab.artifacts import sha256_file

KEYS = frozenset({
    "NGA_EXPECTED_NODES", "NGA_GPUS_PER_NODE", "NGA_SEQUENCE_LENGTH",
    "NGA_MICRO_BATCH_SIZE", "NGA_GLOBAL_BATCH_SIZE", "NGA_TARGET_TRAIN_TOKENS",
    "NGA_CHECKPOINT_INTERVAL_TOKENS", "NGA_RUNTIME_PREFLIGHT", "NGA_PREFLIGHT_STEPS",
    "NGA_PROBE_STEPS", "NGA_PROBE_SAVE_INTERVAL", "NGA_MODEL_SCALE",
    "NGA_FLASH_NEXT_MODEL_VARIANT",
})
MODEL_KEYS = {"dense-27b": "NGA_MODEL_SCALE", "flash-next": "NGA_FLASH_NEXT_MODEL_VARIANT"}
MODELS = {"dense-27b": {"quarter", "full"}, "flash-next": {
    "full", "quarter-depth48-no-mtp", "1b-depth48-no-mtp", "w320-e32-depth48-no-mtp"}}


def resolve_recipe(path: Path, environment: Mapping[str, str]) -> dict[str, str]:
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or set(recipe) != {"schema_version", "family", "environment"}:
        raise ValueError("launch recipe requires exactly schema_version, family and environment")
    if (type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1
            or not isinstance(recipe["family"], str) or recipe["family"] not in MODEL_KEYS):
        raise ValueError("unsupported launch recipe schema or family")
    values = recipe["environment"]
    if not isinstance(values, dict) or set(values) - KEYS:
        raise ValueError("recipe contains unsupported environment keys")
    family = recipe["family"]
    model_key = MODEL_KEYS[family]
    if values.get(model_key) not in MODELS[family]:
        raise ValueError("recipe must explicitly select a supported model")
    if MODEL_KEYS[{"dense-27b": "flash-next", "flash-next": "dense-27b"}[family]] in values:
        raise ValueError("recipe mixes unrelated model families")
    if environment.get(model_key) not in (None, "", values[model_key]):
        raise ValueError(f"{model_key} conflicts with recipe; select the matching recipe explicitly")
    result = {}
    for key, default in values.items():
        value = environment.get(key) or str(default)
        if key != model_key and (not value.isascii() or not value.isdecimal()):
            raise ValueError(f"{key} must be an unsigned decimal integer")
        if key != model_key:
            if int(value) > 2**63 - 1:
                raise ValueError(f"{key} exceeds the shell's signed integer range")
            value = str(int(value))  # Bash arithmetic must not interpret leading zeros as octal.
        result[key] = value
    required = {"NGA_EXPECTED_NODES", "NGA_GPUS_PER_NODE", "NGA_SEQUENCE_LENGTH",
                "NGA_MICRO_BATCH_SIZE", "NGA_GLOBAL_BATCH_SIZE", "NGA_TARGET_TRAIN_TOKENS",
                "NGA_CHECKPOINT_INTERVAL_TOKENS"}
    if not required <= result.keys() or any(int(result[k]) < 1 for k in required):
        raise ValueError("recipe needs positive topology, batch and token-budget controls")
    controls = {"NGA_PROBE_STEPS"} | ({"NGA_RUNTIME_PREFLIGHT", "NGA_PREFLIGHT_STEPS"}
                                      if family == "dense-27b" else {"NGA_PROBE_SAVE_INTERVAL"})
    if not controls <= result.keys():
        raise ValueError("recipe must explicitly set all preflight/probe controls")
    if result.get("NGA_RUNTIME_PREFLIGHT", "0") not in {"0", "1"}:
        raise ValueError("runtime preflight must be 0 or 1")
    if result.get("NGA_RUNTIME_PREFLIGHT") == "1" and int(result.get("NGA_PREFLIGHT_STEPS", "0")) < 1:
        raise ValueError("enabled preflight requires positive steps")
    result["NGA_LAUNCH_FAMILY"] = family
    result["NGA_LAUNCH_RECIPE_SHA256"] = sha256_file(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--bindings", action="store_true")
    args = parser.parse_args()
    values = resolve_recipe(args.recipe, os.environ)
    if args.bindings:
        # No shell code/eval. Keys are allowlisted; values cannot contain LF.
        print("\n".join(f"{key}={value}" for key, value in sorted(values.items())))
    else:
        print(json.dumps(values, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
