import os
import subprocess
import sys
from pathlib import Path

import pytest

from archlab.megatron.launch_config import resolve_recipe

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes" / "launches"


def test_output_name_cannot_select_model_or_preflight():
    path = RECIPES / "qwen38_dense_quarter.yaml"
    expected = resolve_recipe(path, {})
    for name in ("/tmp/arbitrary", "/tmp/qwen38-27b-full-foo", "/tmp/compat-qwen38-flash-next-1b"):
        assert resolve_recipe(path, {"NGA_OUTPUT_ROOT": name}) == expected
    assert expected["NGA_MODEL_SCALE"] == "quarter"
    assert expected["NGA_PREFLIGHT_STEPS"] == "400"
    assert resolve_recipe(RECIPES / "qwen38_dense_full.yaml", {})["NGA_PREFLIGHT_STEPS"] == "5"


@pytest.mark.parametrize("name", ["qwen38_dense_quarter", "qwen38_dense_full", "qwen38_flash_next_full",
                                  "qwen38_flash_next_quarter", "qwen38_flash_next_1b", "qwen38_flash_next_w320_e32"])
def test_all_launch_recipes_have_explicit_geometry_and_budget(name):
    result = resolve_recipe(RECIPES / f"{name}.yaml", {})
    assert int(result["NGA_EXPECTED_NODES"]) * int(result["NGA_GPUS_PER_NODE"]) == 32
    assert int(result["NGA_TARGET_TRAIN_TOKENS"]) > 0
    assert len(result["NGA_LAUNCH_RECIPE_SHA256"]) == 64


def test_model_override_must_match_recipe_and_numbers_cannot_be_shell_code():
    path = RECIPES / "qwen38_dense_quarter.yaml"
    with pytest.raises(ValueError, match="conflicts"):
        resolve_recipe(path, {"NGA_MODEL_SCALE": "full"})
    for value in ("1; echo bad", "$(pwd)", "1\nX=1", "-1", "1.5"):
        with pytest.raises(ValueError, match="integer"):
            resolve_recipe(path, {"NGA_TARGET_TRAIN_TOKENS": value})
    assert resolve_recipe(path, {"NGA_TARGET_TRAIN_TOKENS": "1000"})["NGA_TARGET_TRAIN_TOKENS"] == "1000"


def test_numeric_bindings_do_not_trigger_shell_octal_or_overflow():
    path = RECIPES / "qwen38_dense_quarter.yaml"
    assert resolve_recipe(path, {"NGA_GPUS_PER_NODE": "08"})["NGA_GPUS_PER_NODE"] == "8"
    with pytest.raises(ValueError, match="integer range"):
        resolve_recipe(path, {"NGA_TARGET_TRAIN_TOKENS": str(2**63)})


def test_recipe_requires_explicit_probe_controls(tmp_path):
    path = tmp_path / "incomplete.yaml"
    path.write_text((RECIPES / "qwen38_dense_quarter.yaml").read_text().replace("  NGA_PROBE_STEPS: 0\n", ""))
    with pytest.raises(ValueError, match="explicitly set"):
        resolve_recipe(path, {})


@pytest.mark.parametrize("script", ["lib/dlc_runtime.sh", "run_qwen38_27b_full_dlc.sh",
                                    "run_qwen38_27b_quarter_dlc.sh", "run_qwen38_flash_next_full_dlc.sh"])
def test_launch_scripts_parse_without_execution(script):
    subprocess.run(["bash", "--noprofile", "--norc", "-n", str(ROOT / "scripts" / script)], check=True)


def test_shell_recipe_loading_propagates_failure_even_inside_condition():
    env = {key: value for key, value in os.environ.items() if not key.startswith("NGA_")}
    env.update(NGA_PYTHON=sys.executable, NGA_REPO_ROOT=str(ROOT), NGA_MODEL_SCALE="full")
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c",
         'source "$NGA_REPO_ROOT/scripts/lib/dlc_runtime.sh"; '
         'if nga_load_recipe "$NGA_REPO_ROOT/recipes/launches/qwen38_dense_quarter.yaml"; '
         'then exit 99; else exit 0; fi'], env=env, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "conflicts with recipe" in completed.stderr
