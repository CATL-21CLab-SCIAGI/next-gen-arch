from __future__ import annotations

import ast
from pathlib import Path

import pytest

from next_gen_arch.architectures import require_backend_support
from next_gen_arch.backends import get_backend
from next_gen_arch.backends.megatron import MEGATRON_COMMIT, validate_submodule
from next_gen_arch.prompts import load_prompts
from next_gen_arch.spec import SpecError, load_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_speedrun_spec_composes_and_preserves_frozen_command(monkeypatch, tmp_path):
    monkeypatch.setenv("NGA_DATA_ROOT", str(tmp_path / "data"))
    spec = load_experiment(ROOT / "configs/experiments/speedrun_qwen_gdn_100m_seed42.yaml")
    plan = get_backend(spec.backend).render(spec)

    assert spec.backend == "speedrun"
    assert spec.variant == "qwen-gdn"
    assert plan.argv[:3] == ("python", "-m", "next_gen_arch.training.base_train")
    assert "--frontier-variant=qwen_gdn" in plan.argv
    assert plan.env["NANOCHAT_BASE_DIR"] == str((tmp_path / "data").resolve())
    assert plan.metadata["parameter_count"] == 110_617_902
    assert len(spec.source_files) == 4


def test_megatron_spec_renders_eight_rank_scaling_plan(monkeypatch, tmp_path):
    for name, value in {
        "NGA_TRAIN_DATA": tmp_path / "train_text_document",
        "NGA_VALID_DATA": tmp_path / "valid_text_document",
        "NGA_DATA_CACHE": tmp_path / "cache",
        "NGA_TOKENIZER": tmp_path / "tokenizer",
        "NGA_OUTPUT_DIR": tmp_path / "output",
    }.items():
        monkeypatch.setenv(name, str(value))
    spec = load_experiment(ROOT / "configs/experiments/megatron_baseline_1b_seed42.yaml")
    plan = get_backend(spec.backend).render(spec)

    assert plan.argv[:3] == ("torchrun", "--standalone", "--nproc-per-node=8")
    assert "--use-mcore-models" in plan.argv
    assert plan.metadata["world_size"] == 8
    assert plan.metadata["submodule"]["commit"] == MEGATRON_COMMIT
    assert plan.metadata["effective_training_tokens"] <= 12_000_000_000


def test_megatron_submodule_is_pinned_and_clean():
    assert validate_submodule()["commit"] == MEGATRON_COMMIT


def test_backend_capabilities_do_not_claim_unported_equivalence():
    require_backend_support("qwen-gdn", "speedrun")
    with pytest.raises(ValueError, match="no validated megatron adapter"):
        require_backend_support("qwen-gdn", "megatron")


def test_committed_specs_reject_absolute_paths(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "schema_version: 1\nname: invalid\nbackend: speedrun\n"
        "variant: baseline\nseed: 42\npaths:\n  data_root: /private/data\n",
        encoding="utf-8",
    )
    with pytest.raises(SpecError, match="must be portable"):
        load_experiment(path)


def test_prompt_set_is_versioned_and_stable():
    prompts = load_prompts()
    assert len(prompts) == 7
    assert prompts[0].id == "capital-france"
    assert prompts[-1].text.endswith("x is")


def test_python_sources_use_one_src_package_layout():
    architecture_files = {path.name for path in (ROOT / "src/next_gen_arch/arch").glob("*.py")}
    assert architecture_files == {
        "__init__.py",
        "base.py",
        "combinations.py",
        "dsa.py",
        "fog.py",
        "frontier.py",
        "kimi.py",
        "sota.py",
    }
    assert (ROOT / "src/next_gen_arch/training/base_train.py").is_file()
    assert (ROOT / "src/next_gen_arch/training/optim.py").is_file()
    assert (ROOT / "src/next_gen_arch/training/dataset.py").is_file()
    assert not (ROOT / "nanochat/gpt.py").exists()
    assert not (ROOT / "scripts/base_train.py").exists()


def test_architecture_modules_do_not_import_execution_layers():
    forbidden = ("next_gen_arch.training", "next_gen_arch.backends")
    for source in (ROOT / "src/next_gen_arch/arch").glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(name.startswith(forbidden) for name in imports), source
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "setup_optimizer"
            for node in ast.walk(tree)
        ), source
