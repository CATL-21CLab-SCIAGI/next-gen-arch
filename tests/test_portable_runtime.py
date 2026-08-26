from __future__ import annotations

import ast
from pathlib import Path

import pytest

from archlab.capabilities import require_backend_support
from archlab.launch import get_backend
from archlab.megatron.backend import MEGATRON_COMMIT, validate_submodule
from archlab.prompts import load_prompts
from archlab.spec import SpecError, load_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_speedrun_spec_composes_and_preserves_frozen_command(monkeypatch, tmp_path):
    monkeypatch.setenv("NGA_DATA_ROOT", str(tmp_path / "data"))
    spec = load_experiment(ROOT / "recipes/experiments/speedrun_qwen_gdn_100m_seed42.yaml")
    plan = get_backend(spec.backend).render(spec)

    assert spec.backend == "speedrun"
    assert spec.variant == "qwen-gdn"
    assert plan.argv[:3] == ("python", "-m", "archlab.speedrun.base_train")
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
    spec = load_experiment(ROOT / "recipes/experiments/megatron_baseline_1b_seed42.yaml")
    plan = get_backend(spec.backend).render(spec)

    assert plan.argv[:3] == ("torchrun", "--standalone", "--nproc-per-node=8")
    assert "--use-mcore-models" in plan.argv
    assert plan.metadata["world_size"] == 8
    assert plan.metadata["submodule"]["commit"] == MEGATRON_COMMIT
    assert plan.metadata["effective_training_tokens"] <= 12_000_000_000


@pytest.mark.parametrize(
    ("config_name", "model_parallel", "data_parallel", "expert_parallel", "microbatches"),
    (
        ("megatron_native_dense_100m_dp_seed42.yaml", 1, 12, 1, 15),
        ("megatron_native_dense_100m_tp2_seed42.yaml", 2, 6, 1, 30),
        ("megatron_native_dense_100m_pp2_seed42.yaml", 2, 6, 1, 30),
        ("megatron_native_dense_100m_cp2_seed42.yaml", 2, 6, 1, 30),
        ("megatron_native_moe_100m_ep1_seed42.yaml", 1, 12, 1, 15),
        ("megatron_native_moe_100m_ep6_seed42.yaml", 1, 12, 6, 15),
    ),
)
def test_native_parallelism_campaign_renders_frozen_12_rank_contract(
    monkeypatch,
    tmp_path,
    config_name,
    model_parallel,
    data_parallel,
    expert_parallel,
    microbatches,
):
    for name, value in {
        "NGA_TRAIN_DATA": tmp_path / "train_text_document",
        "NGA_VALID_DATA": tmp_path / "valid_text_document",
        "NGA_DATA_CACHE": tmp_path / "cache",
        "NGA_TOKENIZER": tmp_path / "tokenizer",
        "NGA_OUTPUT_DIR": tmp_path / "output",
    }.items():
        monkeypatch.setenv(name, str(value))

    spec = load_experiment(ROOT / "recipes" / "experiments" / config_name)
    plan = get_backend(spec.backend).render(spec)

    assert plan.argv[:9] == (
        "torchrun",
        "--nnodes=2",
        "--nproc-per-node=6",
        "--node-rank",
        "env:NODE_RANK",
        "--master-addr",
        "env:MASTER_ADDR",
        "--master-port",
        "env:MASTER_PORT",
    )
    assert plan.metadata["world_size"] == 12
    assert plan.metadata["model_parallel_size"] == model_parallel
    assert plan.metadata["data_parallel_size"] == data_parallel
    assert plan.metadata["expert_parallel_size"] == expert_parallel
    assert plan.metadata["expert_data_parallel_size"] == data_parallel // expert_parallel
    assert plan.metadata["num_microbatches"] == microbatches
    assert plan.metadata["effective_training_tokens"] == 73_728_000
    assert "--save" not in plan.argv
    assert "--tensorboard-dir" not in plan.argv
    assert plan.env == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    }
    assert '--node-rank "${NODE_RANK}"' in plan.shell()

    if "tp2" in config_name:
        assert "--sequence-parallel" in plan.argv
    if "cp2" in config_name:
        assert plan.argv[plan.argv.index("--cp-comm-type") + 1] == "p2p"
    if "moe" in config_name:
        assert plan.argv[plan.argv.index("--num-experts") + 1] == "6"
        assert plan.argv[plan.argv.index("--moe-ffn-hidden-size") + 1] == "768"
        assert "--moe-grouped-gemm" in plan.argv
        assert "--moe-permute-fusion" in plan.argv


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
    architecture_files = {
        path.name for path in (ROOT / "src/archlab/architectures").glob("*.py")
    }
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
    assert (ROOT / "src/archlab/speedrun/base_train.py").is_file()
    assert (ROOT / "src/archlab/optimizers/speedrun.py").is_file()
    assert (ROOT / "src/archlab/megatron/backend.py").is_file()
    assert (ROOT / "src/archlab/megatron/train.py").is_file()
    assert (ROOT / "src/archlab/speedrun/dataset.py").is_file()
    assert not (ROOT / "nanochat/gpt.py").exists()
    assert not (ROOT / "scripts/base_train.py").exists()


def test_architecture_modules_do_not_import_execution_layers():
    forbidden = ("archlab.speedrun", "archlab.backends", "archlab.megatron")
    for source in (ROOT / "src/archlab/architectures").glob("*.py"):
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
