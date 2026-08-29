from __future__ import annotations

from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from archlab.capabilities import require_backend_support
from archlab.launch import get_backend
from archlab.megatron import backend as megatron_backend
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


def test_research_speedrun_plan_requires_durable_artifacts(monkeypatch, tmp_path):
    for name, value in {
        "NGA_DATA_ROOT": tmp_path / "data",
        "NGA_DATA_MANIFEST": tmp_path / "climbmix.manifest.json",
        "NGA_OUTPUT_DIR": tmp_path / "output",
    }.items():
        monkeypatch.setenv(name, str(value))
    path = tmp_path / "research.yaml"
    path.write_text(
        "schema_version: 1\n"
        "name: research-baseline\n"
        "backend: speedrun\n"
        "variant: baseline\n"
        "seed: 42\n"
        "selection:\n  size: 100m\n"
        "comparison:\n  regime: controlled\n"
        "artifacts:\n  mode: research\n"
        "paths:\n"
        "  data_root: env:NGA_DATA_ROOT\n"
        "  data_manifest: env:NGA_DATA_MANIFEST\n"
        "  output_dir: env:NGA_OUTPUT_DIR\n",
        encoding="utf-8",
    )

    plan = get_backend("speedrun").render(load_experiment(path))

    assert "--no-save-final-checkpoint" not in plan.argv
    assert "--save-final-checkpoint" in plan.argv
    assert "--artifact-policy=research" in plan.argv
    assert "--initialization-hash=shared" in plan.argv
    assert f"--data-manifest={tmp_path / 'climbmix.manifest.json'}" in plan.argv
    assert f"--metrics-path={tmp_path / 'output' / 'metrics.jsonl'}" in plan.argv
    assert f"--checkpoint-dir={tmp_path / 'output' / 'checkpoints'}" in plan.argv


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
    assert plan.argv[3:5] == ("--module", "archlab.megatron.backend")
    assert "--use-mcore-models" in plan.argv
    assert plan.metadata["world_size"] == 8
    assert plan.metadata["runtime"]["provider"] == "system"
    assert plan.metadata["runtime"]["validated_container_profile"] == "nemo-26.06"
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
    assert plan.argv[9:11] == ("--module", "archlab.megatron.backend")
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


def test_megatron_runtime_resolves_system_package(monkeypatch, tmp_path):
    package_path = tmp_path / "site-packages" / "megatron"
    package_path.mkdir(parents=True)
    pretrain_script = package_path.parent / "pretrain_gpt.py"
    pretrain_script.write_text("pass\n", encoding="utf-8")

    def fake_find_spec(name):
        spec = ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(package_path)]
        return spec

    monkeypatch.setattr(megatron_backend, "find_spec", fake_find_spec)
    versions = {
        "megatron-core": "0.13.0",
        "torch": "2.9.1",
        "transformer-engine": "2.11.0",
        "nemo-toolkit": "2.5.0",
    }
    monkeypatch.setattr(megatron_backend, "version", versions.__getitem__)
    monkeypatch.setattr(megatron_backend.platform, "python_version", lambda: "3.12.0")

    runtime = megatron_backend.validate_runtime()
    assert runtime == {
        "provider": "system",
        "distribution": "megatron-core",
        "version": "0.13.0",
        "package_path": str(package_path),
        "validated_container_profile": "nemo-26.06",
        "python": "3.12.0",
        "torch": "2.9.1",
        "transformer_engine": "2.11.0",
        "nemo_toolkit": "2.5.0",
        "pretrain_script": str(pretrain_script),
    }


def test_generic_megatron_renderer_rejects_unenforced_research_policy(monkeypatch, tmp_path):
    for name, value in {
        "NGA_TRAIN_DATA": tmp_path / "train_text_document",
        "NGA_VALID_DATA": tmp_path / "valid_text_document",
        "NGA_DATA_CACHE": tmp_path / "cache",
        "NGA_TOKENIZER": tmp_path / "tokenizer",
        "NGA_OUTPUT_DIR": tmp_path / "output",
        "NGA_DATA_MANIFEST": tmp_path / "manifest.json",
    }.items():
        monkeypatch.setenv(name, str(value))
    spec = load_experiment(ROOT / "recipes/experiments/megatron_baseline_1b_seed42.yaml")
    spec.config["comparison"] = {"regime": "scaling"}
    spec.config["artifacts"] = {"mode": "research"}
    spec.config["paths"]["data_manifest"] = "env:NGA_DATA_MANIFEST"
    with pytest.raises(SpecError, match="cannot yet enforce"):
        get_backend("megatron").render(spec)


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


def test_research_spec_requires_complete_comparison_and_artifact_contract(tmp_path):
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text(
        "schema_version: 1\nname: incomplete\nbackend: speedrun\n"
        "variant: baseline\nseed: 42\ncomparison:\n  regime: controlled\n",
        encoding="utf-8",
    )
    with pytest.raises(SpecError, match="both comparison and artifacts"):
        load_experiment(incomplete)

    complete = tmp_path / "complete.yaml"
    complete.write_text(
        "schema_version: 1\nname: complete\nbackend: speedrun\n"
        "variant: baseline\nseed: 42\ncomparison:\n  regime: controlled\n"
        "artifacts:\n  mode: research\n",
        encoding="utf-8",
    )
    spec = load_experiment(complete)
    assert spec.comparison is not None
    assert spec.comparison.regime.value == "controlled"
    assert spec.artifacts is not None
    assert spec.artifacts.save_final_checkpoint is True


def test_prompt_set_is_versioned_and_stable():
    prompts = load_prompts()
    assert len(prompts) == 7
    assert prompts[0].id == "capital-france"
    assert prompts[-1].text.endswith("x is")
