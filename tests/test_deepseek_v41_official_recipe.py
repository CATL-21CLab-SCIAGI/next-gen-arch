import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from archlab.artifacts import sha256_file
from archlab.automodel.deepseek_v41_official_execution import UPSTREAM_COMMIT
from archlab.automodel.deepseek_v41_official_recipe import OfficialV41Config, admit_mesh, resolve_path
from archlab.automodel.deepseek_v41_official_training import REPLAY_PROTOCOL


def _config():
    root = Path(__file__).parents[1]
    path = root / "recipes/experiments/deepseek_v41_simplicial_math_1b_official.yaml"
    return OfficialV41Config(**yaml.safe_load(path.read_text()))


def test_official_migration_preserves_the_existing_experiment_settings():
    root = Path(__file__).parents[1]
    existing = yaml.safe_load((root / "recipes/experiments/deepseek_v41_simplicial_math_1b.yaml").read_text())
    config = _config()
    for key in ("world_size", "ep_size", "context", "supervised_tokens", "validation_tokens",
                "validation_interval_tokens", "checkpoint_interval_tokens", "warmup_steps",
                "query_chunk", "assets", "weights", "train_data", "validation_data"):
        assert getattr(config, key) == existing[key]


@pytest.mark.parametrize("change", [
    {"world_size": 8}, {"ep_size": 32}, {"context": 2048},
    {"supervised_tokens": 1000}, {"warmup_steps": 0}, {"query_chunk": -1},
    {"validation_tokens": True},
])
def test_config_rejects_changed_topology_budget_or_invalid_intervals(change):
    with pytest.raises(ValueError):
        replace(_config(), **change)


def test_config_rejects_a_different_forward_or_checkpoint_precision():
    config = _config()
    for key, value in (("upstream_commit", "stale"), ("linear_compute", "float32"),
                       ("kv_index_quantization", "disabled"), ("dense_fsdp", 8)):
        with pytest.raises(ValueError, match="migration contract"):
            replace(config, backend={**config.backend, key: value})


def test_recipe_paths_are_launch_injected(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHLAB_FIXTURE_PATH", str(tmp_path))
    assert resolve_path("env:ARCHLAB_FIXTURE_PATH") == tmp_path.resolve()
    monkeypatch.delenv("ARCHLAB_FIXTURE_PATH")
    for path in ("env:ARCHLAB_FIXTURE_PATH", str(tmp_path)):
        with pytest.raises(ValueError, match="populate"):
            resolve_path(path)


@pytest.fixture
def mesh_receipts(tmp_path):
    hashes = {"automodel/deepseek_v41_official_training.py": "fixture-source-sha256"}
    replicas = {"replicated_fp32_masters": True, "world_gradient_agreement": True,
                "world_parameter_agreement": True, "gradient_sha256": "same-gradient",
                "adapter_sha256": "same-adapter", "gradients": [{"name": "fixture", "nonzero": True}]}
    for rank in range(32):
        group = list(range(rank // 8 * 8, (rank // 8 + 1) * 8))
        receipt = {
            "passed": True, "rank": rank, "world_size": 32, "ep_size": 8,
            "expert_fsdp_size": 4, "engram_owners": 32, "tiny": True,
            "context": 128, "automodel_commit": UPSTREAM_COMMIT,
            "adapter": {"backend": "deterministic"},
            "implementation_sha256": hashes, "unique_rank_windows": True,
            "mesh": {"ep_ranks": group, "ep_hosts": [f"node-{rank // 8}"] * 8,
                     "expert_fsdp_ranks": list(range(rank % 8, 32, 8)),
                     "engrams": [{"owner_ranks": list(range(32))}]},
            "updates": [{"loss": 2.0, **replicas}, {"loss": 1.9, **replicas}],
            "checkpoint_replay": {"passed": True, "exact_state_restoration": True,
                                  "protocol": REPLAY_PROTOCOL["version"],
                                  "protocol_config": REPLAY_PROTOCOL, **replicas},
            "frozen_base": {"unchanged": True, "no_gradients": True},
            "container_image": "fixture-container", "packages": {"torch": "fixture"},
            "cuda": "fixture", "nccl": [2, 27, 5],
            "reproducibility": {"deterministic_algorithms": False, "cudnn_deterministic": True,
                                "cudnn_benchmark": False, "fill_uninitialized_memory": False,
                                "cublas_workspace_config": ":4096:8"},
        }
        (tmp_path / f"rank{rank}.json").write_text(json.dumps(receipt))
    (tmp_path / "COMPLETE.json").write_text(json.dumps({
        "passed": True, "world_size": 32,
        "rank_receipts": [f"rank{rank}.json" for rank in range(32)],
    }))
    return tmp_path, hashes


def test_mesh_admission_returns_all_rank_content_hashes(mesh_receipts):
    path, hashes = mesh_receipts
    assert admit_mesh(path, hashes) == [sha256_file(path / f"rank{rank}.json") for rank in range(32)]


@pytest.mark.parametrize("change", [
    {"passed": False}, {"rank": 2}, {"automodel_commit": "stale"},
    {"implementation_sha256": {"changed": "source"}},
    {"world_size": 8}, {"ep_size": 4}, {"expert_fsdp_size": 1}, {"engram_owners": 8},
    {"checkpoint_replay": {"passed": False}},
    {"frozen_base": {"unchanged": False, "no_gradients": True}},
    {"unique_rank_windows": False},
    {"adapter": {"backend": "triton"}},
])
def test_mesh_admission_requires_every_rank_to_have_passed_the_actual_contract(mesh_receipts, change):
    path, hashes = mesh_receipts
    target = path / "rank17.json"
    receipt = json.loads(target.read_text())
    target.write_text(json.dumps({**receipt, **change}))
    with pytest.raises(ValueError):
        admit_mesh(path, hashes)


@pytest.mark.parametrize("marker", [{"passed": False, "world_size": 32},
                                    {"passed": True, "world_size": 8}])
def test_mesh_admission_rejects_failed_or_smaller_collective_run(mesh_receipts, marker):
    path, hashes = mesh_receipts
    (path / "COMPLETE.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="actual 32-rank"):
        admit_mesh(path, hashes)
