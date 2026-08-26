import csv
import json
from pathlib import Path

import pytest
import torch

from next_gen_arch.training.campaign_compare import (
    DEFAULT_REFERENCE,
    RunResult,
    campaign_provenance,
    cross_backend_metrics,
    load_megatron_results,
    load_speedrun_reference,
    merge_recovery_results,
    overlay_results,
    summarize,
    validate_results,
)
from next_gen_arch.training.campaigns import (
    TEN_M_VARIANTS,
    campaign_model_config_kwargs,
    get_campaign_variant,
    ten_m_model_config_kwargs,
    verify_ten_m_contract,
)
from next_gen_arch.training.models import build_model_config, instantiate_model
from next_gen_arch.training.optimization_recipes import OPTIMIZATION_RECIPES

PUBLISHED_RUNS = DEFAULT_REFERENCE.with_name("backend-10m-runs.csv")
PUBLISHED_COMPARISON = DEFAULT_REFERENCE.with_name("backend-10m-comparison.json")
OPTIMIZATION_RUNS = (
    Path(__file__).resolve().parents[1] / "results" / "megatron-10m-optimization-runs-b300.csv"
)
SAFE_AUTOTUNE_RESULTS = (
    Path(__file__).resolve().parents[1] / "results" / "megatron-10m-safe-autotune-b300"
)
MULTINODE_100M_RESULTS = (
    Path(__file__).resolve().parents[1] / "results" / "100m-multinode-b300"
)


def test_ten_m_grid_is_complete_and_parameter_counts_are_frozen():
    assert verify_ten_m_contract() == {"variants": 16, "seeds": 3, "runs": 48}
    for variant in TEN_M_VARIANTS:
        config = build_model_config(**ten_m_model_config_kwargs(variant))
        with torch.device("meta"):
            model = instantiate_model(config)
        assert model.num_scaling_params()["total"] == variant.parameter_count


def test_100m_baseline_geometry_and_parameter_count_are_frozen():
    variant = get_campaign_variant("100m", "baseline")
    config = build_model_config(**campaign_model_config_kwargs("100m", variant))
    with torch.device("meta"):
        model = instantiate_model(config)

    assert config.n_layer == 10
    assert config.n_embd == 384
    assert config.n_head == 6
    assert variant.steps == 3_228
    assert model.num_scaling_params()["total"] == 105_775_510


def test_speedrun_10m_reference_matches_frozen_contract():
    rows = validate_results(load_speedrun_reference(DEFAULT_REFERENCE), allow_partial=False)
    assert len(rows) == 48
    summaries = {row.variant: row for row in summarize(rows)}
    assert abs(summaries["baseline"].mean_bpb - 1.5549804898262447) < 1e-12
    assert abs(summaries["kimi-k3-kda-update"].paired_delta_bpb + 0.09338652662580766) < 1e-12


def test_correction_overlay_replaces_only_matching_run():
    primary = RunResult("megatron", "dsa", 42, 1, 2, 3, 1.5, 10.0, 2.0)
    corrected = RunResult("megatron", "dsa", 42, 1, 2, 3, 1.7, 9.0, 3.0, "fixed", "upstream")

    assert overlay_results([primary], [corrected]) == [corrected]


def test_recovery_results_fill_missing_keys_without_replacing_rows() -> None:
    primary = RunResult("megatron", "baseline", 42, 1, 2, 3, 1.5, 10.0, 2.0)
    recovery = RunResult("megatron", "kda", 42, 1, 2, 3, 1.4, 9.0, 2.0)

    assert merge_recovery_results([primary], [recovery]) == [primary, recovery]
    with pytest.raises(ValueError, match="replace existing rows"):
        merge_recovery_results([primary], [primary])


def test_summary_prefers_steady_state_step_throughput_when_available() -> None:
    baseline = RunResult(
        "megatron",
        "baseline",
        42,
        1,
        2,
        3,
        1.5,
        10.0,
        2.0,
        steady_state_tokens_per_second=20.0,
    )
    variant = RunResult(
        "megatron",
        "dsa",
        42,
        1,
        2,
        3,
        1.4,
        5.0,
        2.0,
        steady_state_tokens_per_second=20.0,
    )

    summaries = {row.variant: row for row in summarize([baseline, variant])}

    assert summaries["dsa"].normalized_throughput == 1.0
    assert summaries["dsa"].mean_tokens_per_second == 20.0
    assert summaries["dsa"].mean_cold_inclusive_tokens_per_second == 5.0
    assert summaries["dsa"].throughput_basis == "median steady-state step"


def test_megatron_loader_keeps_optimization_provenance(tmp_path):
    run = tmp_path / "baseline-seed42"
    run.mkdir()
    (run / "result.json").write_text(
        json.dumps(
            {
                "backend": "megatron",
                "mode": "full",
                "variant": {"name": "baseline"},
                "seed": 42,
                "parameter_count": 9_363_488,
                "training_steps": 286,
                "training_tokens": 112_459_776,
                "final_bpb": 1.5,
                "tokens_per_second": 1_000_000,
                "wall_seconds": 120,
                "source_commit": "source",
                "megatron_commit": "upstream",
                "backend_profile": {"name": "compile-max-autotune"},
                "optimization_recipe": {"name": "z-loss-5e-6-clip01"},
                "source_dirty": True,
                "source_diff_sha256": "digest",
                "source_worktree_sha256": "worktree-digest",
            }
        ),
        encoding="utf-8",
    )

    [result] = load_megatron_results(tmp_path)

    assert result.backend_profile == "compile-max-autotune"
    assert result.optimization_recipe == "z-loss-5e-6-clip01"
    assert result.source_dirty is True
    assert result.source_diff_sha256 == "digest"
    assert result.source_worktree_sha256 == "worktree-digest"
    assert campaign_provenance([result])["megatron"] == {
        "backend_profiles": ["compile-max-autotune"],
        "optimization_recipes": ["z-loss-5e-6-clip01"],
        "source_commits": ["source"],
        "megatron_commits": ["upstream"],
        "source_dirty_values": [True],
        "source_diff_sha256": ["digest"],
        "source_worktree_sha256": ["worktree-digest"],
    }


def test_published_backend_comparison_is_complete_and_provenanced():
    rows = validate_results(load_speedrun_reference(PUBLISHED_RUNS), allow_partial=False)
    assert len(rows) == 96

    megatron_rows = [row for row in rows if row.backend == "megatron"]
    assert len(megatron_rows) == 48
    assert {row.source_commit for row in megatron_rows if row.variant == "dsa"} == {
        "ed8336e5403d8da75082502a96a115f06ee17334"
    }
    assert {row.source_commit for row in megatron_rows if row.variant != "dsa"} == {
        "e6d9b0b1153e74078dbb87d4c0e8b12c8d4df513"
    }
    assert {row.megatron_commit for row in megatron_rows} == {
        "55ac7082517c3878ae653c07c09c534b8aed49f6"
    }

    metrics = cross_backend_metrics(summarize(rows))
    assert abs(metrics["delta_pearson"] - 0.9759478282715947) < 1e-12
    assert metrics["delta_sign_agreement_count"] == 12
    assert abs(metrics["baseline_absolute_throughput_ratio"] - 0.5529545577133159) < 1e-12

    payload = json.loads(PUBLISHED_COMPARISON.read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 96
    assert len(payload["summary"]) == 32
    assert payload["cross_backend"] == metrics


def test_optimization_ledger_covers_every_executable_recipe() -> None:
    with OPTIMIZATION_RUNS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 93
    assert {row["optimization_recipe"] for row in rows if row["optimization_recipe"]} == set(
        OPTIMIZATION_RECIPES
    )
    assert {row["accelerator"] for row in rows} == {"NVIDIA B300"}
    assert {row["compute_capability"] for row in rows} == {"10.3"}


def test_published_safe_autotune_comparison_is_complete_and_provenanced() -> None:
    rows = validate_results(
        load_speedrun_reference(SAFE_AUTOTUNE_RESULTS / "runs.csv"), allow_partial=False
    )
    assert len(rows) == 96

    megatron_rows = [row for row in rows if row.backend == "megatron"]
    assert len(megatron_rows) == 48
    assert sum(row.backend_profile == "compile-max-autotune" for row in megatron_rows) == 39
    assert sum(row.backend_profile == "compile" for row in megatron_rows) == 9
    assert {row.source_commit for row in megatron_rows} == {
        "f8bc91df1aa10a2e4fd193cb9acdc4df3cdba975"
    }
    assert {row.source_dirty for row in megatron_rows} == {False}
    assert {row.source_worktree_sha256 for row in megatron_rows} == {
        "33c2da2f369cc6a117055a71d327f41de48fe448131a6a62c936189bf2d3e4c3"
    }
    assert {row.megatron_commit for row in megatron_rows} == {
        "55ac7082517c3878ae653c07c09c534b8aed49f6"
    }

    metrics = cross_backend_metrics(summarize(rows))
    assert abs(metrics["delta_pearson"] - 0.9713608264480493) < 1e-12
    assert metrics["delta_sign_agreement_count"] == 13
    assert abs(metrics["mean_absolute_delta_gap"] - 0.009241204591297257) < 1e-12

    campaign = json.loads((SAFE_AUTOTUNE_RESULTS / "campaign.json").read_text())
    assert campaign["hardware"]["accelerator"] == "NVIDIA B300"
    assert campaign["hardware"]["compute_capability"] == "10.3"
    assert campaign["contract"]["accepted_megatron_runs"] == 48
    assert campaign["contract"]["primary_fail_fast_runs"] == 5
    assert campaign["contract"]["recovery_finite_runs"] == 9


def test_published_100m_multinode_result_matches_frozen_contract() -> None:
    campaign = json.loads((MULTINODE_100M_RESULTS / "campaign.json").read_text())
    comparison = json.loads((MULTINODE_100M_RESULTS / "comparison.json").read_text())
    with (MULTINODE_100M_RESULTS / "learning_curve.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        curve = list(csv.DictReader(handle))

    assert campaign["status"] == "complete"
    assert campaign["hardware"] == {
        "accelerator": "NVIDIA B300",
        "nodes": 3,
        "gpus_per_node": 5,
        "world_size": 15,
        "intra_node_transport": "NVLink",
        "inter_node_transport": "RoCE with GPU Direct RDMA",
        "physical_gpu_allowlist_per_node": [0, 2, 4, 5, 7],
    }
    assert campaign["source"]["repository_commit"] == (
        "ebb2fc2c8d74d8dd13256231b372b139a91abfbd"
    )
    assert campaign["source"]["megatron_submodule_commit"] == (
        "55ac7082517c3878ae653c07c09c534b8aed49f6"
    )
    assert campaign["contract"]["effective_global_batch_sequences"] == 192
    assert campaign["contract"]["active_rows_by_rank"] == [13] * 12 + [12] * 3
    assert campaign["collective_probe"]["status"] == "pass"
    assert campaign["collective_probe"]["all_reduce_rank_sum"] == 105

    assert len(curve) == 14
    assert curve[0]["step"] == "0"
    assert curve[0]["megatron_bpb"] == ""
    assert curve[-1]["step"] == "3228"
    assert float(curve[-1]["current_speedrun_bpb"]) == pytest.approx(
        comparison["historical_reproduction"]["current_speedrun_final_bpb"]
    )
    assert float(curve[-1]["megatron_bpb"]) == pytest.approx(
        comparison["megatron_vs_current_speedrun"]["megatron_final_bpb"]
    )
    assert comparison["historical_reproduction"]["gate"]["status"] == "pass"
    assert comparison["historical_reproduction"]["pearson"] >= 0.999
