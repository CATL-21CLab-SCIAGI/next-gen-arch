import json

import torch

from next_gen_arch.training.campaign_compare import (
    DEFAULT_REFERENCE,
    RunResult,
    cross_backend_metrics,
    load_speedrun_reference,
    overlay_results,
    summarize,
    validate_results,
)
from next_gen_arch.training.campaigns import (
    TEN_M_VARIANTS,
    ten_m_model_config_kwargs,
    verify_ten_m_contract,
)
from next_gen_arch.training.models import build_model_config, instantiate_model

PUBLISHED_RUNS = DEFAULT_REFERENCE.with_name("backend-10m-runs.csv")
PUBLISHED_COMPARISON = DEFAULT_REFERENCE.with_name("backend-10m-comparison.json")


def test_ten_m_grid_is_complete_and_parameter_counts_are_frozen():
    assert verify_ten_m_contract() == {"variants": 16, "seeds": 3, "runs": 48}
    for variant in TEN_M_VARIANTS:
        config = build_model_config(**ten_m_model_config_kwargs(variant))
        with torch.device("meta"):
            model = instantiate_model(config)
        assert model.num_scaling_params()["total"] == variant.parameter_count


def test_speedrun_10m_reference_matches_frozen_contract():
    rows = validate_results(load_speedrun_reference(DEFAULT_REFERENCE), allow_partial=False)
    assert len(rows) == 48
    summaries = {row.variant: row for row in summarize(rows)}
    assert abs(summaries["baseline"].mean_bpb - 1.5549804898262447) < 1e-12
    assert abs(summaries["kimi-k3-kda-update"].paired_delta_bpb + 0.09338652662580766) < 1e-12


def test_correction_overlay_replaces_only_matching_run():
    primary = RunResult("megatron", "dsa", 42, 1, 2, 3, 1.5, 10.0, 2.0)
    corrected = RunResult(
        "megatron", "dsa", 42, 1, 2, 3, 1.7, 9.0, 3.0, "fixed", "upstream"
    )

    assert overlay_results([primary], [corrected]) == [corrected]


def test_published_backend_comparison_is_complete_and_provenanced():
    rows = validate_results(load_speedrun_reference(PUBLISHED_RUNS), allow_partial=False)
    assert len(rows) == 96

    megatron_rows = [row for row in rows if row.backend == "megatron"]
    assert len(megatron_rows) == 48
    assert {
        row.source_commit for row in megatron_rows if row.variant == "dsa"
    } == {"ed8336e5403d8da75082502a96a115f06ee17334"}
    assert {
        row.source_commit for row in megatron_rows if row.variant != "dsa"
    } == {"e6d9b0b1153e74078dbb87d4c0e8b12c8d4df513"}
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
