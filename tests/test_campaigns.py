import torch

from next_gen_arch.training.campaign_compare import (
    DEFAULT_REFERENCE,
    load_speedrun_reference,
    summarize,
    validate_results,
)
from next_gen_arch.training.campaigns import (
    TEN_M_VARIANTS,
    ten_m_model_config_kwargs,
    verify_ten_m_contract,
)
from next_gen_arch.training.models import build_model_config, instantiate_model


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
