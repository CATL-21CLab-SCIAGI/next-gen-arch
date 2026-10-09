"""ETA estimates count actual targets and exclude cold qualification updates."""

import pytest

from archlab.automodel.deepseek_v41_scratch_training import (
    learning_rate,
    summarize_training_timings,
    training_budgets,
)


def test_historical_prefix_keeps_lr_schedule_but_has_five_shorter_milestones():
    matched = {"training": {"supervised_tokens": 10_000_000_000}}
    stop, schedule = training_budgets(matched, 1_000_000_000)
    assert stop == 1_000_000_000
    assert schedule == 10_000_000_000
    assert stop // 5 == 200_000_000
    assert learning_rate(1000, 500_000_000, budget=schedule) == learning_rate(1000, 500_000_000)
    assert learning_rate(1000, 500_000_000, budget=stop) != learning_rate(1000, 500_000_000)
    assert training_budgets(None) == (10_000_000_000, 10_000_000_000)
    assert training_budgets(None, cell={"unique_targets": 10, "epochs": 4}) == (40, 40)


@pytest.mark.parametrize("prefix", [0, -5, 3, True, 1.0, 10_000_000_005])
def test_historical_prefix_rejects_invalid_budgets(prefix):
    with pytest.raises(ValueError):
        training_budgets({"training": {"supervised_tokens": 10_000_000_000}}, prefix)


def test_historical_prefix_requires_explicit_matched_geometry():
    with pytest.raises(ValueError):
        training_budgets(None, 1_000_000_000)


def test_throughput_excludes_warmup_and_weights_variable_target_counts():
    records = [
        dict(wall_seconds=1000, supervised_tokens=100),
        dict(wall_seconds=100, supervised_tokens=100),
        dict(wall_seconds=2, supervised_tokens=100),
        dict(wall_seconds=4, supervised_tokens=500),
    ]
    result = summarize_training_timings(records, warmup_updates=2)
    assert result["valid_tokens_per_second"] == 100
    assert result["median_step_seconds"] == 3
    assert result["projected_10B_training_seconds"] == 100_000_000
    assert result["measured_updates"] == 2


@pytest.mark.parametrize("records,warmup", [
    ([], 0), ([dict(wall_seconds=1, supervised_tokens=1)], 1),
    ([dict(wall_seconds=0, supervised_tokens=1)], 0),
    ([dict(wall_seconds=1, supervised_tokens=0)], 0),
    ([dict(wall_seconds=1, supervised_tokens=1)], -1),
])
def test_no_eta_from_missing_or_invalid_measurements(records, warmup):
    with pytest.raises(ValueError):
        summarize_training_timings(records, warmup_updates=warmup)


@pytest.mark.parametrize("sequence", [128, 256, 512, 1024, 1536, 2048])
@pytest.mark.parametrize("ratio", [4, 128])
def test_trim_uses_exactly_the_same_valid_indexer_samples(sequence, ratio):
    import torch

    from archlab.automodel.deepseek_v41_full_indexer import sampled_query_ids

    original = sampled_query_ids(2048, ratio, 64, "cpu", sample_context=2048, fixed_cpu_grid=True)
    trimmed = sampled_query_ids(sequence, ratio, 64, "cpu", sample_context=2048, fixed_cpu_grid=True)
    torch.testing.assert_close(trimmed, original[original < sequence], rtol=0, atol=0)
