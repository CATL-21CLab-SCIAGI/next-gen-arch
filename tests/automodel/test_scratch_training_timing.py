"""ETA estimates count actual targets and exclude cold qualification updates."""

import pytest


@pytest.mark.parametrize("sequence", [128, 256, 512, 1024, 1536, 2048])
@pytest.mark.parametrize("ratio", [4, 128])
def test_trim_uses_exactly_the_same_valid_indexer_samples(sequence, ratio):
    import torch

    from archlab.automodel.deepseek_v41_full_indexer import sampled_query_ids

    original = sampled_query_ids(2048, ratio, 64, "cpu", sample_context=2048, fixed_cpu_grid=True)
    trimmed = sampled_query_ids(sequence, ratio, 64, "cpu", sample_context=2048, fixed_cpu_grid=True)
    torch.testing.assert_close(trimmed, original[original < sequence], rtol=0, atol=0)
