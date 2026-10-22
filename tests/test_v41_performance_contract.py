import torch

from archlab.architectures.engram_scaling import scaled_engram_geometry
from archlab.optimizers.sinkhorn import sinkhorn_direction


def test_engram_scales_buckets_and_aligns_channels_at_d128():
    base = dict(
        hidden_size=5120,
        engram_head_dim=256,
        engram_vocab_size=16_000_000,
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        engram_n_heads=8,
    )
    actual = scaled_engram_geometry(base, 128)
    assert actual["engram_head_dim"] == 8
    assert actual["engram_vocab_size"] == 400_000
    assert all(9_600_000 < rows < 9_620_000 for rows in actual["engram_num_embeddings"])
    assert actual["engram_num_embeddings"][0] != actual["engram_num_embeddings"][1]


def test_factored_sinkhorn_matches_paper_dense_algorithm_with_masked_rows():
    torch.manual_seed(17)
    update = torch.randn(137, 6)
    update[:3] = 0
    update[3] *= 1e-6
    update[:, -1] = 0
    expected = update.clone()
    rho = expected.norm(dim=1)
    expected[rho <= 1e-3 * rho.mean()] = 0
    for k in range(11):
        expected = expected / (expected.norm(dim=1 if k % 2 == 0 else 0, keepdim=True) + 1e-20)
    expected *= 6**0.5
    actual = sinkhorn_direction(update)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(actual[:4]) == 0
    assert torch.count_nonzero(actual[:, -1]) == 0
    torch.testing.assert_close(
        sinkhorn_direction(torch.zeros_like(update)), torch.zeros_like(update)
    )


def test_batched_expert_optimizer_matches_individual_updates_and_resume():
    from archlab.optimizers.sharded_adafactor import ShardedAdafactor

    torch.manual_seed(91)
    reference = [
        torch.nn.Parameter(torch.randn(*shape)) for shape in [(7, 11, 13), (19, 13), (13,)]
    ]
    actual = [torch.nn.Parameter(p.detach().clone()) for p in reference]
    left = ShardedAdafactor(reference, lr=0.001, stochastic_rounding=False, chunk_elements=143)
    right = ShardedAdafactor(
        actual, lr=0.001, stochastic_rounding=False, chunk_elements=143, batched_updates=True
    )
    for _ in range(4):
        for p, q in zip(reference, actual, strict=True):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        left.step()
        right.step()
        for p, q in zip(reference, actual, strict=True):
            torch.testing.assert_close(p, q, atol=3e-7, rtol=3e-6)
        right.load_state_dict(right.state_dict())
    assert (
        left.last_metrics["updated_parameter_tensors"]
        == right.last_metrics["updated_parameter_tensors"]
        == 3
    )
    assert right.last_metrics["changed_local_elements"] > 0


def test_performance_contract_is_explicit_and_rejects_partial_options(tmp_path):
    import json
    from pathlib import Path

    import pytest

    from archlab.automodel.deepseek_v41_performance import read_performance_contract

    source = (
        Path(__file__).resolve().parents[1] / "recipes/deepseek_v41/performance/performance-v1.json"
    )
    contract = read_performance_contract(source)
    assert contract["table_lr_scale"] * 0.01 == 0.00026
    assert contract["scale_engram"] is True
    incomplete = dict(contract)
    del incomplete["expert_dispatcher"]
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="unrecognized"):
        read_performance_contract(path)


def test_fixed_engram_budget_is_independent_of_width_and_legacy_is_unchanged():
    base = dict(hidden_size=5120, engram_head_dim=256, engram_vocab_size=16_000_000,
                engram_layer_ids=[1, 14], engram_max_ngram_size=4, engram_n_heads=8)
    anchor = scaled_engram_geometry(base, 128)
    for width in (128, 256, 384, 640, 1280):
        assert scaled_engram_geometry(base, width, anchor_width=128) == anchor
    assert scaled_engram_geometry(base, 640)["engram_head_dim"] == 32
    assert scaled_engram_geometry(base, 640)["engram_vocab_size"] == 2_000_000
    assert base["engram_head_dim"] == 256
