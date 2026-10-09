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


def test_trim_preserves_targets_and_document_rows():
    import numpy as np

    from archlab.automodel.deepseek_v41_scratch_data import ScratchData

    data = object.__new__(ScratchData)
    data.sequence, data.contract = 2048, {"pad_id": 0}
    windows = [(np.arange(18), 17), (np.arange(250), 249), (np.array([2, 2]), 0)]
    data.window = lambda index: windows[index]
    full = data.batch([0, 1, 2], device="cpu")
    trimmed = data.batch([0, 1, 2], device="cpu", trim_alignment=128)
    assert trimmed[0].shape == (3, 256)
    assert full[2] == trimmed[2] == 266
    torch.testing.assert_close(full[1][:, :256], trimmed[1])
    mask = trimmed[1] != -100
    torch.testing.assert_close(full[0][:, :256][mask], trimmed[0][mask])
    assert (full[1][:, 256:] == -100).all()


def test_sinkhorn_full_checkpoint_retains_fp32_momentum_and_next_update(tmp_path, monkeypatch):
    import torch.distributed as dist

    from archlab.automodel.deepseek_v41_full_checkpoint import (
        restore_full_checkpoint,
        save_full_checkpoint,
    )
    from archlab.optimizers.sinkhorn import EngramSinkhornAdafactor

    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", torch.set_rng_state)
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1
    )
    try:
        model = torch.nn.Module()
        for name in ("a", "b"):
            block = torch.nn.Module()
            block.engram = torch.nn.Module()
            block.engram.embed = torch.nn.Embedding(19, 6, dtype=torch.bfloat16)
            model.add_module(name, block)
        model.norm = torch.nn.Parameter(torch.ones(6))
        optimizer = EngramSinkhornAdafactor(
            model.named_parameters(), lr=0.001, table_lr_scale=0.026, batched_updates=True
        )

        def step():
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            optimizer.step()

        torch.manual_seed(22)
        step()
        contract, cursor = {"performance": "fixture"}, {"step": 1}
        save_full_checkpoint(tmp_path / "checkpoint", model, optimizer, cursor, contract)
        step()
        expected = {name: p.detach().clone() for name, p in model.named_parameters()}
        restore_full_checkpoint(tmp_path / "checkpoint", model, optimizer, contract)
        assert optimizer.state[model.a.engram.embed.weight]["momentum"].dtype == torch.float32
        assert optimizer.param_groups[0]["table_lr_scale"] == 0.026
        step()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, expected[name], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_controlled_batch_change_preserves_the_global_window_sequence():
    from archlab.automodel.deepseek_v41_scratch_training import build_batches

    class Reader:
        def batch(self, indices, **kwargs):
            return indices

    for microbatch in (4, 16):
        batches = [
            build_batches(Reader(), 137, rank, microbatch=microbatch, accumulation=2, world_size=16)
            for rank in range(16)
        ]
        windows = [index for rank in batches for batch in rank for index in batch]
        assert sorted(windows) == list(range(137, 137 + 2 * 16 * microbatch))


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


def test_replicated_gradient_batching_preserves_views_and_dtype_groups(monkeypatch):
    import torch.distributed as dist

    from archlab.automodel.deepseek_v41_full_training import reduce_replicated_gradients

    gradients = [
        torch.arange(15, dtype=dtype).reshape(3, 5).T
        for dtype in (torch.float32, torch.bfloat16, torch.float32)
    ]
    expected = [x.clone() * 3 for x in gradients]
    calls = []

    def reduce(value):
        calls.append(value.dtype)
        value.mul_(3)

    monkeypatch.setattr(dist, "all_reduce", reduce)
    reduce_replicated_gradients(gradients)
    assert len(calls) == 2
    for actual, reference in zip(gradients, expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    reduce_replicated_gradients([])
    assert len(calls) == 2


def test_fixed_engram_budget_is_independent_of_width_and_legacy_is_unchanged():
    base = dict(hidden_size=5120, engram_head_dim=256, engram_vocab_size=16_000_000,
                engram_layer_ids=[1, 14], engram_max_ngram_size=4, engram_n_heads=8)
    anchor = scaled_engram_geometry(base, 128)
    for width in (128, 256, 384, 640, 1280):
        assert scaled_engram_geometry(base, width, anchor_width=128) == anchor
    assert scaled_engram_geometry(base, 640)["engram_head_dim"] == 32
    assert scaled_engram_geometry(base, 640)["engram_vocab_size"] == 2_000_000
    assert base["engram_head_dim"] == 256


def test_fixed_engram_contract_and_evaluation_preserve_explicit_anchor(tmp_path):
    import json
    from pathlib import Path

    import pytest

    from archlab.automodel.deepseek_v41_performance import read_performance_contract
    from archlab.automodel.deepseek_v41_scratch_evaluate import construction_options

    recipe = Path(__file__).resolve().parents[1] / "recipes/deepseek_v41/performance"
    performance = read_performance_contract(recipe / "performance-fixed-engram-v2.json")
    saved = dict(runtime={"geometry": {"text_config": {"hidden_size": 1280}}},
                 scaling_study=True, performance=performance)
    options = construction_options(json.loads(json.dumps(saved)))
    assert options["width"] == 1280 and options["engram_anchor_width"] == 128
    for anchor in (None, True, 0, 640):
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps({**performance, "engram_anchor_width": anchor}))
        with pytest.raises(ValueError, match="d128 lookup"):
            read_performance_contract(path)
    path.write_text(json.dumps({**performance, "format": "archlab-v41-performance-v1"}))
    with pytest.raises(ValueError, match="unrecognized"):
        read_performance_contract(path)
