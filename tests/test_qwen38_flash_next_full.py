from dataclasses import replace

import torch
from torch.distributed.nn import functional as dist_nn_functional

from archlab.architectures.qwen38_flash_next_full import (
    DistributedPLE,
    FourStreamGatedResidual,
    GatedDeltaNet,
    OwnerShardedPLEEmbedding,
    PLEHash,
    Qwen38FlashNextFullConfig,
    build_hash_multipliers,
    parameter_count_contract,
    ple_partition_ownership,
)


def test_full_geometry_and_exact_parameter_contract():
    config = Qwen38FlashNextFullConfig()
    counts = parameter_count_contract(config)

    assert config.num_hidden_layers == 48
    assert config.hidden_size == 2_560
    assert config.pipeline_layers == (12, 13, 13, 10)
    assert config.attention_heads == 24
    assert config.attention_kv_heads == 2
    assert config.attention_head_dim == 256
    assert config.linear_qk_heads == 16
    assert config.linear_v_heads == 48
    assert config.linear_key_dim == config.linear_value_dim == 128
    assert config.num_experts == 512
    assert config.num_experts_per_token == 10
    assert config.residual_streams == 4
    assert config.residual_low_rank == 320
    assert config.mtp_num_layers == 3
    assert config.mtp_use_repeated_layer is True
    assert counts == {
        "embeddings_and_head": 1_271_398_400,
        "ple_tables": 51_200_245_760,
        "ple_projection": 32_839_680,
        "backbone": 124_231_001_984,
        "shared_mtp_inner": 2_576_663_040,
        "native_mtp_wrapper": 13_114_880,
        "total": 179_325_263_744,
    }


def test_quarter_geometry_retains_depth_and_removes_mtp_exactly():
    config = Qwen38FlashNextFullConfig.quarter_depth48_no_mtp()
    counts = parameter_count_contract(config)

    assert config.num_hidden_layers == 48
    assert config.hidden_size == 640
    assert config.pipeline_layers == (12, 12, 12, 12)
    assert config.attention_heads == 6
    assert config.attention_kv_heads == 1
    assert config.attention_head_dim == 64
    assert config.linear_qk_heads == 4
    assert config.linear_v_heads == 12
    assert config.linear_key_dim == config.linear_value_dim == 32
    assert config.num_experts == 128
    assert config.num_experts_per_token == 3
    assert config.residual_streams == 1
    assert config.residual_low_rank == 80
    assert config.ngram_partitions == 32
    assert config.mtp_num_layers == 0
    assert config.mtp_use_repeated_layer is False
    assert config.mtp_loss_scaling_factor == 0
    assert counts == {
        "embeddings_and_head": 317_849_600,
        "ple_tables": 3_200_040_960,
        "ple_projection": 823_680,
        "backbone": 1_956_170_336,
        "shared_mtp_inner": 0,
        "native_mtp_wrapper": 0,
        "total": 5_474_884_576,
    }


def test_ple_prime_sizes_padding_and_hash_multipliers_match_pinned_source():
    config = Qwen38FlashNextFullConfig()

    assert config.ngram_head_vocab_sizes == (
        20_000_003,
        20_000_023,
        20_000_033,
        20_000_047,
        20_000_059,
        20_000_063,
        20_000_069,
        20_000_077,
        20_000_081,
        20_000_093,
        20_000_107,
        20_000_147,
        20_000_153,
        20_000_159,
        20_000_161,
        20_000_171,
    )
    assert config.ngram_total_rows == 320_001_446
    assert config.ngram_padded_rows == 320_001_536
    assert config.ngram_rows_per_partition == 2_500_012
    assert build_hash_multipliers(config.vocab_size, config.ngram_size).tolist() == [
        23_703_573_157_769,
        20_109_073_645_365,
        8_052_911_324_071,
    ]


def test_ple_hash_never_crosses_eos_segment_boundaries():
    hashing = PLEHash(Qwen38FlashNextFullConfig.tiny())
    first = torch.tensor([[1, 2, 63, 4, 5, 6]])
    second = torch.tensor([[11, 12, 63, 4, 5, 6]])

    assert torch.equal(hashing(first)[:, 3:], hashing(second)[:, 3:])


def test_owner_sharding_is_balanced_and_local_lookup_preserves_global_order():
    config = Qwen38FlashNextFullConfig.tiny()
    ownership = ple_partition_ownership(config.ngram_partitions, 8)

    assert ownership == tuple((rank,) for rank in range(8))
    embedding = OwnerShardedPLEEmbedding(config, owner_rank=0, owner_world_size=1)
    for partition, table in zip(embedding.global_partitions, embedding.tables, strict=True):
        values = torch.arange(table.numel(), dtype=table.dtype)
        table.data.copy_(values + partition * 100_000)
        assert table.ndim == 1
        assert table.archlab_optimizer == "adam"
        assert table.archlab_no_weight_decay is True
        assert table.allreduce is False

    ids = torch.tensor([[0, config.ngram_rows_per_partition + 2, config.ngram_padded_rows - 1]])
    output = embedding(ids)
    expected = []
    for value in ids.flatten().tolist():
        partition, row = divmod(value, config.ngram_rows_per_partition)
        start = row * config.ngram_branch_dim
        expected.append(
            torch.arange(start, start + config.ngram_branch_dim, dtype=output.dtype)
            + partition * 100_000
        )
    assert torch.equal(output.flatten(0, 1), torch.stack(expected))


def test_owner_sharded_remote_return_keeps_the_autograd_graph(monkeypatch):
    config = replace(Qwen38FlashNextFullConfig.tiny(), ngram_partitions=2)
    embedding = OwnerShardedPLEEmbedding(config, owner_rank=0, owner_world_size=2)
    embedding.reset_parameters()

    def copy_all_to_all(output, inputs, **_kwargs):
        output.copy_(inputs)

    def differentiable_all_to_all(_output, inputs, **_kwargs):
        return inputs.clone()

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", copy_all_to_all)
    monkeypatch.setattr(
        dist_nn_functional,
        "all_to_all_single",
        differentiable_all_to_all,
    )
    output = embedding(torch.tensor([0, 1]))
    assert output.requires_grad
    output.sum().backward()
    assert embedding.tables[0].grad is not None
    assert embedding.tables[0].grad.abs().sum() > 0


def test_four_stream_gr_matches_explicit_official_equations_and_backpropagates():
    torch.manual_seed(3)
    config = Qwen38FlashNextFullConfig.tiny()
    residual = FourStreamGatedResidual(config)
    packed = torch.randn(2, 3, 4 * config.hidden_size, requires_grad=True)

    mixed, original, injection = residual(packed)
    normalized = residual.norm(packed)
    latent = torch.nn.functional.silu(
        residual.input_mix_weight_down(normalized) / config.residual_streams
    )
    weights = (
        residual.input_mix_weight_up(latent)
        .sigmoid()
        .unflatten(-1, (config.residual_streams, config.hidden_size))
    )
    expected = (
        weights * normalized.unflatten(-1, (config.residual_streams, config.hidden_size))
    ).mean(-2)
    expected_injection = 2 * torch.sigmoid(
        residual.block_inject_weight(normalized) / config.residual_streams
    )

    assert original is packed
    assert torch.allclose(mixed, expected)
    assert torch.allclose(injection, expected_injection)
    FourStreamGatedResidual.inject(original, mixed, injection).sum().backward()
    assert residual.input_mix_weight_down.weight.grad.abs().sum() > 0


def test_gdn_has_separate_excluded_gates_and_cpu_oracle_is_differentiable():
    torch.manual_seed(7)
    config = Qwen38FlashNextFullConfig.tiny()
    gdn = GatedDeltaNet(config)
    hidden = torch.randn(6, 2, config.hidden_size, requires_grad=True)

    output = gdn(hidden)

    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    assert gdn.in_proj_b.weight is not gdn.in_proj_a.weight
    assert torch.equal(gdn.dt_bias, torch.ones_like(gdn.dt_bias))
    assert torch.all((gdn.A_log.exp() >= 0.01) & (gdn.A_log.exp() <= 16))
    assert gdn.in_proj_z.weight.archlab_optimizer == "adamw"
    assert gdn.in_proj_b.weight.archlab_optimizer == "adamw"
    assert gdn.in_proj_a.weight.archlab_optimizer == "adamw"
    output.square().mean().backward()
    assert gdn.in_proj_qkv.weight.grad.abs().sum() > 0
    assert gdn.in_proj_b.weight.grad.abs().sum() > 0
    assert gdn.in_proj_a.weight.grad.abs().sum() > 0


def test_tiny_distributed_ple_single_owner_forward_backward():
    torch.manual_seed(11)
    config = replace(Qwen38FlashNextFullConfig.tiny(), ngram_partitions=1)
    ple = DistributedPLE(config, owner_rank=0, owner_world_size=1)
    ple.embedding.reset_parameters()
    assert torch.count_nonzero(ple.conv.weight) == 0
    tokens = torch.randint(0, config.vocab_size, (2, config.sequence_len))
    packed = torch.randn(
        config.sequence_len,
        2,
        config.residual_streams * config.hidden_size,
        requires_grad=True,
    )

    output = ple(tokens, packed)
    output.square().mean().backward()

    assert output.shape == packed.shape
    assert torch.isfinite(output).all()
    assert packed.grad is not None and packed.grad.abs().sum() > 0
    assert ple.embedding.tables[0].grad is not None
