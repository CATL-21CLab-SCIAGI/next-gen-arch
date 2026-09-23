"""PLE adapter numerical parity and native checkpoint metadata compatibility.

Metadata checks require real Megatron Core. They do not claim multi-rank
checkpoint save/load parity; the single-owner numerical oracle is CPU-only.
"""

from dataclasses import replace

import pytest
import torch

from archlab.architectures.qwen38_flash_next_full import (
    DistributedPLE as ArchitecturePLE,
    OwnerShardedPLEEmbedding as ArchitectureEmbedding,
    Qwen38FlashNextFullConfig,
)
from archlab.megatron.ple_checkpoint import DistributedPLE, OwnerShardedPLEEmbedding


def test_checkpoint_adapter_preserves_initialization_state_forward_and_gradients():
    config = replace(Qwen38FlashNextFullConfig.tiny(), ngram_partitions=1)

    def construct(cls):
        torch.manual_seed(71)
        module = cls(config, owner_rank=0, owner_world_size=1)
        module.embedding.reset_parameters()
        # Exercise the convolution and its input norm, not just its zero-init path.
        torch.nn.init.normal_(module.conv.weight, std=.01)
        return module, torch.get_rng_state().clone()

    base, base_rng = construct(ArchitecturePLE)
    adapted, adapted_rng = construct(DistributedPLE)
    assert torch.equal(base_rng, adapted_rng)
    assert isinstance(adapted.embedding, OwnerShardedPLEEmbedding)
    assert not hasattr(base, "sharded_state_dict")
    assert not hasattr(base.embedding, "sharded_state_dict")
    assert list(base.state_dict()) == list(adapted.state_dict())
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, adapted.state_dict()[name], rtol=0, atol=0)

    tokens = torch.randint(0, config.vocab_size, (2, config.sequence_len))
    x = torch.randn(config.sequence_len, 2, config.residual_streams * config.hidden_size)
    x_base, x_adapted = x.clone().requires_grad_(), x.clone().requires_grad_()
    expected, actual = base(tokens, x_base), adapted(tokens, x_adapted)
    cotangent = torch.randn_like(actual)
    expected.backward(cotangent)
    actual.backward(cotangent)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x_adapted.grad, x_base.grad, rtol=0, atol=0)
    for (name, parameter), (adapted_name, adapted_parameter) in zip(
        base.named_parameters(), adapted.named_parameters(), strict=True
    ):
        assert name == adapted_name
        for attribute in ("allreduce", "expert_parallel", "is_embedding_or_output_parameter",
                          "archlab_optimizer", "archlab_no_weight_decay"):
            assert getattr(parameter, attribute, None) == getattr(adapted_parameter, attribute, None)
        torch.testing.assert_close(adapted_parameter.grad, parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize("owners", [1, 2, 8])
@pytest.mark.parametrize("replica", [0, 3])
@pytest.mark.parametrize("offsets", [(), ((0, 1, 3),), ((0, 1, 3), (1, 0, 2))])
def test_native_embedding_keys_offsets_and_replica_ids(owners, replica, offsets):
    pytest.importorskip("megatron.core.dist_checkpointing.mapping")
    config = Qwen38FlashNextFullConfig.tiny()
    prefix = "decoder.layers.7.ple.embedding."
    elements = config.ngram_rows_per_partition * config.ngram_branch_dim
    for owner in range(owners):
        module = OwnerShardedPLEEmbedding(config, owner_rank=owner,
                                         owner_world_size=owners, replica_rank=replica)
        state = module.sharded_state_dict(prefix, offsets, {"ignored": True})
        assert list(state) == [f"{prefix}tables.{slot}" for slot in range(len(module.tables))]
        for slot, partition in enumerate(range(owner, config.ngram_partitions, owners)):
            shard = state[f"{prefix}tables.{slot}"]
            assert shard.key == f"{prefix}tables.weight"
            assert shard.data is module.tables[slot]
            assert shard.local_shape == (elements,)
            assert shard.global_shape == tuple(fragmentation for _, _, fragmentation in offsets) + (
                elements * config.ngram_partitions,)
            assert shard.global_offset == tuple(rank for _, rank, _ in offsets) + (partition * elements,)
            assert shard.axis_fragmentations == tuple(f for _, _, f in offsets) + (config.ngram_partitions,)
            assert shard.prepend_axis_num == len(offsets)
            assert shard.replica_id == (0, 0, replica)


def test_native_ple_checkpoint_recurses_into_owner_tables(monkeypatch):
    utils = pytest.importorskip("megatron.core.transformer.utils")
    config = Qwen38FlashNextFullConfig.tiny()
    module = DistributedPLE(config, owner_rank=1, owner_world_size=2, replica_rank=3)
    offsets = ((0, 1, 2),)
    metadata = {"probe": "preserve-by-identity"}
    delegated = []

    def checkpoint_child(child, prefix, sharded_offsets, received_metadata):
        delegated.append((child, prefix, sharded_offsets, received_metadata))
        if isinstance(child, ArchitectureEmbedding):
            return child.sharded_state_dict(prefix, sharded_offsets, received_metadata)
        return {prefix + "sentinel": child}

    # Check the recursion protocol independently of global Megatron groups;
    # owner-table descriptors still use the real native ShardedTensor.
    monkeypatch.setattr(utils, "sharded_state_dict_default", checkpoint_child)
    state = module.sharded_state_dict("ple.", offsets, metadata)
    children = list(module.named_children())
    assert len(delegated) == len(children)
    for (child, prefix, received_offsets, received_metadata), (name, expected) in zip(
        delegated, children, strict=True
    ):
        assert child is expected
        assert prefix == f"ple.{name}."
        assert received_offsets is offsets
        assert received_metadata is metadata
    tables = {name: value for name, value in state.items() if name.startswith("ple.embedding.")}
    assert len(tables) == len(module.embedding.tables)
    assert all(value.key == "ple.embedding.tables.weight" for value in tables.values())
    assert all(value.replica_id == (0, 0, 3) for value in tables.values())
