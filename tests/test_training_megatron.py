"""The native model is preserved across the Megatron provider boundary."""
# ruff: noqa: E402
from copy import deepcopy
from importlib import import_module

import pytest
import torch

pytest.importorskip("megatron")

import_module("archlab.training.megatron")

from megatron.bridge import AutoBridge
from test_training_limite import SNAPSHOT, policy
from test_training_limite import native as native

from archlab.megatron.limite import NativeLimiteBridge
from archlab.training.limite import LimiteConfig, export_model


def provider_for(model, tmp_path):
    root = tmp_path / "native-export"
    export_model(model, SNAPSHOT, root)
    config = LimiteConfig.from_pretrained(root)
    provider = AutoBridge.from_hf_config(config).to_megatron_provider(load_weights=False)
    provider.apply_overrides_and_finalize(dtype=torch.bfloat16)
    return provider


def test_native_provider_keeps_exact_weights_dtypes_and_forward(tmp_path, native):
    expected = policy(native).eval()
    provider = provider_for(expected, tmp_path)
    actual = provider.provide().eval()
    actual.policy.policy.load_state_dict(expected.state_dict())
    actual.policy.to(dtype=torch.bfloat16)
    actual.train().eval()  # Rebuild publisher inference constants after loading.
    for name, parameter in actual.policy.policy.named_parameters():
        want = dict(expected.named_parameters())[name]
        assert parameter.dtype == want.dtype
        torch.testing.assert_close(parameter, want, rtol=0, atol=0, msg=name)
    ids = torch.tensor([[1, 4, 8, 5, 2]])
    torch.testing.assert_close(actual(ids), expected(input_ids=ids, use_cache=False).logits,
                               rtol=0, atol=0)
    assert actual.shared_embedding_or_output_weight() is actual.policy.get_output_embeddings().weight


@pytest.mark.parametrize("field,value", [
    ("tensor_model_parallel_size", 2), ("pipeline_model_parallel_size", 2),
    ("context_parallel_size", 2), ("expert_model_parallel_size", 2),
    ("sequence_parallel", True), ("fp8", "e4m3"),
])
def test_provider_rejects_unimplemented_execution_contract(tmp_path, native, field, value):
    provider = deepcopy(provider_for(policy(native), tmp_path))
    setattr(provider, field, value)
    with pytest.raises(ValueError, match="native Limite"):
        provider.finalize()


def test_bridge_maps_all_native_parameters_without_geometry_conversion(tmp_path, native):
    expected = policy(native)
    provider = provider_for(expected, tmp_path)
    registry = NativeLimiteBridge().mapping_registry()
    actual = provider.provide()
    for name, _ in actual.named_parameters():
        mapping = registry.megatron_to_hf_lookup(name)
        expected_name = name.removeprefix("policy.")
        if expected_name == "policy.model.base.embed_tokens.weight":
            expected_name = "policy.lm_head.weight"
        assert mapping.hf_param == expected_name
    assert NativeLimiteBridge.megatron_to_hf_config(provider) == provider.native_config


def test_provider_rejects_optimizer_with_broken_mixed_dtype_checkpointing(tmp_path, native):
    from megatron.core.distributed import DistributedDataParallelConfig

    provider = provider_for(policy(native), tmp_path)
    with pytest.raises(ValueError, match="use_distributed_optimizer=True"):
        provider.provide_distributed_model(ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=False))


def test_package_dtype_partition_preserves_standard_optimizer_overrides(monkeypatch):
    from megatron.core.optimizer import OptimizerConfig, get_standard_config_overrides
    from verl.utils.megatron import optimizer as factory

    model = torch.nn.Module()
    model.register_parameter("matrix", torch.nn.Parameter(torch.zeros(3, 4, dtype=torch.bfloat16)))
    model.register_parameter("gain", torch.nn.Parameter(torch.ones(3, dtype=torch.float32)))
    config = OptimizerConfig(lr=2e-5, min_lr=1e-6, weight_decay=.1)
    captured = {}
    monkeypatch.setattr(factory, "get_megatron_optimizer_native",
                        lambda **kwargs: captured.update(kwargs))
    factory.get_megatron_optimizer([model], config)
    overrides = captured["config_overrides"]
    standard = get_standard_config_overrides(config)
    for key, value in standard.items():
        assert overrides[key] == value
    partitions = {key: value for key, value in overrides.items() if key not in standard}
    assert len(partitions) == 2
    for name, parameter in model.named_parameters():
        matched = [value for key, value in partitions.items() if key.matches(parameter, name)]
        assert matched == [{"_storage_dtype_partition": str(parameter.dtype)}]
    assert captured["config"] is config
