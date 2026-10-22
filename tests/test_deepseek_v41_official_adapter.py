"""Run official-backbone tests with the pinned AutoModel checkout on PYTHONPATH."""

import copy
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
from archlab.automodel.deepseek_v41_official_adapter import (
    ADAPTER_MARKER,
    PRODUCTION_LAYER_INDICES,
    install_official_adapters,
)


def _small_config():
    return V41AdapterConfig(
        width=16, query_heads=2, kv_heads=1, head_dim=16, short_window=2, long_window=4
    )


class _Connection(nn.Module):
    """An expansion oracle that makes its contribution distinguishable."""

    def __init__(self):
        super().__init__()
        self.streams = 4
        self.weight = nn.Parameter(torch.tensor(3.0))

    @staticmethod
    def expand(output, residual, mix):
        return residual + output.unsqueeze(-2) * mix


def _skeleton(config=V41AdapterConfig(), count=40):  # noqa: B008 -- immutable configuration
    model = nn.Module()
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=config.width, hc_mult=config.streams)
    )
    model.model = nn.Module()
    model.model.layers = nn.ModuleDict()
    for index in range(count):
        layer = nn.Module()
        layer.attn_hc, layer.ffn_hc = _Connection(), _Connection()
        model.model.layers[str(index)] = layer
    return model


def test_production_layer_selection_master_precision_and_budget():
    model = _skeleton()
    original = dict(model.named_parameters())
    adapters = install_official_adapters(model, device="meta")
    assert tuple(adapters) == PRODUCTION_LAYER_INDICES
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 167_816_256
    assert all(p.dtype == torch.float32 for a in adapters.values() for p in a.parameters())
    after = dict(model.named_parameters())
    assert all(after[name] is value and not value.requires_grad for name, value in original.items())
    assert all((ADAPTER_MARKER in name) == p.requires_grad for name, p in after.items())


def test_expansion_wrapper_keeps_upstream_result_and_deepcopy_ownership():
    config = _small_config()
    model = _skeleton(config, count=1)
    adapter = install_official_adapters(model, config, layer_indices=(0,), backend="reference")[0]
    connection = model.model.layers["0"].attn_hc
    x = torch.randn(1, 3, 4, 16)
    output, mix = torch.randn(1, 3, 16), torch.tensor(0.7)
    expected = _Connection.expand(output, x, mix)
    torch.testing.assert_close(connection.expand(output, x, mix), expected, atol=0, rtol=0)
    with torch.no_grad():
        adapter.output.weight.normal_(std=0.1)
    torch.testing.assert_close(connection.expand(output, x, mix), adapter(expected), atol=0, rtol=0)
    cloned = copy.deepcopy(model)
    cloned_connection = cloned.model.layers["0"].attn_hc
    assert cloned_connection.expand.__self__ is cloned_connection
    with torch.no_grad():
        cloned_connection.simplicial_adapter.output.weight.zero_()
    torch.testing.assert_close(cloned_connection.expand(output, x, mix), expected, atol=0, rtol=0)
    assert not torch.equal(connection.expand(output, x, mix), expected)
    connection.adapter_enabled = False
    torch.testing.assert_close(connection.expand(output, x, mix), expected, atol=0, rtol=0)


def _official_model():
    if importlib.util.find_spec("nemo_automodel") is None:
        pytest.skip("requires pinned official AutoModel checkout")
    try:
        from nemo_automodel.components.models.deepseek_v41.config import (
            DeepseekV41Config,
            DeepseekV41TextConfig,
            DeepseekV41VisionConfig,
        )
    except ModuleNotFoundError as error:
        if error.name.startswith("nemo_automodel.components.models.deepseek_v41"):
            pytest.skip("requires official AutoModel V4.1 implementation")
        raise
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM

    config = DeepseekV41Config(
        vision_config=DeepseekV41VisionConfig(num_hidden_layers=0),
        text_config=DeepseekV41TextConfig(
            vocab_size=64,
            hidden_size=16,
            moe_intermediate_size=16,
            num_hidden_layers=6,
            num_attention_heads=2,
            head_dim=8,
            qk_rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=8,
            o_groups=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            compress_ratios=[0, 0, 2, 2, 1, 1],
            kv_source_layer_ids=[2, 4],
            index_source_layer_ids=[2, 4, 5],
            index_n_heads=2,
            index_head_dim=8,
            index_topk=2,
            candidate_source_layer_id=4,
            candidate_topk_blocks=2,
            candidate_block_size=2,
            engram_layer_ids=[],
            engram_num_embeddings=[],
            num_nextn_predict_layers=0,
            dspark_block_size=0,
            dspark_noise_token_id=0,
            dtype="float32",
        ),
    )
    backend = BackendConfig(
        attn="eager",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch_mm",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(config, backend=backend)
    model.to_empty(device="cpu")
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model


def test_official_identity_insertion_order_gradients_and_checkpoint_resume():
    torch.manual_seed(67)
    model = _official_model()
    tokens = torch.randint(0, 64, (1, 12))
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        before = model(tokens, output_hidden_states=True)
    random_state = torch.get_rng_state().clone()
    adapters = install_official_adapters(
        model, _small_config(), layer_indices=(0, 4), backend="reference"
    )
    assert torch.equal(random_state, torch.get_rng_state())
    after = model(tokens, output_hidden_states=True)
    torch.testing.assert_close(after.logits, before.logits, atol=0, rtol=0)
    for left, right in zip(after.hidden_states, before.hidden_states, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)

    optimizer = torch.optim.AdamW([p for a in adapters.values() for p in a.parameters()], lr=0.01)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(tokens, labels=tokens).loss.backward()
        for name, parameter in model.named_parameters():
            if ADAPTER_MARKER not in name:
                assert parameter.grad is None, name
            else:
                assert parameter.grad is not None and parameter.grad.isfinite().all(), name
                assert bool(parameter.grad.count_nonzero()) == (
                    step > 0 or name.endswith("output.weight")
                ), name
        optimizer.step()
    current = dict(model.named_parameters())
    for name, value in original.items():
        torch.testing.assert_close(current[name], value, atol=0, rtol=0)

    # The FFN predictor must receive the actual adapter output, while attention's
    # carried pre-mix is still produced exactly once by its existing predictor.
    layer = model.model.layers["4"]
    observed = {}
    layer.attn_hc.identity_observer = lambda previous, adapted: observed.update(
        previous=previous, adapted=adapted
    )
    handle = layer.ffn_hc.register_forward_pre_hook(
        lambda module, args: observed.update(ffn_input=args[0])
    )
    model(tokens)
    handle.remove()
    layer.attn_hc.identity_observer = None
    assert observed["ffn_input"] is observed["adapted"]
    assert not torch.equal(observed["previous"], observed["adapted"])

    restored = _official_model()
    restored_adapters = install_official_adapters(
        restored, _small_config(), layer_indices=(0, 4), backend="reference"
    )
    restored.load_state_dict(model.state_dict(), strict=True)
    restored_optimizer = torch.optim.AdamW(
        [p for a in restored_adapters.values() for p in a.parameters()], lr=0.01
    )
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for candidate, optim in ((model, optimizer), (restored, restored_optimizer)):
        optim.zero_grad(set_to_none=True)
        candidate(tokens, labels=tokens).loss.backward()
        optim.step()
    torch.testing.assert_close(restored(tokens).logits, model(tokens).logits, atol=0, rtol=0)


def test_fail_closed_for_reinstallation_and_unsupported_batches():
    model = _official_model()
    install_official_adapters(model, _small_config(), layer_indices=(4,), backend="reference")
    with pytest.raises(ValueError, match="already installed"):
        install_official_adapters(model, _small_config(), layer_indices=(4,))
    tokens = torch.ones(1, 8, dtype=torch.int64)
    for kwargs in (
        {"attention_mask": torch.ones_like(tokens)},
        {"position_ids": torch.ones_like(tokens)},
        {"inputs_embeds": torch.zeros(1, 8, 16)},
    ):
        with pytest.raises(ValueError):
            model.model(tokens, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires frozen-container GPU FSDP2")
def test_install_after_fsdp_keeps_replicated_adapter_masters_and_gradients(tmp_path):
    import torch.distributed as dist
    from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor

    if dist.is_initialized():
        pytest.skip("one-rank isolated process-group test")
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    try:
        model = _official_model().cuda()
        model.requires_grad_(False)
        mesh = init_device_mesh("cuda", (1,))
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
            cast_forward_inputs=False,
        )
        for layer in model.model.layers.values():
            fully_shard_by_dtype(
                layer,
                mesh=mesh,
                mp_policy=policy,
                offload_policy=None,
                fp32_compute_module_names=tuple(model._keep_in_fp32_modules_strict),
                reshard_after_forward=True,
            )
        fully_shard(
            model.model.embed_tokens, mesh=mesh, mp_policy=policy, reshard_after_forward=True
        )
        fully_shard(
            model.lm_head,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32),
            reshard_after_forward=True,
        )
        fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
        original = dict(model.named_parameters())
        adapters = install_official_adapters(
            model, _small_config(), layer_indices=(0, 4), backend="reference", device="cuda"
        )
        assert all(isinstance(p, DTensor) for p in original.values())
        assert all(
            not isinstance(p, DTensor) and p.dtype == torch.float32
            for adapter in adapters.values()
            for p in adapter.parameters()
        )
        tokens = torch.randint(0, 64, (1, 8), device="cuda")
        optimizer = torch.optim.AdamW(
            [p for a in adapters.values() for p in a.parameters()], lr=0.01
        )
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = model(tokens, labels=tokens)
            assert result.loss.isfinite()
            result.loss.backward()
            for name, parameter in model.named_parameters():
                if ADAPTER_MARKER in name:
                    assert (
                        not isinstance(parameter, DTensor) and parameter.dtype == torch.float32
                    ), name
                    assert parameter.grad is not None and parameter.grad.isfinite().all(), name
                    if step:
                        assert parameter.grad.count_nonzero(), name
                else:
                    assert parameter.grad is None, name
            optimizer.step()
    finally:
        dist.destroy_process_group()
