import copy

import pytest
import torch

from next_gen_arch.arch.base import GPT, GPTConfig
from next_gen_arch.arch.kimi import AttnResRead, KimiAttnRes, KimiAttnResConfig
from next_gen_arch.training.models import (
    build_model_config,
    build_model_from_config_kwargs,
    instantiate_model,
    model_config_to_dict,
    patch_model_config_kwargs,
)
from next_gen_arch.training.optim import setup_model_optimizer


def tiny_config(**overrides):
    values = dict(
        sequence_len=8,
        vocab_size=256,
        n_layer=4,
        n_head=1,
        n_kv_head=1,
        n_embd=128,
        window_pattern="L",
        arch_family="kimi_attnres",
        attn_res_block_size=2,
        attn_res_recompute=False,
    )
    values.update(overrides)
    return KimiAttnResConfig(**values)


def test_d14_block_map_matches_scaled_k3_eight_source_contract():
    config = build_model_config(
        arch_family="kimi_attnres",
        depth=14,
        aspect_ratio=64,
        head_dim=128,
        max_seq_len=2048,
        vocab_size=32768,
        window_pattern="L",
        fog_variant="flash",
    )
    with torch.device("meta"):
        model = instantiate_model(config)
    assert model.residual_source_counts() == {
        "pre_attention": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
        "pre_mlp": [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
        "final": 8,
    }
    state = model.get_architecture_state()
    assert state["completed_transformer_blocks"] == 7
    assert state["final_source_count_including_embedding"] == 8


def test_attn_res_read_forward_backward_matches_fp32_k3_reference():
    torch.manual_seed(3)
    actual = AttnResRead(16)
    actual.query.data.normal_(std=0.2)
    actual.norm_weight.data.uniform_(0.8, 1.2)
    reference = copy.deepcopy(actual)
    actual_sources = [torch.randn(2, 5, 16, requires_grad=True) for _ in range(4)]
    reference_sources = [source.detach().clone().requires_grad_(True) for source in actual_sources]

    output = actual(*actual_sources)
    values = torch.stack(reference_sources, dim=-2).float()
    keys = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + reference.eps)
    score_weight = reference.norm_weight.float() * reference.query.float()
    probabilities = torch.softmax((keys * score_weight).sum(-1), dim=-1)
    expected = (probabilities.unsqueeze(-1) * values).sum(-2)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)

    gradient = torch.randn_like(output)
    output.backward(gradient)
    expected.backward(gradient)
    for lhs, rhs in zip(actual_sources, reference_sources, strict=True):
        torch.testing.assert_close(lhs.grad, rhs.grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.query.grad, reference.query.grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        actual.norm_weight.grad, reference.norm_weight.grad, rtol=2e-5, atol=2e-6
    )


def test_zero_query_is_uniform_and_bf16_gradients_are_finite():
    read = AttnResRead(32).to(dtype=torch.bfloat16)
    sources = [torch.randn(2, 4, 32, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    output = read(*sources)
    expected = torch.stack(sources).float().mean(0).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=8e-3, atol=4e-3)
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(source.grad).all() for source in sources)
    assert torch.isfinite(read.query.grad).all()
    assert torch.isfinite(read.norm_weight.grad).all()


def test_multi_head_attnres_matches_block_diagonal_reference_without_extra_params():
    torch.manual_seed(23)
    read = AttnResRead(32, heads=8)
    read.query.data.normal_(std=0.2)
    read.norm_weight.data.uniform_(0.8, 1.2)
    sources = [torch.randn(2, 5, 32, requires_grad=True) for _ in range(4)]
    output = read(*sources)

    values = torch.stack(sources, dim=-2).float()  # B,T,N,D
    keys = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + read.eps)
    keys = keys.view(2, 5, 4, 8, 4)
    query = (read.norm_weight.float() * read.query.float()).view(8, 4)
    weights = torch.softmax((keys * query).sum(-1), dim=-2)
    expected = (weights.unsqueeze(-1) * values.view(2, 5, 4, 8, 4)).sum(-3).reshape(2, 5, 32)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    assert sum(p.numel() for p in read.parameters()) == sum(
        p.numel() for p in AttnResRead(32, heads=1).parameters()
    )
    output.square().mean().backward()
    assert all(source.grad is not None and torch.isfinite(source.grad).all() for source in sources)


def test_mhar_configuration_uses_eight_zero_parameter_routing_heads():
    config = build_model_config(
        arch_family="kimi_attnres",
        depth=14,
        aspect_ratio=64,
        head_dim=128,
        max_seq_len=2048,
        vocab_size=32768,
        window_pattern="L",
        fog_variant="flash",
        attn_res_variant="multi_head_attnres",
        attn_res_heads=8,
    )
    with torch.device("meta"):
        model = instantiate_model(config)
    state = model.get_architecture_state()
    assert state["family"] == "kimi_attnres"
    assert state["variant"] == "multi_head_attnres"
    assert state["routing_heads"] == 8
    assert state["routing_head_dim"] == 112
    assert model.num_scaling_params()["total"] == 399_166_821


def test_shared_initialization_is_bit_identical_to_paired_baseline():
    baseline_config = GPTConfig(
        **{
            key: value
            for key, value in tiny_config().__dict__.items()
            if not key.startswith("attn_res_") and key != "arch_family"
        }
    )
    torch.manual_seed(42)
    with torch.device("meta"):
        baseline = GPT(baseline_config)
    baseline.to_empty(device="cpu")
    baseline.init_weights()
    torch.manual_seed(42)
    with torch.device("meta"):
        treatment = KimiAttnRes(tiny_config())
    treatment.to_empty(device="cpu")
    treatment.init_weights()

    treatment_state = treatment.state_dict()
    replaced = {"resid_lambdas", "x0_lambdas", "backout_lambda"}
    for name, tensor in baseline.state_dict().items():
        if name in replaced:
            continue
        torch.testing.assert_close(tensor, treatment_state[name], rtol=0, atol=0)
    assert all(torch.count_nonzero(read.query) == 0 for read in treatment._attn_res_reads())
    assert all(
        torch.equal(read.norm_weight, torch.ones_like(read.norm_weight))
        for read in treatment._attn_res_reads()
    )


def test_value_embeddings_remain_connected_to_the_attention_layers():
    model = KimiAttnRes(tiny_config())
    model.init_weights()
    seen = []
    for block in model.transformer.h:
        original = block.attn.forward

        def capture(*args, _original=original, **kwargs):
            seen.append(args[1] is not None)
            return _original(*args, **kwargs)

        block.attn.forward = capture
    model.eval()
    model(torch.randint(0, 256, (1, 8)))
    assert seen == [False, True, False, True]


def test_optimizer_group_parameter_and_flop_contracts():
    config = build_model_config(
        arch_family="kimi_attnres",
        depth=14,
        aspect_ratio=64,
        head_dim=128,
        max_seq_len=2048,
        vocab_size=32768,
        window_pattern="L",
        fog_variant="flash",
    )
    with torch.device("meta"):
        meta_model = instantiate_model(config)
    counts = meta_model.num_scaling_params()
    assert counts["attn_res"] == 51_968
    assert counts["total"] == 399_166_821
    assert counts["total"] < 800_000_000
    assert meta_model.estimate_flops() == 1_295_200_200
    assert meta_model.estimate_executed_flops() == 1_295_200_200

    model = KimiAttnRes(tiny_config())
    model.init_weights()
    optimizer = setup_model_optimizer(model, scalar_lr=0.5)
    attn_res_group = next(group for group in optimizer.param_groups if group.get("attn_res"))
    assert attn_res_group["lr"] == pytest.approx(0.005)
    assert attn_res_group["weight_decay"] == 0.0
    assert {id(p) for p in attn_res_group["params"]} == {
        id(p) for read in model._attn_res_reads() for p in read.parameters()
    }
    trainable_names = {name for name, _ in model.named_parameters()}
    assert "resid_lambdas" not in trainable_names
    assert "x0_lambdas" not in trainable_names
    assert "backout_lambda" not in trainable_names


def test_checkpoint_config_round_trip_and_legacy_baseline_loading():
    model = KimiAttnRes(tiny_config())
    model.init_weights()
    rebuilt, rebuilt_config = build_model_from_config_kwargs(model_config_to_dict(model.config))
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert rebuilt_config.arch_family == "kimi_attnres"
    assert rebuilt_config.attn_res_block_size == 2

    legacy = patch_model_config_kwargs(
        {
            "sequence_len": 8,
            "vocab_size": 256,
            "n_layer": 2,
            "n_head": 1,
            "n_kv_head": 1,
            "n_embd": 128,
        }
    )
    assert legacy["arch_family"] == "nanochat"
    baseline, baseline_config = build_model_from_config_kwargs(legacy)
    assert isinstance(baseline, GPT)
    assert baseline_config.arch_family == "nanochat"
