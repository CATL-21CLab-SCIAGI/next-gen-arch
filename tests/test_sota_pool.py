import math

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.model_factory import build_model_from_config_kwargs, model_config_to_dict
from nanochat.sota_pool import (
    CausalDepthwiseConv,
    DynamicTanh,
    SOTA_VARIANTS,
    SotaPoolConfig,
    SotaPoolGPT,
    XIELU,
)


def tiny_config(variant="baseline", **kwargs):
    values = dict(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        arch_family="sota_pool",
        sota_variant=variant,
    )
    values.update(kwargs)
    return SotaPoolConfig(**values)


def materialize(model, seed=42):
    torch.manual_seed(seed)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def test_pool_baseline_is_bit_identical_to_legacy_nanochat():
    legacy_config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    with torch.device("meta"):
        legacy = GPT(legacy_config)
        candidate = SotaPoolGPT(tiny_config())
    legacy = materialize(legacy, seed=123)
    candidate = materialize(candidate, seed=123)
    assert legacy.state_dict().keys() == candidate.state_dict().keys()
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)
    tokens = torch.randint(0, 128, (2, 8))
    torch.testing.assert_close(legacy(tokens), candidate(tokens), rtol=0, atol=0)


def test_meta_materialization_initializes_residual_controls_deterministically():
    with torch.device("meta"):
        first = SotaPoolGPT(tiny_config())
        second = SotaPoolGPT(tiny_config())
    first = materialize(first, seed=321)
    second = materialize(second, seed=321)
    torch.testing.assert_close(first.smear_gate.weight, second.smear_gate.weight, rtol=0, atol=0)
    torch.testing.assert_close(first.smear_lambda, torch.zeros_like(first.smear_lambda))
    torch.testing.assert_close(first.backout_lambda, torch.full_like(first.backout_lambda, 0.2))
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("variant", sorted(SOTA_VARIANTS - {"baseline"}))
def test_every_pool_variant_has_finite_forward_and_backward(variant):
    with torch.device("meta"):
        model = SotaPoolGPT(tiny_config(variant))
    model = materialize(model, seed=7)
    tokens = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    loss = model(tokens, targets)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)


def test_xielu_matches_reference_equation_and_initialization():
    module = XIELU()
    module.reset_parameters()
    x = torch.tensor([-2.0, -0.25, 0.0, 0.5, 2.0])
    alpha_p = torch.tensor(0.8)
    alpha_n = torch.tensor(0.8)
    expected = torch.where(
        x > 0,
        alpha_p * x.square() + 0.5 * x,
        alpha_n * torch.expm1(torch.minimum(x, torch.tensor(-1e-6))) - alpha_n * x + 0.5 * x,
    )
    torch.testing.assert_close(module(x), expected)


def test_dynamic_tanh_matches_official_llama_form():
    module = DynamicTanh(4)
    module.reset_parameters()
    x = torch.randn(3, 4)
    torch.testing.assert_close(module(x), torch.tanh(x))


def test_canon_depthwise_conv_is_causal_and_residual():
    module = CausalDepthwiseConv(3, kernel_size=4)
    with torch.no_grad():
        module.weight.fill_(0.25)
    x = torch.randn(2, 9, 3)
    changed = x.clone()
    changed[:, 6:] += 100
    y = module(x)
    y_changed = module(changed)
    torch.testing.assert_close(y[:, :6], y_changed[:, :6])
    assert y.shape == x.shape


def test_exclusive_attention_output_is_orthogonal_to_actual_self_value():
    with torch.device("meta"):
        model = SotaPoolGPT(tiny_config("exclusive_attention", n_layer=1))
    model = materialize(model, seed=11)
    attention = model.transformer.h[0].attn
    captured = {}

    def capture_input(_module, args):
        captured["heads"] = args[0].detach().view(2, 8, 2, 16)

    handle = attention.c_proj.register_forward_pre_hook(capture_input)
    tokens = torch.randint(0, 128, (2, 8))
    model(tokens)
    handle.remove()
    x = norm(model.transformer.wte(tokens))
    v = attention.c_v(x).view(2, 8, 2, 16)
    if attention.ve_gate is not None:
        ve = model.value_embeds["0"](tokens).view_as(v)
        gate = 3 * torch.sigmoid(attention.ve_gate(x[..., :12]))
        v = v + gate.unsqueeze(-1) * ve
    dot = (captured["heads"] * F_normalize(v)).sum(dim=-1)
    # The released training contract is BF16; normalization, projection, and
    # subtraction at that precision leave a small numerical residual.
    atol = 3e-2 if captured["heads"].dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(dot, torch.zeros_like(dot), atol=atol, rtol=0)


def F_normalize(x):
    return torch.nn.functional.normalize(x, dim=-1)


def test_bank_of_values_targets_last_third_and_uses_aligned_init():
    config = tiny_config("bank_of_values", n_layer=3)
    legacy_config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=3,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    with torch.device("meta"):
        candidate = SotaPoolGPT(config)
        legacy = GPT(legacy_config)
    effective_meta_count = candidate.num_scaling_params()["total"]
    candidate = materialize(candidate, seed=19)
    legacy = materialize(legacy, seed=19)
    target = candidate.transformer.h[2].attn
    assert target.is_bov_target
    assert not hasattr(target, "c_v")
    assert "2" not in candidate.value_embeds
    expected = legacy.transformer.h[2].attn.c_v(norm(legacy.transformer.wte.weight))
    torch.testing.assert_close(target.value_table.weight, expected, rtol=0, atol=0)
    assert candidate.get_architecture_state()["target_layers"] == [2]
    assert candidate.num_scaling_params()["total"] == effective_meta_count


def test_differential_attention_configuration_and_lambda_schedule():
    with torch.device("meta"):
        model = SotaPoolGPT(tiny_config("differential_attention"))
    attn0, attn1 = (block.attn for block in model.transformer.h)
    assert attn0.lambda_q1.shape == (8,)
    assert math.isclose(attn0.lambda_init, 0.2)
    assert attn1.lambda_init > attn0.lambda_init


def test_factory_round_trip_persists_pool_configuration():
    config = tiny_config("canon_abcd", canon_kernel_size=3, sota_extra_lr=0.004)
    model, restored = build_model_from_config_kwargs(model_config_to_dict(config))
    assert isinstance(model, SotaPoolGPT)
    assert restored == config
    assert model.get_architecture_state()["canon_set"] == "ABCD"
