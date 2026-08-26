import math

import pytest
import torch

from archlab.architectures.base import GPT, GPTConfig
from archlab.architectures.kimi import (
    KDA_CHUNK_SIZE,
    KDA_CONV_SIZE,
    KimiDeltaAttention,
    KimiKDA,
    KimiKDAConfig,
    NoPEGatedBlock,
    _call_fused_kda_gate,
    kda_gate_reference,
    kda_layer_map,
    kda_recurrent_reference,
)
from archlab.optimizers.speedrun import setup_model_optimizer
from archlab.speedrun.models import build_model_from_config_kwargs


def tiny_config(**overrides):
    values = dict(
        sequence_len=16,
        vocab_size=256,
        n_layer=4,
        n_head=1,
        n_kv_head=1,
        n_embd=128,
        window_pattern="L",
    )
    values.update(overrides)
    return KimiKDAConfig(**values)


def test_d14_layer_map_and_global_rope_policy():
    assert kda_layer_map(14) == {
        "kda": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13],
        "global": [4, 8, 12, 14],
    }
    config = tiny_config(n_layer=14)
    assert config.kda_rope_policy == "global_only"
    assert config.kda_chunk_size == KDA_CHUNK_SIZE == 64
    assert config.kda_conv_size == KDA_CONV_SIZE == 4


def test_reference_recurrence_matches_explicit_equation_and_backward_is_finite():
    torch.manual_seed(7)
    q = torch.randn(2, 5, 2, 4, requires_grad=True)
    k = torch.randn(2, 5, 2, 4, requires_grad=True)
    v = torch.randn(2, 5, 2, 3, requires_grad=True)
    g = -torch.rand(2, 5, 2, 4, requires_grad=True)
    beta = torch.sigmoid(torch.randn(2, 5, 2, requires_grad=True))
    actual = kda_recurrent_reference(q, k, v, g, beta)

    qn = torch.nn.functional.normalize(q.float(), dim=-1)
    kn = torch.nn.functional.normalize(k.float(), dim=-1)
    state = torch.zeros(2, 2, 4, 3)
    expected = []
    for t in range(5):
        state = state * g[:, t].float().exp().unsqueeze(-1)
        pred = torch.einsum("bhk,bhkv->bhv", kn[:, t], state)
        state = state + kn[:, t].unsqueeze(-1) * (
            beta[:, t].float().unsqueeze(-1) * (v[:, t].float() - pred)
        ).unsqueeze(-2)
        expected.append(torch.einsum("bhkv,bhk->bhv", state, qn[:, t]) * q.size(-1) ** -0.5)
    expected = torch.stack(expected, dim=1)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_gate_reference_matches_kda_parameterization():
    raw = torch.zeros(1, 2, 8)
    a_log = torch.tensor([0.0, math.log(2.0)])
    bias = torch.zeros(8)
    gate = kda_gate_reference(raw, a_log, bias, head_dim=4)
    assert gate.shape == (1, 2, 2, 4)
    torch.testing.assert_close(gate[..., 0, :], torch.full((1, 2, 4), -math.log(2)))
    torch.testing.assert_close(gate[..., 1, :], torch.full((1, 2, 4), -2 * math.log(2)))


def test_fla_042_gate_adapter_reshapes_and_preserves_dtype():
    raw = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    a_log = torch.zeros(2)
    bias = torch.zeros(8)
    call = {}

    def fused(g, A_log, *, dt_bias, output_dtype):
        call.update(g=g, A_log=A_log, dt_bias=dt_bias, output_dtype=output_dtype)
        return g

    result = _call_fused_kda_gate(fused, "0.4.2", raw, a_log, bias, 2, 4)

    assert result.shape == (1, 2, 2, 4)
    assert call == {
        "g": result,
        "A_log": a_log,
        "dt_bias": bias,
        "output_dtype": torch.bfloat16,
    }


def test_fla_040_gate_adapter_preserves_legacy_call_contract():
    raw = torch.zeros(1, 2, 8)
    a_log = torch.zeros(2)
    bias = torch.zeros(8)
    call = {}

    def fused(g, A_log, head_dim, *, g_bias):
        call.update(g=g, A_log=A_log, head_dim=head_dim, g_bias=g_bias)
        return g.view(1, 2, 2, 4)

    result = _call_fused_kda_gate(fused, "0.4.0", raw, a_log, bias, 2, 4)

    assert result.shape == (1, 2, 2, 4)
    assert call == {"g": raw, "A_log": a_log, "head_dim": 4, "g_bias": bias}


@pytest.mark.parametrize("variant", ["kimi_k3", "solar_negative"])
def test_frontier_kda_variants_use_nope_gated_globals_and_full_rank_gate(variant):
    config = tiny_config(kda_variant=variant, kda_rope_policy="none")
    model = KimiKDA(config)
    globals_ = [
        block
        for index, block in enumerate(model.transformer.h)
        if index + 1 in kda_layer_map(config.n_layer, config.kda_pattern)["global"]
    ]
    assert globals_ and all(isinstance(block, NoPEGatedBlock) for block in globals_)
    kda = next(
        block.attn for block in model.transformer.h if isinstance(block.attn, KimiDeltaAttention)
    )
    assert kda.g_proj.weight.shape == (config.n_embd, config.n_embd)
    state = model.get_architecture_state()
    assert state["rope_policy"] == "none"
    assert state["output_gate"] == "full_rank"
    assert state["negative_eigenvalues"] is (variant == "solar_negative")


def test_kimi_k3_decay_is_lower_bounded_and_solar_beta_allows_over_relaxation():
    k3 = KimiDeltaAttention(tiny_config(kda_variant="kimi_k3", kda_rope_policy="none"), layer_idx=0)
    k3.a_log.data.fill_(math.log(2.0))
    k3.dt_bias.data.zero_()
    raw = torch.linspace(-20, 20, 128).view(1, 1, 128)
    decay = k3._gate(raw)
    assert torch.all(decay < 0)
    assert torch.all(decay >= -5.0)

    solar = KimiDeltaAttention(
        tiny_config(kda_variant="solar_negative", kda_rope_policy="none"), layer_idx=0
    )
    solar.beta_proj.weight.data.zero_()
    beta = 2.0 * torch.sigmoid(solar.beta_proj(torch.zeros(1, 1, 128)))
    torch.testing.assert_close(beta, torch.ones_like(beta))


def test_kda_variant_rope_contract_is_rejected_early():
    with pytest.raises(ValueError, match="K3/Solar variants require NoPE"):
        KimiKDA(tiny_config(kda_variant="kimi_k3", kda_rope_policy="global_only"))
    with pytest.raises(ValueError, match="legacy controlled KDA"):
        KimiKDA(tiny_config(kda_variant="kimi_linear", kda_rope_policy="none"))


def test_solar_softmax_first_pattern_does_not_force_final_global():
    config = tiny_config(
        n_layer=8,
        kda_variant="solar_negative",
        kda_rope_policy="none",
        kda_pattern="GKKK",
        kda_force_final_global=False,
    )
    model = KimiKDA(config)
    assert kda_layer_map(8, "GKKK", force_final_global=False) == {
        "kda": [2, 3, 4, 6, 7, 8],
        "global": [1, 5],
    }
    state = model.get_architecture_state()
    assert state["force_final_global"] is False
    assert state["negative_eigenvalues"] is True


def test_value_embedding_changes_kda_output_on_cpu():
    torch.manual_seed(3)
    attn = KimiDeltaAttention(tiny_config(n_layer=2), layer_idx=1)
    for module in attn.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    for conv in (attn.q_conv, attn.k_conv, attn.v_conv):
        conv.weight.data.zero_()
        conv.weight.data[:, -1] = 1.0
    attn.a_log.data.zero_()
    attn.dt_bias.data.zero_()
    attn.output_norm_weight.data.fill_(1.0)
    x = torch.randn(1, 6, 128)
    ve = torch.randn(1, 6, 128)
    without = attn(x, torch.zeros_like(ve), None, None, None)
    with_value = attn(x, ve, None, None, None)
    assert torch.isfinite(with_value).all()
    assert not torch.allclose(without, with_value)


def test_optimizer_groups_kda_vectors_at_fixed_adamw_lr():
    model = KimiKDA(tiny_config())
    model.init_weights()
    optimizer = setup_model_optimizer(model, scalar_lr=0.5)
    vector_groups = [
        group
        for group in optimizer.param_groups
        if group["kind"] == "adamw" and math.isclose(group["lr"], 0.005)
    ]
    assert vector_groups
    kda_vectors = [
        parameter
        for block in model.transformer.h
        for parameter in block.attn.parameters()
        if parameter.ndim < 2
    ]
    grouped_ids = {id(parameter) for group in vector_groups for parameter in group["params"]}
    assert {id(parameter) for parameter in kda_vectors} <= grouped_ids
    assert model.num_scaling_params()["total"] == sum(p.numel() for p in model.parameters())
    assert model.estimate_flops() > 0


@pytest.mark.parametrize(
    ("variant", "rope_policy"),
    [("kimi_linear", "global_only"), ("kimi_k3", "none")],
)
def test_kda_shared_forward_has_empty_engram_interface(variant, rope_policy):
    model = KimiKDA(tiny_config(kda_variant=variant, kda_rope_policy=rope_policy))
    model.init_weights()
    tokens = torch.randint(0, model.config.vocab_size, (1, model.config.sequence_len))

    loss = model(tokens, targets=tokens.roll(-1, dims=1))

    assert not model.engrams
    assert torch.isfinite(loss)


def test_kda_shared_initialization_is_bit_identical_to_paired_baseline():
    config = tiny_config()
    baseline_config = GPTConfig(
        **{
            key: value
            for key, value in config.__dict__.items()
            if not key.startswith("kda_") and key != "arch_family"
        }
    )
    torch.manual_seed(42)
    with torch.device("meta"):
        baseline = GPT(baseline_config)
    baseline.to_empty(device="cpu")
    baseline.init_weights()

    torch.manual_seed(42)
    with torch.device("meta"):
        treatment = KimiKDA(config)
    treatment.to_empty(device="cpu")
    treatment.init_weights()

    for name in (
        "transformer.wte.weight",
        "lm_head.weight",
        "resid_lambdas",
        "x0_lambdas",
        "smear_gate.weight",
        "smear_lambda",
        "backout_lambda",
    ):
        torch.testing.assert_close(
            baseline.state_dict()[name], treatment.state_dict()[name], rtol=0, atol=0
        )
    for name, tensor in baseline.value_embeds.state_dict().items():
        torch.testing.assert_close(
            tensor, treatment.value_embeds.state_dict()[name], rtol=0, atol=0
        )
    for base_block, treatment_block in zip(
        baseline.transformer.h, treatment.transformer.h, strict=True
    ):
        torch.testing.assert_close(
            base_block.mlp.c_fc.weight, treatment_block.mlp.c_fc.weight, rtol=0, atol=0
        )
        torch.testing.assert_close(
            base_block.mlp.c_proj.weight, treatment_block.mlp.c_proj.weight, rtol=0, atol=0
        )
        if isinstance(treatment_block.attn, KimiDeltaAttention):
            pairs = (
                (base_block.attn.c_q.weight, treatment_block.attn.q_proj.weight),
                (base_block.attn.c_k.weight, treatment_block.attn.k_proj.weight),
                (base_block.attn.c_v.weight, treatment_block.attn.v_proj.weight),
                (base_block.attn.c_proj.weight, treatment_block.attn.o_proj.weight),
            )
        else:
            pairs = (
                (base_block.attn.c_q.weight, treatment_block.attn.c_q.weight),
                (base_block.attn.c_k.weight, treatment_block.attn.c_k.weight),
                (base_block.attn.c_v.weight, treatment_block.attn.c_v.weight),
                (base_block.attn.c_proj.weight, treatment_block.attn.c_proj.weight),
            )
        for expected, actual in pairs:
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        if base_block.attn.ve_gate is not None:
            torch.testing.assert_close(
                base_block.attn.ve_gate.weight,
                treatment_block.attn.ve_gate.weight,
                rtol=0,
                atol=0,
            )


def test_legacy_nanochat_config_still_loads():
    config = GPTConfig(
        sequence_len=16,
        vocab_size=256,
        n_layer=2,
        n_head=1,
        n_kv_head=1,
        n_embd=128,
        window_pattern="L",
    )
    model, restored = build_model_from_config_kwargs(config.__dict__)
    assert isinstance(model, GPT)
    assert restored.arch_family == "nanochat"


def test_kda_grouped_query_attention_is_rejected():
    with pytest.raises(ValueError, match="does not support GQA"):
        KimiDeltaAttention(tiny_config(n_embd=256, n_head=2, n_kv_head=1), layer_idx=0)
