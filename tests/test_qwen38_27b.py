from dataclasses import replace

import torch

from archlab.architectures.qwen38_27b import (
    SOURCE_CONFIG_SHA256,
    SOURCE_MODEL,
    SOURCE_REVISION,
    TOKENIZER_SHA256,
    Qwen38Dense,
    Qwen38DenseAttention,
    Qwen38DenseConfig,
)
from archlab.architectures.qwen38_flash_next import GatedDeltaAttention


def tiny_config(**overrides) -> Qwen38DenseConfig:
    config = Qwen38DenseConfig(
        vocab_size=64,
        sequence_len=8,
        max_position_embeddings=64,
        eos_token_id=63,
        num_hidden_layers=4,
        hidden_size=32,
        intermediate_size=48,
        attention_heads=4,
        attention_kv_heads=2,
        attention_head_dim=8,
        linear_qk_heads=2,
        linear_v_heads=4,
        linear_key_dim=4,
        linear_value_dim=4,
    )
    return replace(config, **overrides)


def test_source_identity_and_quarter_shape_match_pinned_qwen38_27b():
    config = Qwen38DenseConfig()

    assert SOURCE_MODEL == "Qwen/Qwen3.8-27B"
    assert len(SOURCE_REVISION) == 40
    assert len(SOURCE_CONFIG_SHA256) == 64
    assert len(TOKENIZER_SHA256) == 64
    assert config.num_hidden_layers == 64 // 4
    assert config.hidden_size == 5_120 // 4
    assert config.intermediate_size == 17_408 // 4
    assert config.attention_heads == 24 // 4
    assert config.attention_kv_heads == 4 // 4
    assert config.attention_head_dim == 256 // 4
    assert config.linear_qk_heads == 16 // 4
    assert config.linear_v_heads == 48 // 4
    assert config.linear_key_dim == 128 // 4
    assert config.linear_value_dim == 128 // 4
    assert config.linear_conv_kernel == 4
    assert config.vocab_size == 248_320
    assert config.mtp_layers == 1


def test_dense_layer_pattern_preserves_three_gdn_then_full_attention():
    model = Qwen38Dense(tiny_config())

    assert [layer.attention_kind for layer in model.layers] == [
        "gdn",
        "gdn",
        "gdn",
        "full_attention",
    ]
    assert model.mtp_block.attention_kind == "full_attention"
    assert isinstance(model.layers[0].attention, GatedDeltaAttention)
    assert isinstance(model.layers[-1].attention, Qwen38DenseAttention)


def test_dense_optimizer_partition_assigns_every_parameter_once():
    model = Qwen38Dense(tiny_config())
    contract = model.optimizer_contract(require_two_dimensional_muon=True)
    gdn = model.layers[0].attention
    attention = model.layers[-1].attention

    assert contract["all_trainable_parameters_assigned_once"] is True
    assert sum(bucket["parameters"] for bucket in contract["optimizers"].values()) == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert gdn.qkv.weight.archlab_optimizer == "muon"
    assert not hasattr(gdn.qkv.weight, "archlab_muon_split_rows")
    assert gdn.z.weight.archlab_optimizer == "adamw"
    assert attention.q_gate.weight.archlab_optimizer == "muon"
    assert not hasattr(attention.q_gate.weight, "archlab_muon_split_rows")
    assert model.layers[0].mlp.gate.weight.archlab_optimizer == "muon"
    assert model.token_embedding.weight.archlab_optimizer == "adamw"
    assert model.lm_head.weight.archlab_optimizer == "adamw"


def test_dense_tiny_forward_backward_is_finite():
    torch.manual_seed(7)
    model = Qwen38Dense(tiny_config())
    model.init_weights()
    tokens = torch.randint(0, model.config.vocab_size, (2, model.config.sequence_len))
    labels = torch.roll(tokens, shifts=-1, dims=1)
    labels[:, -1] = -1

    losses = model(tokens, labels, loss_reduction="none")

    assert losses.shape == labels.shape
    assert torch.isfinite(losses).all()
    losses.mean().backward()
    assert torch.isfinite(model.layers[-1].attention.q_gate.weight.grad).all()
    assert torch.isfinite(model.layers[0].mlp.gate.weight.grad).all()


def test_dense_full_attention_is_causal():
    torch.manual_seed(17)
    attention = Qwen38DenseAttention(tiny_config(), runtime_backend="native")
    for parameter in attention.parameters():
        torch.nn.init.normal_(parameter)
    x = torch.randn(2, 8, 32)
    rotary = torch.zeros(1, 8, 1, 2)

    output = attention(x, rotary.cos(), rotary.sin())
    perturbed = x.clone()
    perturbed[:, 1:] += 100.0
    perturbed_output = attention(perturbed, rotary.cos(), rotary.sin())

    assert torch.allclose(output[:, 0], perturbed_output[:, 0], atol=1e-5, rtol=1e-5)


def test_default_dense_model_constructs_on_meta_at_about_one_billion_parameters():
    with torch.device("meta"):
        model = Qwen38Dense(Qwen38DenseConfig())
    counts = model.num_scaling_params()

    assert 850_000_000 < counts["total"] < 1_100_000_000
    assert counts["embeddings_and_head"] > counts["text_backbone"]
