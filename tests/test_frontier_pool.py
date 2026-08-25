from __future__ import annotations

import pytest
import torch

from next_gen_arch.arch.frontier import (
    FRONTIER_VARIANTS,
    FrontierPoolConfig,
    FrontierPoolGPT,
    DeepSeekCompressedAttention,
    GLMMultiLatentAttention,
    MotifGroupedDifferentialLatentAttention,
    MotifMHCConnection,
    InklingShortConvolution,
    HeadSplitLinear,
    QwenGatedDeltaAttention,
    ZeroCenteredRMSNorm,
)
from next_gen_arch.training.models import build_model_from_config_kwargs


def config(variant: str, *, n_layer: int = 2) -> FrontierPoolConfig:
    return FrontierPoolConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=n_layer,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        frontier_variant=variant,
        relative_dim=4,
        relative_extent=8,
    )


@pytest.mark.parametrize("variant", sorted(FRONTIER_VARIANTS))
def test_frontier_variant_forward_backward(variant: str):
    torch.manual_seed(7)
    model = FrontierPoolGPT(config(variant))
    model.init_weights()
    model.train()
    tokens = torch.randint(0, 64, (2, 12))
    targets = torch.randint(0, 64, (2, 12))
    loss = model(tokens, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert model.num_scaling_params()["total"] == sum(p.numel() for p in model.parameters())


def test_relative_attention_has_no_rope_dependency():
    torch.manual_seed(3)
    model = FrontierPoolGPT(config("inkling_relative_attention"))
    model.init_weights()
    model.eval()
    tokens = torch.randint(0, 64, (1, 8))
    expected = model(tokens)
    model.cos.zero_()
    model.sin.fill_(1)
    actual = model(tokens)
    torch.testing.assert_close(actual, expected)


def test_sconv_is_fp32_residual_and_causal():
    conv = InklingShortConvolution(3, kernel_size=4)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[:, -1].fill_(1.0)
    x = torch.randn(2, 7, 3, dtype=torch.bfloat16)
    out = conv(x)
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, 2 * x)
    changed = x.clone()
    changed[:, -1].add_(10)
    torch.testing.assert_close(conv(changed)[:, :-1], out[:, :-1])


def test_hybrid_pattern_is_exact_and_final_global():
    model = FrontierPoolGPT(config("hybrid_swa_5_1_w512", n_layer=14))
    assert model.window_sizes == [
        (512, 0), (512, 0), (512, 0), (512, 0), (512, 0), (16, 0),
        (512, 0), (512, 0), (512, 0), (512, 0), (512, 0), (16, 0),
        (512, 0), (16, 0),
    ]


def test_zero_centered_norm_is_identity_parameterization():
    module = ZeroCenteredRMSNorm(8)
    module.reset_parameters()
    x = torch.randn(2, 3, 8)
    torch.testing.assert_close(module(x), torch.nn.functional.rms_norm(x, (8,)))


def test_shared_mtp_logs_lm_and_auxiliary_loss():
    model = FrontierPoolGPT(config("shared_mtp3"))
    model.init_weights()
    model.train()
    tokens = torch.randint(0, 64, (2, 12))
    loss = model(tokens, tokens.roll(-1, dims=1))
    metrics = model.consume_training_metrics()
    assert torch.isfinite(loss)
    assert set(metrics) == {"train/lm_loss", "mtp/loss"}
    assert metrics["mtp/loss"] > 0


def test_checkpoint_config_round_trip():
    cfg = config("partial_rope_25")
    model, restored = build_model_from_config_kwargs(cfg.__dict__)
    assert isinstance(model, FrontierPoolGPT)
    assert restored == cfg


def test_per_head_muon_uses_independent_2d_qkv_matrices():
    model = FrontierPoolGPT(config("per_head_muon"))
    model.init_weights()
    for block in model.transformer.h:
        for projection in (block.attn.c_q, block.attn.c_k, block.attn.c_v):
            assert isinstance(projection, HeadSplitLinear)
            assert len(projection.weights) == 2
            assert all(weight.shape == (16, 32) for weight in projection.weights)
    optimizer = model.setup_optimizer()
    qkv_ids = {
        id(weight)
        for block in model.transformer.h
        for projection in (block.attn.c_q, block.attn.c_k, block.attn.c_v)
        for weight in projection.weights
    }
    muon_ids = {id(p) for group in optimizer.param_groups if group["kind"] == "muon" for p in group["params"]}
    assert qkv_ids <= muon_ids


def test_qwen_gdn_and_glm_simple_gdn_layer_patterns():
    qwen = FrontierPoolGPT(config("qwen_gdn", n_layer=8))
    assert [isinstance(block.attn, QwenGatedDeltaAttention) for block in qwen.transformer.h] == [
        True, True, True, False, True, True, True, False
    ]
    assert all(
        not block.attn.simple
        for block in qwen.transformer.h
        if isinstance(block.attn, QwenGatedDeltaAttention)
    )

    simple = FrontierPoolGPT(config("glm_simple_gdn", n_layer=8))
    assert [isinstance(block.attn, QwenGatedDeltaAttention) for block in simple.transformer.h] == [
        True, False, True, False, True, False, True, False
    ]
    assert all(
        block.attn.simple
        for block in simple.transformer.h
        if isinstance(block.attn, QwenGatedDeltaAttention)
    )


def test_qwen_gdn_checkpoint_metadata_and_parameter_cap():
    cfg = config("qwen_gdn", n_layer=4)
    model, restored = build_model_from_config_kwargs(cfg.__dict__)
    model.init_weights()
    state = model.get_architecture_state()
    assert restored.frontier_variant == "qwen_gdn"
    assert state["layer_pattern"] == "GGGF"
    assert state["conv_kernel"] == 4
    assert model.num_scaling_params()["total"] < 800_000_000


@pytest.mark.parametrize(
    "variant,compression,overlap",
    [("deepseek_csa", 4, True), ("deepseek_hca", 128, False)],
)
def test_deepseek_compressed_attention_bundle(variant, compression, overlap):
    model = FrontierPoolGPT(config(variant, n_layer=2))
    model.init_weights()
    assert all(isinstance(block.attn, DeepSeekCompressedAttention) for block in model.transformer.h)
    state = model.get_architecture_state()
    assert state["compression"] == compression
    assert state["overlap"] is overlap
    assert state["local_window"] == 128
    assert state["inverse_rope_output"] is True
    tokens = torch.randint(0, 64, (1, 12))
    loss = model(tokens, tokens.roll(-1, 1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(embedding.weight.grad is not None for embedding in model.value_embeds.values())


def test_glm_mla_caps_latent_at_width_and_uses_head_split_muon_updates():
    model = FrontierPoolGPT(config("glm_mla_muon_split", n_layer=2))
    model.init_weights()
    for block in model.transformer.h:
        assert isinstance(block.attn, GLMMultiLatentAttention)
        assert block.attn.latent_dim == min(256, model.config.n_embd)
        assert all(
            isinstance(projection, HeadSplitLinear)
            for projection in (block.attn.q_up, block.attn.k_up, block.attn.v_up)
        )
    optimizer = model.setup_optimizer()
    split_ids = {
        id(weight)
        for block in model.transformer.h
        for projection in (block.attn.q_up, block.attn.k_up, block.attn.v_up)
        for weight in projection.weights
    }
    muon_ids = {
        id(parameter)
        for group in optimizer.param_groups
        if group["kind"] == "muon"
        for parameter in group["params"]
    }
    assert split_ids <= muon_ids


def test_motif_gdla_grouped_head_adaptation_and_attention_schedule():
    model = FrontierPoolGPT(config("motif_gdla", n_layer=8))
    model.init_weights()
    assert model.window_sizes == [
        (16, 0), (128, 0), (128, 0), (128, 0),
        (16, 0), (128, 0), (128, 0), (128, 0),
    ]
    assert all(
        isinstance(block.attn, MotifGroupedDifferentialLatentAttention)
        for block in model.transformer.h
    )
    state = model.get_architecture_state()
    assert state["signal_heads"] == 1
    assert state["noise_heads"] == 1
    assert state["query_dependent_lambda"] is True
    assert state["elementwise_output_gate"] is True


def test_motif_mhc_post_scale_anneals_two_to_one_and_stays_doubly_stochastic():
    cfg = config("motif_mhc_anneal", n_layer=2)
    cfg.mhc_anneal_steps = 100
    model = FrontierPoolGPT(cfg)
    model.init_weights()
    assert len(model.mhc_connections) == 4
    assert all(isinstance(connection, MotifMHCConnection) for connection in model.mhc_connections)
    scale_buffers = [connection.post_scale for connection in model.mhc_connections]
    assert all(torch.is_tensor(scale) and scale.ndim == 0 for scale in scale_buffers)
    assert all("post_scale" in dict(connection.named_buffers()) for connection in model.mhc_connections)
    scale_pointers = [scale.data_ptr() for scale in scale_buffers]
    model.set_training_step(50)
    assert all(connection.post_scale == 1.5 for connection in model.mhc_connections)
    assert [scale.data_ptr() for scale in scale_buffers] == scale_pointers
    streams = torch.randn(2, 3, cfg.mhc_num_streams, cfg.n_embd)
    _, post, residual = model.mhc_connections[0].mappings(streams)
    assert post.min() >= 0 and post.max() <= 1.5
    torch.testing.assert_close(residual.sum(-1), torch.ones_like(residual.sum(-1)), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(residual.sum(-2), torch.ones_like(residual.sum(-2)), atol=1e-4, rtol=1e-4)
    model.set_training_step(100)
    assert all(connection.post_scale == 1.0 for connection in model.mhc_connections)


def test_motif_mhc_post_scale_updates_do_not_recompile():
    cfg = config("motif_mhc_anneal", n_layer=2)
    connection = MotifMHCConnection(cfg)
    compiled_graphs = []

    def counting_backend(graph_module, _example_inputs):
        compiled_graphs.append(graph_module)
        return graph_module.forward

    compiled = torch.compile(connection.mappings, backend=counting_backend)
    streams = torch.randn(1, 3, cfg.mhc_num_streams, cfg.n_embd)
    compiled(streams)
    connection.set_post_scale(1.5)
    compiled(streams)
    connection.set_post_scale(1.0)
    compiled(streams)
    assert len(compiled_graphs) == 1
