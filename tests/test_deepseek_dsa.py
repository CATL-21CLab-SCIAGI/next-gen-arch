import math
import inspect

import pytest
import torch

from nanochat.deepseek_dsa import (
    DSA_BACKEND,
    DSACausalSelfAttention,
    DeepSeekDSA,
    DeepSeekDSAConfig,
    LightningIndexer,
    _apply_partial_noninterleaved_rope,
)
from nanochat.gpt import GPT, GPTConfig
from nanochat.model_factory import build_model_from_config_kwargs


def tiny_config(**overrides):
    values = dict(
        sequence_len=8,
        vocab_size=256,
        n_layer=2,
        n_head=1,
        n_kv_head=1,
        n_embd=128,
        window_pattern="L",
        dsa_top_k=2,
        dsa_index_heads=2,
        dsa_index_head_dim=128,
        dsa_index_rope_dim=64,
        dsa_dense_warmup_steps=2,
        dsa_query_chunk_size=3,
    )
    values.update(overrides)
    return DeepSeekDSAConfig(**values)


def shared_state(model):
    return {
        key: value
        for key, value in model.state_dict().items()
        if ".indexer." not in key
    }


def test_d14_parameter_contract_and_all_layer_placement_on_meta():
    config = DeepSeekDSAConfig(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=14,
        n_head=7,
        n_kv_head=7,
        n_embd=896,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = DeepSeekDSA(config)
    assert all(isinstance(block.attn, DSACausalSelfAttention) for block in model.transformer.h)
    counts = model.num_scaling_params()
    assert counts["indexer"] == 8_081_920
    assert counts["total"] == 407_196_802
    assert model.estimate_flops() < model.estimate_executed_flops()


def test_indexer_dimensions_partial_rope_and_backend_contract():
    config = tiny_config()
    indexer = LightningIndexer(config)
    assert indexer.n_heads == 2
    assert indexer.head_dim == 128
    assert indexer.rope_dim == 64
    assert config.dsa_backend == DSA_BACKEND
    x = torch.randn(1, 4, 2, 128)
    cos = torch.ones(1, 4, 1, 64)
    sin = torch.zeros_like(cos)
    y = _apply_partial_noninterleaved_rope(x, (cos, sin), 64)
    torch.testing.assert_close(y, x)
    with pytest.raises(ValueError, match="backend"):
        DeepSeekDSA(tiny_config(dsa_backend="unreleased_sparse_backward"))


def test_indexer_layer_norm_accepts_bf16_activations_with_fp32_parameters():
    indexer = LightningIndexer(tiny_config())
    hidden = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    cos = torch.ones(1, 4, 1, 64, dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    q, k, weights = indexer.project(hidden, (cos, sin))
    assert q.dtype == k.dtype == torch.bfloat16
    assert weights.dtype == torch.float32
    (q.float().square().mean() + k.float().square().mean() + weights.square().mean()).backward()
    assert indexer.k_norm.weight.dtype == torch.float32
    assert indexer.k_norm.weight.grad is not None
    assert torch.isfinite(indexer.k_norm.weight.grad).all()


def test_shared_initialization_is_bit_identical_to_baseline():
    baseline_config = GPTConfig(
        sequence_len=8,
        vocab_size=256,
        n_layer=2,
        n_head=1,
        n_kv_head=1,
        n_embd=128,
        window_pattern="L",
    )
    torch.manual_seed(42)
    with torch.device("meta"):
        baseline = GPT(baseline_config)
    baseline.to_empty(device="cpu")
    baseline.init_weights()
    torch.manual_seed(42)
    with torch.device("meta"):
        dsa = DeepSeekDSA(tiny_config())
    dsa.to_empty(device="cpu")
    dsa.init_weights()
    baseline_state = baseline.state_dict()
    dsa_state = shared_state(dsa)
    assert baseline_state.keys() == dsa_state.keys()
    for key in baseline_state:
        torch.testing.assert_close(baseline_state[key], dsa_state[key], rtol=0, atol=0)


def test_dense_warmup_logits_match_baseline():
    torch.manual_seed(11)
    baseline = GPT(GPTConfig(**{
        key: value for key, value in tiny_config().__dict__.items()
        if not key.startswith("dsa_") and key != "arch_family"
    }))
    baseline.init_weights()
    dsa = DeepSeekDSA(tiny_config())
    dsa.init_weights()
    dsa.load_state_dict({**dsa.state_dict(), **baseline.state_dict()})
    dsa.set_training_step(0)
    baseline.eval()
    dsa.eval()
    tokens = torch.randint(0, 256, (2, 8))
    torch.testing.assert_close(dsa(tokens), baseline(tokens), rtol=2e-5, atol=2e-5)


def test_sparse_attention_matches_explicit_selected_reference_and_is_causal():
    torch.manual_seed(7)
    config = tiny_config(sequence_len=6, dsa_top_k=2, dsa_query_chunk_size=2)
    attn = DSACausalSelfAttention(config, layer_idx=0)
    x = torch.randn(2, 6, 128, requires_grad=True)
    cos = torch.ones(1, 6, 1, 64)
    sin = torch.zeros_like(cos)
    attn.train()
    attn.set_sparse_enabled(True)

    captured = {}
    original_forward = attn.indexer.forward

    def capture(*args, **kwargs):
        indices, kl, mass = original_forward(*args, **kwargs)
        captured["indices"] = indices.detach()
        return indices, kl, mass

    attn.indexer.forward = capture
    actual = attn(x, None, (cos, sin), (6, 0), None)
    indices = captured["indices"]
    query_positions = torch.arange(6)[None, :, None]
    assert torch.all(indices[indices <= query_positions] <= query_positions.expand_as(indices)[indices <= query_positions])

    q = attn.c_q(x).view(2, 6, 1, 128)
    k = attn.c_k(x).view(2, 6, 1, 128)
    v = attn.c_v(x).view(2, 6, 1, 128)
    q = torch.nn.functional.rms_norm(q, (128,)) * 1.2
    k = torch.nn.functional.rms_norm(k, (128,)) * 1.2
    expected_rows = []
    for batch in range(2):
        rows = []
        for position in range(6):
            selected = indices[batch, position]
            selected = selected[selected <= position]
            logits = (q[batch, position, 0] @ k[batch, selected, 0].T) / math.sqrt(128)
            mixed = logits.softmax(dim=-1) @ v[batch, selected, 0]
            rows.append(mixed)
        expected_rows.append(torch.stack(rows))
    expected = attn.c_proj(torch.stack(expected_rows))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    actual.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_indexer_kl_is_detached_from_backbone_but_trains_indexer():
    torch.manual_seed(5)
    config = tiny_config(sequence_len=5, dsa_top_k=2)
    attn = DSACausalSelfAttention(config, layer_idx=0)
    attn.train()
    x = torch.randn(1, 5, 128, requires_grad=True)
    cos = torch.ones(1, 5, 1, 64)
    sin = torch.zeros_like(cos)
    attn(x, None, (cos, sin), (5, 0), None)
    kl = attn.last_indexer_kl
    x_grad, main_grad, index_grad = torch.autograd.grad(
        kl,
        (x, attn.c_q.weight, attn.indexer.q_proj.weight),
        allow_unused=True,
    )
    assert x_grad is None
    assert main_grad is None
    assert index_grad is not None and torch.isfinite(index_grad).all()


def test_optimizer_group_and_phase_lrs():
    model = DeepSeekDSA(tiny_config())
    model.init_weights()
    optimizer = model.setup_optimizer()
    groups = [group for group in optimizer.param_groups if group.get("dsa_indexer")]
    assert len(groups) == 1
    group = groups[0]
    assert group["kind"] == "adamw"
    assert group["betas"] == (0.8, 0.95)
    assert group["weight_decay"] == 0.0
    assert math.isclose(group["dsa_warmup_lr"], 1e-3)
    assert math.isclose(group["dsa_sparse_lr"], 7.3e-6)
    model.set_training_step(1)
    assert model.get_architecture_state()["phase"] == "dense_warmup"
    model.set_training_step(2)
    assert model.get_architecture_state()["phase"] == "sparse"
    assert "training_step" not in inspect.getsource(DeepSeekDSA.forward)


def test_config_round_trip_and_legacy_loading():
    config = tiny_config()
    restored, restored_config = build_model_from_config_kwargs(config.__dict__)
    assert isinstance(restored, DeepSeekDSA)
    assert restored_config.dsa_top_k == 2
    legacy = GPTConfig(
        sequence_len=8,
        vocab_size=256,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=128,
        window_pattern="L",
    )
    model, restored_legacy = build_model_from_config_kwargs(legacy.__dict__)
    assert isinstance(model, GPT)
    assert restored_legacy.arch_family == "nanochat"
