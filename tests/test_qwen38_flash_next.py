from dataclasses import replace
from types import ModuleType

import torch

import archlab.architectures.qwen38_flash_next as qwen38
from archlab.architectures.qwen38_flash_next import (
    ChunkedGroupedLinear,
    NativeGroupedLinear,
    NativeLinear,
    Qwen38FlashNext,
    Qwen38FlashNextConfig,
    SingleStreamResidual,
    SparseMoE,
    _pad_grouped_tokens,
)


def tiny_config(**overrides):
    config = Qwen38FlashNextConfig(
        vocab_size=64,
        sequence_len=8,
        max_position_embeddings=32,
        num_hidden_layers=4,
        hidden_size=32,
        full_attention_interval=4,
        attention_heads=4,
        attention_kv_heads=1,
        attention_head_dim=8,
        indexer_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=4,
        linear_qk_heads=2,
        linear_v_heads=4,
        linear_key_dim=4,
        linear_value_dim=4,
        num_experts=8,
        num_experts_per_token=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        residual_low_rank=8,
        ngram_vocab_size=31,
        ngram_heads_per_order=2,
        ngram_embedding_dim=32,
    )
    return replace(config, **overrides)


def test_quarter_shape_contract_matches_agreed_source_scaling():
    config = Qwen38FlashNextConfig()

    assert config.num_hidden_layers == 48 // 4
    assert config.hidden_size == 2_560 // 4
    assert config.num_experts == 512 // 4
    assert config.num_experts_per_token == 3
    assert config.moe_intermediate_size == 640 // 4
    assert config.attention_heads == 24 // 4
    assert config.attention_kv_heads == 1
    assert config.attention_head_dim == 256 // 4
    assert config.linear_v_heads == 48 // 4
    assert config.linear_qk_heads == 16 // 4
    assert config.linear_key_dim == 128 // 4
    assert config.residual_streams == 1
    assert config.residual_low_rank == 320 // 4
    assert config.ngram_vocab_size == 20_000_000 // 4
    assert config.ngram_embedding_dim == 2_560 // 4
    assert config.vocab_size == 248_320
    assert config.mtp_layers == 1


def test_layer_pattern_preserves_three_gdn_then_one_qsa():
    model = Qwen38FlashNext(tiny_config())
    assert [layer.attention_kind for layer in model.layers] == ["gdn", "gdn", "gdn", "qsa"]
    assert model.mtp_block.attention_kind == "qsa"


def test_tiny_forward_backward_and_auxiliary_loss_are_finite():
    torch.manual_seed(7)
    model = Qwen38FlashNext(tiny_config())
    model.init_weights()
    tokens = torch.randint(0, model.config.vocab_size, (2, model.config.sequence_len))
    labels = torch.roll(tokens, shifts=-1, dims=1)
    labels[:, -1] = -1

    losses = model(tokens, labels, loss_reduction="none")
    assert losses.shape == labels.shape
    assert torch.isfinite(losses).all()
    losses.mean().backward()
    assert model.layers[0].moe.router.weight.grad is not None
    assert model.layers[-1].attention.index_q.weight.grad is not None
    assert torch.isfinite(model.layers[-1].attention.index_q.weight.grad).all()


def test_default_model_parameter_contract_constructs_on_meta():
    with torch.device("meta"):
        model = Qwen38FlashNext(Qwen38FlashNextConfig())
    counts = model.num_scaling_params()

    assert counts["ngram_ple"] > 3_200_000_000
    assert 4_000_000_000 < counts["total"] < 4_500_000_000
    assert model.estimate_executed_flops() < model.estimate_flops()


def test_fp4_residual_keeps_rank_80_reduction_gates_in_bf16(monkeypatch):
    calls = []

    def fake_fp4_linear(in_features, out_features, *, runtime_backend, bias=False):
        calls.append((in_features, out_features, runtime_backend, bias))
        return NativeLinear(in_features, out_features, bias=bias)

    monkeypatch.setattr(qwen38, "_linear", fake_fp4_linear)
    residual = SingleStreamResidual(Qwen38FlashNextConfig(), runtime_backend="te_fp4")

    assert calls == [
        (640, 80, "te_fp4", False),
        (80, 640, "te_fp4", False),
        (80, 1, "te_fp4", False),
    ]
    assert isinstance(residual.read, NativeLinear)
    assert isinstance(residual.write, NativeLinear)


def test_fp4_linear_falls_back_unless_forward_and_backward_are_tile_aligned():
    assert isinstance(qwen38._linear(640, 24, runtime_backend="te_fp4"), NativeLinear)
    assert isinstance(qwen38._linear(640, 64, runtime_backend="te_fp4"), NativeLinear)
    assert isinstance(qwen38._linear(640, 80, runtime_backend="te_fp4"), NativeLinear)
    assert isinstance(qwen38._linear(640, 320, runtime_backend="te_fp4"), NativeLinear)
    assert isinstance(qwen38._linear(160, 640, runtime_backend="te_fp4"), NativeLinear)
    assert isinstance(qwen38._linear(80, 640, runtime_backend="te_fp4"), NativeLinear)
    assert isinstance(qwen38._linear(80, 1, runtime_backend="te_fp4"), NativeLinear)


def test_bf16_backend_uses_transformer_engine_for_dense_and_grouped_linears(monkeypatch):
    linear_calls = []
    grouped_calls = []

    class FakeLinear(NativeLinear):
        def __init__(self, in_features, out_features, **kwargs):
            linear_calls.append((in_features, out_features, kwargs))
            super().__init__(in_features, out_features, bias=kwargs["bias"])

    class FakeGroupedLinear(NativeGroupedLinear):
        def __init__(self, experts, in_features, out_features, **kwargs):
            grouped_calls.append((experts, in_features, out_features, kwargs))
            super().__init__(experts, in_features, out_features)

    transformer_engine = ModuleType("transformer_engine")
    fake_te = ModuleType("transformer_engine.pytorch")
    fake_te.Linear = FakeLinear
    fake_te.GroupedLinear = FakeGroupedLinear
    transformer_engine.pytorch = fake_te
    modules = __import__("sys").modules
    monkeypatch.setitem(modules, "transformer_engine", transformer_engine)
    monkeypatch.setitem(modules, "transformer_engine.pytorch", fake_te)

    qwen38._linear(640, 80, runtime_backend="te_bf16")
    grouped = qwen38._grouped_linear(128, 640, 160, runtime_backend="te_bf16")

    assert linear_calls[0][:2] == (640, 80)
    assert isinstance(grouped, FakeGroupedLinear)
    assert grouped_calls[0][:3] == (128, 640, 160)


def test_grouped_token_padding_preserves_real_rows_and_gradients():
    inputs = torch.arange(18, dtype=torch.float32).view(6, 3).requires_grad_()
    splits = torch.tensor([1, 0, 2, 3], dtype=torch.int32)

    padded, padded_splits, real_indices = _pad_grouped_tokens(inputs, splits)

    assert padded.shape == (192, 3)
    assert padded_splits.tolist() == [64, 0, 64, 64]
    assert torch.equal(padded.index_select(0, real_indices), inputs)
    padded.index_select(0, real_indices).sum().backward()
    assert torch.equal(inputs.grad, torch.ones_like(inputs))


def test_chunked_grouped_linear_preserves_expert_order_and_gradients():
    torch.manual_seed(11)
    reference = NativeGroupedLinear(5, 3, 4)
    chunked = ChunkedGroupedLinear(
        5,
        3,
        4,
        runtime_backend="native",
        max_experts_per_group=2,
    )
    with torch.no_grad():
        offset = 0
        for group in chunked.groups:
            stop = offset + group.weight.size(0)
            group.weight.copy_(reference.weight[offset:stop])
            offset = stop

    splits = torch.tensor([1, 0, 2, 3, 0], dtype=torch.int32)
    reference_inputs = torch.randn(6, 3, requires_grad=True)
    chunked_inputs = reference_inputs.detach().clone().requires_grad_()
    reference_output = reference(reference_inputs, splits)
    chunked_output = chunked(chunked_inputs, splits)

    assert chunked.group_sizes == (2, 2, 1)
    assert torch.equal(chunked_output, reference_output)
    reference_output.sum().backward()
    chunked_output.sum().backward()
    assert torch.equal(chunked_inputs.grad, reference_inputs.grad)
    assert sum(parameter.numel() for parameter in chunked.parameters()) == reference.weight.numel()


def test_fp4_grouped_linear_splits_128_experts_at_te_kernel_limit(monkeypatch):
    calls = []

    def fake_raw(experts, in_features, out_features, *, runtime_backend):
        calls.append((experts, in_features, out_features, runtime_backend))
        return NativeGroupedLinear(experts, in_features, out_features)

    monkeypatch.setattr(qwen38, "_raw_grouped_linear", fake_raw)
    module = qwen38._grouped_linear(128, 640, 320, runtime_backend="te_fp4")

    assert isinstance(module, ChunkedGroupedLinear)
    assert module.group_sizes == (64, 64)
    assert calls == [(64, 640, 320, "te_fp4"), (64, 640, 320, "te_fp4")]


def test_moe_auxiliary_loss_excludes_masked_padding_token():
    torch.manual_seed(17)
    moe = SparseMoE(tiny_config(), runtime_backend="native")
    inputs = torch.randn(1, 8, 32)
    mask = torch.tensor([[True, True, True, True, True, True, True, False]])

    moe(inputs, token_mask=mask)
    masked_aux = moe.last_aux_loss.detach().clone()
    moe(inputs[:, :-1])

    assert torch.allclose(masked_aux, moe.last_aux_loss)


def test_mtp_pads_fp4_token_axis_without_changing_loss_shape():
    torch.manual_seed(19)
    model = Qwen38FlashNext(tiny_config())
    model.fp4_token_multiple = 8
    model.init_weights()
    observed = []

    def record_shape(_module, args):
        observed.append(args[0].shape)

    handle = model.mtp_block.register_forward_pre_hook(record_shape)
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    labels = torch.roll(tokens, shifts=-1, dims=1)
    labels[:, -1] = -1
    losses = model(tokens, labels, loss_reduction="none")
    handle.remove()

    assert observed == [torch.Size([1, 8, model.config.hidden_size])]
    assert losses.shape == labels.shape
    assert torch.isfinite(losses).all()
