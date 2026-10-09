from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from archlab.architectures.limite_replay import (
    NativeReplayMask,
    _bounded_sdpa,
    _ReplayInterface,
    enable_native_replay,
    replay_attention,
)


@pytest.mark.parametrize("window", [None, 1, 5, 19])
@pytest.mark.parametrize("backend", ["sdpa", "sdpa_bounded", "sdpa_native"])
def test_replay_reduction_matches_explicit_gqa_scores_and_gradients(window, backend):
    torch.manual_seed(41)
    q = torch.randn(2, 6, 19, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 19, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    age = torch.arange(19)[:, None] - torch.arange(19)[None, :]
    allowed = (age >= 0) & (age < (window or 19))
    score = q @ k.repeat_interleave(3, dim=1).transpose(-1, -2) * .1
    expected = (score.masked_fill(~allowed, -torch.inf).softmax(-1)
                @ v.repeat_interleave(3, dim=1)).transpose(1, 2)
    actual = replay_attention(q, k, v, scaling=.1, window_span=window, backend=backend)
    grad = torch.randn_like(expected)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(torch.autograd.grad((actual * grad).sum(), (q, k, v)),
                         torch.autograd.grad((expected * grad).sum(), (q, k, v)), strict=True):
        torch.testing.assert_close(got, want)


def test_bounded_sdpa_multiple_tiles_preserve_window_and_checkpoint_gradients():
    torch.manual_seed(41)
    q = torch.randn(2, 6, 19, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 19, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    actual = _bounded_sdpa(q, k, v, scaling=.1, window_span=5, query_tile=3).transpose(1, 2)
    expected = replay_attention(q, k, v, scaling=.1, window_span=5, backend="sdpa")
    grad = torch.randn_like(actual)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(torch.autograd.grad((actual * grad).sum(), (q, k, v)),
                         torch.autograd.grad((expected * grad).sum(), (q, k, v)), strict=True):
        torch.testing.assert_close(got, want)


def test_descriptor_preserves_original_decode_interface_and_rejects_changed_window():
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "decode", None

    interface = _ReplayInterface(SimpleNamespace(get_interface=lambda *_: original), "sdpa")
    reduce = interface.get_interface("sdpa", None)
    module = SimpleNamespace(is_global=False, window_span=5)
    q = torch.randn(1, 2, 3, 8)
    assert reduce(module, q, q, q, None, scaling=.1) == ("decode", None)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="contract changed"):
        reduce(module, q, q, q, NativeReplayMask(4), scaling=.1)
    with pytest.raises(ValueError, match="contract changed"):
        reduce(module, q, q, q, NativeReplayMask(5), scaling=.1, dropout=.1)


def test_explicit_uncached_replay_does_not_require_optional_config_use_cache():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList()
            self.config = SimpleNamespace(max_position_embeddings=128, sliding_window=5)

        def forward(self, **kwargs):
            return kwargs

    model = torch.nn.Module()
    model.model = Backbone()
    model.archlab_native_checkpoint = True
    enable_native_replay(model, attention_backend="sdpa")
    ids = torch.tensor([[1, 2, 3]])
    result = model.model(input_ids=ids, use_cache=False)
    assert result["attention_mask"] == {
        "full_attention": NativeReplayMask(None), "sliding_attention": NativeReplayMask(5),
    }
    assert torch.equal(result["position_ids"], torch.arange(3)[None])
    for kwargs in ({}, {"use_cache": True}, {"use_cache": None}):
        assert model.model(input_ids=ids, **kwargs) == dict(input_ids=ids, **kwargs)


def tiny_publisher():
    pytest.importorskip("transformers")
    from archlab.architectures.limite_loader import upstream_classes

    snapshot = Path("/mnt/oss/models/limite-1b-violetto-4cf321846e47")
    if not snapshot.exists():
        pytest.skip("verified publisher fixture unavailable")
    config_class, model_class = upstream_classes(snapshot)
    config = config_class.from_pretrained(snapshot, local_files_only=True)
    for name, value in dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        vocab_size=37, padded_vocab_size=37, max_position_embeddings=128,
        global_layers=[1, 3], layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=5, rope_n_pairs=2, attn_gate_channels=8,
        ve_dim=8, ve_stored_heads=2, ve_layers=[1, 2], ve_gate_channels=4,
        xsa_layers=[0, 1, 2, 3], mudd_layers=[2, 3], mudd_at=[2, 3],
        mudd_tap_idx={"2": [0, 1, 2], "3": [0, 2, 3]}, mudd_inter=4,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    ).items():
        setattr(config, name, value)
    config._attn_implementation = "sdpa"
    model = model_class(config)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel() == 1:
                parameter.fill_(1.)
            else:
                parameter.normal_(std=.1)
    model.archlab_native_checkpoint = True
    return model


@pytest.mark.parametrize("checkpoint_layers", [False, True])
@pytest.mark.parametrize("backend", ["sdpa", "sdpa_bounded", "sdpa_native"])
def test_publisher_forward_backward_and_state_keys_unchanged(checkpoint_layers, backend):
    torch.manual_seed(13)
    expected = tiny_publisher().train()
    actual = deepcopy(expected)
    before = {key: value.clone() for key, value in actual.state_dict().items()}
    parameter_ids = [id(p) for p in actual.parameters()]
    enable_native_replay(actual, attention_backend=backend, checkpoint_layers=checkpoint_layers)
    assert actual.state_dict().keys() == before.keys()
    assert [id(p) for p in actual.parameters()] == parameter_ids
    for key, value in actual.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    tokens = torch.randint(1, 37, (2, 19))
    values = []
    for model in (expected, actual):
        output = model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                             use_cache=False, return_dict=True).last_hidden_state
        values.append(output)
        (output.square().mean() + output[..., ::2].mean()).backward()
    torch.testing.assert_close(values[1], values[0], rtol=1e-5, atol=1e-6)
    for (name, left), (other_name, right) in zip(expected.named_parameters(), actual.named_parameters(), strict=True):
        assert name == other_name
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, rtol=2e-4, atol=2e-6)


def test_publisher_replay_is_unpadded_and_decode_binding_survives():
    from archlab.architectures.limite_gqa import set_native_decode_gqa

    model = tiny_publisher()
    set_native_decode_gqa(model)
    enabled = enable_native_replay(model, attention_backend="sdpa")
    assert enabled is model
    assert all(layer.self_attn._archlab_decode_gqa for layer in model.model.layers)
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="unpadded"):
        model.model(input_ids=ids, attention_mask=torch.tensor([[0, 1, 1]]), use_cache=False)
    with pytest.raises(ValueError, match="contiguous positions"):
        model.model(input_ids=ids, position_ids=torch.tensor([[0, 2, 4]]), use_cache=False)
    with pytest.raises(ValueError, match="already configured"):
        enable_native_replay(model)
    # Both cache creation and subsequent decode must use publisher cache logic.
    # Publisher eval MUDD folds assume native BF16 activations. The training
    # oracle deliberately starts FP32; restore native matrix dtypes here.
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.to(dtype=torch.bfloat16)
    model.eval()
    with torch.no_grad():
        first = model.model(input_ids=ids, use_cache=True, return_dict=True)
        second = model.model(input_ids=torch.tensor([[4]]), past_key_values=first.past_key_values,
                             use_cache=True, return_dict=True)
    assert second.last_hidden_state.shape == (1, 1, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="B300 official FA4 qualification")
@pytest.mark.parametrize("window", [None, 1025])
@pytest.mark.parametrize("backend", ["fa4", "sdpa_bounded", "sdpa_native"])
def test_cuda_replay_matches_sdpa_forward_and_backward(window, backend):
    torch.manual_seed(43)
    q = torch.randn(1, 10, 2053, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 2, 2053, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    expected = replay_attention(q, k, v, scaling=.1, window_span=window, backend="sdpa")
    actual = replay_attention(q, k, v, scaling=.1, window_span=window, backend=backend)
    grad = torch.randn_like(actual)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=.03, atol=.015)
    for got, want in zip(torch.autograd.grad((actual * grad).sum(), (q, k, v)),
                         torch.autograd.grad((expected * grad).sum(), (q, k, v)), strict=True):
        relative_rms = ((got.float() - want.float()).square().mean()
                        / want.float().square().mean().clamp_min(1e-12)).sqrt()
        assert float(relative_rms) < .04
        assert torch.isfinite(got).all()
