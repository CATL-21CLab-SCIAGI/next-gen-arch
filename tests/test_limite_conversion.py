"""True replacement, identity initialization and useful short-axis gradients."""

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from archlab.architectures.limite_conversion import (
    LimiteConversionConfig,
    LimiteSimplicialAttention,
    conversion_metadata,
    replace_limite_attention,
)
from archlab.architectures.limite_loader import upstream_classes
from archlab.architectures.simplicial_attention import reference_simplicial


@pytest.fixture
def native():
    pytest.importorskip("transformers")
    snapshot = Path(os.environ.get("LIMITE_BASE_MODEL", "/mnt/oss/models/limite-1b-base-cc612bafcd4a"))
    if not snapshot.exists():
        pytest.skip("verified publisher fixture unavailable")
    config_class, model_class = upstream_classes(snapshot)
    config = config_class.from_pretrained(snapshot, local_files_only=True)
    for name, value in dict(
        hidden_size=32, intermediate_size=64, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=37, padded_vocab_size=37,
        max_position_embeddings=128, sliding_window=5, _serialized_sliding_window=4,
        rope_n_pairs=2, attn_gate_channels=8, ve_dim=8, ve_stored_heads=2,
        ve_gate_channels=4, mudd_inter=4, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    ).items():
        setattr(config, name, value)
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(719)
        model = model_class(config)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.numel() == 1:
                    parameter.fill_(1.)
                else:
                    parameter.normal_(std=.1)
    return model


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_conversion_selects_wide_shared_gradient_accumulation(monkeypatch):
    from archlab.architectures.limite_conversion import _joint_reduce

    calls = []

    def kernel(q, k, v, **kwargs):
        calls.append(kwargs)
        return q.float()

    monkeypatch.setitem(sys.modules, "archlab.architectures.tilelang_attention", SimpleNamespace(
        tilelang_attention=kernel, tilelang_decode=kernel,
    ))
    module = SimpleNamespace(
        _archlab_conversion_config=LimiteConversionConfig(backend="tilelang"),
        _archlab_conversion_short=(torch.ones(1, 3, 2, 128),) * 2,
        _archlab_conversion_decode=False, is_global=True,
    )
    q, k, v = [torch.ones(1, heads, 3, 128) for heads in (16, 2, 2)]
    output, _ = _joint_reduce(module, q, k, v, None, scaling=.1)
    assert output.shape == (1, 3, 16, 128)
    assert calls[0]["shared_gradient_dtype"] == torch.float64
    assert calls[0]["dynamic_length"] is True


@pytest.mark.parametrize("short,long", [(1, 1), (5, 5), (16, 19)])
def test_float64_identity_matches_attention_outputs_and_inherited_gradients(short, long):
    generator = torch.Generator().manual_seed(17)
    q, k, v = [
        torch.randn(2, 19, heads, 128, generator=generator, dtype=torch.float64,
                    requires_grad=True)
        for heads in (10, 2, 2)
    ]
    ones = torch.ones_like(k)
    actual = reference_simplicial(q * (.1 * 128 ** .5), ones, k, ones, v, short, long)
    kg, vg = [value.repeat_interleave(5, dim=2) for value in (k, v)]
    expected = []
    for i in range(q.shape[1]):
        left = max(0, i - long + 1)
        scores = torch.einsum("bhd,bjhd->bhj", q[:, i], kg[:, left:i + 1]) * .1
        expected.append(torch.einsum("bhj,bjhd->bhd", scores.softmax(-1), vg[:, left:i + 1]))
    expected = torch.stack(expected, dim=1)
    cotangent = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    actual_gradients = torch.autograd.grad(actual, (q, k, v), cotangent)
    expected_gradients = torch.autograd.grad(expected, (q, k, v), cotangent)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for got, want in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-11)


def test_all_48_native_attentions_replaced_without_preludes_or_changed_source_tensors(native):
    original = dict(native.named_parameters())
    backbone, mlps = native.model, [layer.mlp for layer in native.model.layers]
    converted = replace_limite_attention(native, LimiteConversionConfig(backend="reference"))
    assert converted is native and converted.model is backbone
    assert len(backbone.layers) == 48
    assert not hasattr(backbone, "adapters") and not backbone.embed_tokens._forward_hooks
    assert backbone.config._attn_implementation == "sdpa"
    actual = dict(converted.named_parameters())
    for name, parameter in original.items():
        renamed = name.replace(".self_attn.", ".self_attn.native.")
        assert actual[renamed] is parameter, name
    for layer, mlp in zip(backbone.layers, mlps, strict=True):
        attention = layer.self_attn
        assert isinstance(attention, LimiteSimplicialAttention)
        assert attention.short_kv_delta.count_nonzero() == 0
        assert layer.mlp is mlp and not layer._forward_hooks
    metadata = conversion_metadata(converted)
    assert metadata["layers"] == list(range(48))
    assert metadata["intervention"] == "replace_attention"
    assert not metadata["cached_decode_qualified"]
    assert json.loads(json.dumps(metadata)) == metadata
    with pytest.raises(ValueError, match="already converted"):
        replace_limite_attention(converted, LimiteConversionConfig(backend="reference"))


def test_full_publisher_training_logits_identity_with_local_global_ve_xsa_and_mudd(native):
    teacher = copy.deepcopy(native).train()
    student = replace_limite_attention(native, LimiteConversionConfig(backend="reference")).train()
    ids = torch.tensor([[1, 4, 8, 5, 11, 2, 17], [1, 3, 9, 6, 10, 2, 18]])
    expected = teacher(input_ids=ids, use_cache=False).logits
    actual = student(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-6)
    assert all(layer.self_attn.native._archlab_conversion_short is None
               for layer in student.model.layers)


@pytest.mark.parametrize("index", [1, 3])
def test_native_attention_eval_identity_preserves_value_embeddings_xsa_and_gates(native, index):
    teacher = copy.deepcopy(native.model.layers[index].self_attn).eval()
    student = LimiteSimplicialAttention(
        native.model.layers[index].self_attn, LimiteConversionConfig(backend="reference"),
    ).eval()
    hidden = torch.randn(2, 7, 32)
    values = torch.randn(2, 7, 2, 8)
    cosine, sine = native.model.rotary_emb(torch.arange(7)[None].expand(2, -1))
    positions = torch.arange(7)
    age = positions[:, None] - positions[None, :]
    allowed = age >= 0
    if not teacher.is_global:
        allowed &= age < teacher.window_span
    mask = allowed[None, None]
    arguments = (hidden, values, cosine, sine, mask, None, False, False)
    expected, _ = teacher(*arguments)
    actual, _ = student(*arguments)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_cross_entropy_activates_both_short_axes_in_every_layer(native):
    student = replace_limite_attention(native, LimiteConversionConfig(backend="reference")).train()
    ids = torch.tensor([[1, 4, 8, 5, 11, 2, 17]])
    loss = student(input_ids=ids, labels=ids, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    for index, layer in enumerate(student.model.layers):
        attention = layer.self_attn
        for branch in attention.short_kv_delta.grad.split(attention.native.kv_size):
            assert torch.isfinite(branch).all(), index
            assert branch.norm() > 0, index
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in student.parameters())


def test_state_and_optimizer_roundtrip_replays_the_next_update(tmp_path, native):
    student = replace_limite_attention(native, LimiteConversionConfig(backend="reference")).train()
    fresh = copy.deepcopy(student)
    ids = torch.tensor([[1, 4, 8, 5, 11, 2, 17]])
    optimizer = torch.optim.SGD(student.parameters(), lr=.001, momentum=.9)

    def step(model, opt):
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        opt.step()
        return loss.detach()

    step(student, optimizer)
    checkpoint = tmp_path / "conversion.pt"
    torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict(),
                "conversion": conversion_metadata(student)}, checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    fresh.load_state_dict(saved["model"], strict=True)
    resumed = torch.optim.SGD(fresh.parameters(), lr=.001, momentum=.9)
    resumed.load_state_dict(saved["optimizer"])
    assert conversion_metadata(fresh) == saved["conversion"]
    torch.testing.assert_close(step(fresh, resumed), step(student, optimizer), rtol=0, atol=0)
    for name, value in fresh.state_dict().items():
        torch.testing.assert_close(value, student.state_dict()[name], rtol=0, atol=0)
    assert fresh.get_input_embeddings().weight is fresh.get_output_embeddings().weight


def test_padding_and_cached_decode_fail_explicitly(native):
    student = replace_limite_attention(native, LimiteConversionConfig(backend="reference"))
    ids = torch.tensor([[1, 4, 8, 5, 2]])
    with pytest.raises(ValueError, match="cached decoding"):
        student(input_ids=ids, use_cache=True)
    with pytest.raises(ValueError, match="padding|causal attention mask"):
        student(input_ids=ids, attention_mask=torch.tensor([[1, 1, 1, 0, 0]]), use_cache=False)


@pytest.mark.parametrize("short_window", [3, 16])
def test_nonzero_short_axes_dynamic_cache_matches_every_full_prefix_and_window_boundary(native, short_window):
    student = replace_limite_attention(native, LimiteConversionConfig(
        backend="reference", short_window=short_window,
    )).train()
    with torch.no_grad():
        for layer in student.model.layers:
            layer.self_attn.short_kv_delta.normal_(std=.02)
        ids = torch.tensor([[1, 4, 8, 5, 11, 2, 17, 6, 7, 9, 13, 14, 18, 21, 25, 19, 22, 28, 3, 12, 23]])
        prefill = student(input_ids=ids[:, :3], use_cache=True)
        cache = prefill.past_key_values
        expected = student(input_ids=ids[:, :3], use_cache=False).logits
        torch.testing.assert_close(prefill.logits, expected, rtol=1e-6, atol=2e-6)
        for position in range(3, ids.shape[1]):
            actual = student(input_ids=ids[:, position:position + 1], past_key_values=cache,
                             use_cache=True).logits
            expected = student(input_ids=ids[:, :position + 1], use_cache=False).logits[:, -1:]
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-6)
        assert set(cache.archlab_conversion_short) == set(range(48))
        for index, (key, value) in cache.archlab_conversion_short.items():
            window = short_window if index in student.config.global_layers else min(short_window, 5)
            assert key.shape == value.shape == (1, window, 2, 8)
        assert not hasattr(cache, "archlab_native_preludes")
        length = cache.get_seq_length()
        with pytest.raises(ValueError, match="full prefill then single-token"):
            student(input_ids=ids[:, :2], past_key_values=cache, use_cache=True)
        assert cache.get_seq_length() == length


def test_dynamic_cache_short_history_is_owned_by_each_request(native):
    student = replace_limite_attention(native, LimiteConversionConfig(backend="reference")).train()
    with torch.no_grad():
        first = student(input_ids=torch.tensor([[1, 4, 8]]), use_cache=True).past_key_values
        second = student(input_ids=torch.tensor([[1, 9, 2]]), use_cache=True).past_key_values
        before = {index: tuple(value.clone() for value in pair)
                  for index, pair in second.archlab_conversion_short.items()}
        student(input_ids=torch.tensor([[7]]), past_key_values=first, use_cache=True)
        assert first.archlab_conversion_short is not second.archlab_conversion_short
        for index, pair in second.archlab_conversion_short.items():
            for value, original in zip(pair, before[index], strict=True):
                torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("values", [{"short_window": 0}, {"short_window": True},
                                   {"short_window": 17}, {"backend": "automatic"}])
def test_invalid_conversion_configuration(values):
    with pytest.raises(ValueError):
        LimiteConversionConfig(**values)
