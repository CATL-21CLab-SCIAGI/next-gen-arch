"""Compare shared masks with the publisher's independent construction."""

import copy
import json
import sys
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import pytest
import torch

from archlab.architectures.limite_adapter import (
    LimiteAdapterConfig,
    install_adapters,
    set_trainable_mode,
)
from archlab.architectures.limite_loader import upstream_classes
from archlab.automodel.limite_adapter_compilation import configure_compilation


@pytest.fixture
def native_pair():
    pytest.importorskip("transformers")
    snapshot = Path("/mnt/oss/models/limite-1b-base-cc612bafcd4a")
    if not snapshot.exists():
        pytest.skip("verified native snapshot is not installed")
    config_class, model_class = upstream_classes(snapshot)
    spec = json.loads((snapshot / "config.json").read_text())
    spec.update(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        padded_vocab_size=64,
        tokenizer_vocab_size=64,
        ve_dim=8,
        ve_stored_heads=2,
        ve_layers=[1],
        ve_gate_channels=4,
        attn_gate_channels=8,
        xsa_layers=list(range(4)),
        global_layers=[3],
        sliding_window=3,
        rope_n_pairs=2,
        mudd=False,
        mudd_mlp=False,
        mudd_layers=[],
        mudd_at=[],
        mudd_tap_idx={},
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config = config_class(**spec)
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng():
        torch.manual_seed(42)
        base = model_class(config)
        small = {
            name: parameter.detach().clone()
            for name, parameter in base.named_parameters()
            if not name.endswith("weight")
        }
        base.to(dtype=torch.bfloat16)
        for name, parameter in base.named_parameters():
            if name in small:
                parameter.data = small[name]
        other = copy.deepcopy(base)
        original = install_adapters(base, LimiteAdapterConfig())
        candidate = install_adapters(other, LimiteAdapterConfig())
        # Exercise attention/gate gradients beyond the initial zero O matrix.
        for adapter in original.model.adapters:
            torch.nn.init.normal_(adapter.native.o_proj.weight, std=0.02)
        candidate.load_state_dict(original.state_dict())
    set_trainable_mode(original, "full")
    set_trainable_mode(candidate, "full")
    return original, candidate, sys.modules[model_class.__module__]


def publisher_pass(model, ids, mask, positions, upstream, *, independently=False, use_cache=False):
    native_forward = model.model.base.forward

    def independent_base(_self, *args, **kwargs):
        # Restore the publisher's public mask/position inputs. It constructs
        # its own masks, providing an independent oracle for the shared map.
        kwargs = dict(kwargs, attention_mask=mask, position_ids=positions)
        return native_forward(*args, **kwargs)

    if independently:
        model.model.base.forward = MethodType(independent_base, model.model.base)
    try:
        model.zero_grad(set_to_none=True)
        with (
            patch.object(upstream, "create_causal_mask", wraps=upstream.create_causal_mask) as full,
            patch.object(
                upstream,
                "create_sliding_window_causal_mask",
                wraps=upstream.create_sliding_window_causal_mask,
            ) as local,
        ):
            output = model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=use_cache,
                return_dict=True,
            ).last_hidden_state
            calls = full.call_count, local.call_count
        output.float().square().sum().backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return output.detach(), gradients, calls
    finally:
        model.model.base.forward = native_forward


@pytest.mark.parametrize("case", ["unmasked", "padding", "positions"])
def test_shared_native_masks_preserve_full_weight_gradients(native_pair, case):
    original, candidate, upstream = native_pair
    ids = torch.arange(34).view(2, 17) % 63 + 1
    mask = torch.tensor([[0, 0, 0] + [1] * 14, [1] * 13 + [0] * 4]) if case == "padding" else None
    positions = torch.arange(13, 30)[None].expand(2, -1) if case == "positions" else None
    before = publisher_pass(original, ids, mask, positions, upstream, independently=True)
    after = publisher_pass(candidate, ids, mask, positions, upstream)
    torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    assert before[1].keys() == after[1].keys()
    for name, gradient in before[1].items():
        torch.testing.assert_close(gradient, after[1][name], rtol=0, atol=0)
    assert before[2] == tuple(count + 1 for count in after[2])


def test_cached_generation_keeps_native_unpadded_classification(native_pair):
    original, candidate, _ = native_pair
    ids = torch.arange(34).view(2, 17) % 63 + 1
    original.eval()
    candidate.eval()
    with torch.no_grad():
        before = original.model(input_ids=ids, use_cache=True, return_dict=True)
        after = candidate.model(input_ids=ids, use_cache=True, return_dict=True)
        assert before.past_key_values._limite_unpadded
        assert after.past_key_values._limite_unpadded
        torch.testing.assert_close(
            before.last_hidden_state, after.last_hidden_state, rtol=0, atol=0
        )
        before = original.model(
            input_ids=ids[:, -1:],
            past_key_values=before.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        after = candidate.model(
            input_ids=ids[:, -1:],
            past_key_values=after.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        torch.testing.assert_close(
            before.last_hidden_state, after.last_hidden_state, rtol=0, atol=0
        )


@pytest.mark.parametrize("padded", [False, True])
def test_compiled_block_norm_boundaries_preserve_native_state_and_gradients(native_pair, padded):
    original, candidate, upstream = native_pair
    native_norm = upstream.rms_norm
    norm_attributes = dict(vars(native_norm))
    decoder_forward = type(candidate.model.base.layers[0]).forward
    decoder_globals = dict(decoder_forward.__globals__)
    keys = list(candidate.state_dict())
    parameters = list(candidate.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=False)
    reference_optimizer = torch.optim.AdamW(original.parameters(), lr=1e-3, foreach=False)
    calls = {"original": 0, "candidate": 0}
    handles = []

    def count(which):
        def hook(module, inputs, output):
            calls[which] += 1

        return hook

    for which, model in (("original", original), ("candidate", candidate)):
        handles.append(model.model.base.embed_tokens.register_forward_hook(count(which)))
        handles.extend(
            layer.register_forward_hook(count(which)) for layer in model.model.base.layers
        )
    old_limit = torch.compiler.config.recompile_limit
    old_accumulated = torch.compiler.config.accumulated_recompile_limit
    try:
        _, contract = configure_compilation(candidate, "strict-blocks", backend="aot_eager")
        assert contract["outer_residuals_compiled"]
        assert "native eager" in contract["block_norm_boundary"]
        assert upstream.rms_norm is native_norm
        assert vars(native_norm) == norm_attributes
        assert decoder_forward.__globals__ == decoder_globals
        assert type(candidate.model.base.layers[0]).forward is decoder_forward
        assert keys == list(candidate.state_dict())
        assert all(a is b for a, b in zip(parameters, candidate.parameters(), strict=True))
        assert all(
            a is b for a, b in zip(parameters, optimizer.param_groups[0]["params"], strict=True)
        )
        ids = torch.arange(34).view(2, 17) % 63 + 1
        mask = torch.tensor([[0, 0] + [1] * 15, [1] * 17]) if padded else None
        before = publisher_pass(original, ids, mask, None, upstream)
        after = publisher_pass(candidate, ids, mask, None, upstream)
        torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
        assert before[1].keys() == after[1].keys()
        for name, gradient in before[1].items():
            torch.testing.assert_close(gradient, after[1][name], rtol=0, atol=0)
        assert calls == {"original": 5, "candidate": 5}
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(candidate.parameters(), original.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert optimizer.state[actual].keys() == reference_optimizer.state[expected].keys()
            for name, value in optimizer.state[actual].items():
                torch.testing.assert_close(
                    value, reference_optimizer.state[expected][name], rtol=0, atol=0
                )
    finally:
        for handle in handles:
            handle.remove()
        torch.compiler.config.recompile_limit = old_limit
        torch.compiler.config.accumulated_recompile_limit = old_accumulated


def assert_same_native_pass(before, after):
    torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    assert before[1].keys() == after[1].keys()
    for name, value in before[1].items():
        torch.testing.assert_close(value, after[1][name], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["full", "adapter"])
@pytest.mark.parametrize("compiled", [False, True])
def test_static_context_preserves_native_training_and_fallbacks(native_pair, mode, compiled):
    original, candidate, upstream = native_pair
    set_trainable_mode(original, mode)
    set_trainable_mode(candidate, mode)
    wrapper = candidate.model
    assert not wrapper._native_context_cache.enabled
    parameters = list(candidate.parameters())
    keys = list(candidate.state_dict())
    handles = tuple(wrapper._handles)
    native_norm = upstream.rms_norm
    native_globals = dict(upstream.__dict__)
    norm_attributes = dict(vars(native_norm))
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=False)
    reference_optimizer = torch.optim.AdamW(original.parameters(), lr=1e-3, foreach=False)
    old_limit = torch.compiler.config.recompile_limit
    old_accumulated = torch.compiler.config.accumulated_recompile_limit
    try:
        wrapper.enable_static_training_context()
        if compiled:
            configure_compilation(candidate, "strict-blocks", backend="aot_eager")
        ids = torch.arange(34).view(2, 17) % 63 + 1
        other_ids = (ids + 7) % 63 + 1
        with patch.object(wrapper, "_masks", wraps=wrapper._masks) as masks:
            assert_same_native_pass(
                publisher_pass(original, ids, None, None, upstream),
                publisher_pass(candidate, ids, None, None, upstream),
            )
            assert masks.call_count == 1
            entry = wrapper._native_context_cache.entry
            assert entry.masks["full_attention"] is None
            assert entry.masks["sliding_attention"].dtype == torch.bool
            torch.testing.assert_close(
                entry.positions, torch.arange(17)[None].expand(2, -1), rtol=0, atol=0
            )
            wrapper.lock_static_training_context()
            assert_same_native_pass(
                publisher_pass(original, other_ids, None, None, upstream),
                publisher_pass(candidate, other_ids, None, None, upstream),
            )
            assert masks.call_count == 1
            assert wrapper._native_context_cache.entry is entry
            assert wrapper._context["masks"] is entry.masks
            with pytest.raises(RuntimeError, match="before CUDA capture"):
                wrapper.prepare_static_training_context(ids[:, :-1])
            assert masks.call_count == 1
            wrapper.lock_static_training_context(False)
            optimizer.step()
            reference_optimizer.step()
            for actual, expected in zip(candidate.parameters(), original.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert optimizer.state[actual].keys() == reference_optimizer.state[expected].keys()
                for name, value in optimizer.state[actual].items():
                    torch.testing.assert_close(
                        value, reference_optimizer.state[expected][name], rtol=0, atol=0
                    )
            # These paths must invoke the original native builders, even with
            # a prepared static training context present.
            fallback_cases = (
                (torch.tensor([[0, 0] + [1] * 15, [1] * 17]), None, False),
                (None, torch.arange(13, 30)[None].expand(2, -1), False),
                (None, None, True),
            )
            for mask, positions, use_cache in fallback_cases:
                count = masks.call_count
                assert_same_native_pass(
                    publisher_pass(original, ids, mask, positions, upstream, use_cache=use_cache),
                    publisher_pass(candidate, ids, mask, positions, upstream, use_cache=use_cache),
                )
                assert masks.call_count > count
                assert wrapper._native_context_cache.entry is entry
        original.eval()
        candidate.eval()
        assert wrapper._native_context_cache.entry is None
        with pytest.raises(RuntimeError, match="training mode"):
            wrapper.prepare_static_training_context(ids)
        with torch.no_grad():
            before = original.model(input_ids=ids, use_cache=True, return_dict=True)
            after = candidate.model(input_ids=ids, use_cache=True, return_dict=True)
            assert before.past_key_values._limite_unpadded
            assert after.past_key_values._limite_unpadded
            torch.testing.assert_close(
                before.last_hidden_state, after.last_hidden_state, rtol=0, atol=0
            )
            before = original.model(
                input_ids=other_ids[:, :1],
                past_key_values=before.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            after = candidate.model(
                input_ids=other_ids[:, :1],
                past_key_values=after.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            torch.testing.assert_close(
                before.last_hidden_state, after.last_hidden_state, rtol=0, atol=0
            )
            before = original.model(input_ids=ids, use_cache=False, return_dict=True)
            after = candidate.model(input_ids=ids, use_cache=False, return_dict=True)
            torch.testing.assert_close(
                before.last_hidden_state, after.last_hidden_state, rtol=0, atol=0
            )
        assert wrapper._native_context_cache.entry is None
        assert keys == list(candidate.state_dict())
        assert all(a is b for a, b in zip(parameters, candidate.parameters(), strict=True))
        assert handles == tuple(wrapper._handles)
        assert upstream.rms_norm is native_norm and vars(native_norm) == norm_attributes
        assert all(upstream.__dict__[key] is value for key, value in native_globals.items())
        assert torch.finfo(wrapper.base.embed_tokens.weight.dtype).eps == 0.0078125
    finally:
        torch.compiler.config.recompile_limit = old_limit
        torch.compiler.config.accumulated_recompile_limit = old_accumulated
