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
