import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from archlab.automodel.limite_performance import baseline_flops, causal_pairs


def native_config():
    return SimpleNamespace(
        hidden_size=1280,
        num_hidden_layers=48,
        num_attention_heads=10,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=3328,
        global_layers=list(range(3, 48, 4)),
        sliding_window=1024,
        attn_gate_channels=128,
        ve_layers=list(range(1, 48, 3)),
        ve_gate_channels=12,
        vocab_size=151680,
        mudd=True,
        mudd_mlp=True,
        mudd_inter=32,
        mudd_layers=[24, 47],
        mudd_tap_idx={"24": [0, 12, 24], "47": [0, 23, 47]},
    )


@pytest.mark.parametrize("length,span", [(1, None), (9, None), (9, 3), (3, 9)])
def test_pairs_match_explicit_causal_mask(length, span):
    actual = sum(
        key <= query and (span is None or key > query - span)
        for query in range(length)
        for key in range(length)
    )
    assert causal_pairs(length, span) == actual


def test_native_operation_ledger_includes_tied_head_but_not_lookup_capacity():
    config = native_config()
    ledger = baseline_flops(config, 2048)
    assert ledger.linear_per_token == 7_112_055_552
    assert ledger.attention_per_token == 1_227_847_680
    assert ledger.per_token == 8_339_903_232
    # Vocabulary lookup capacity only changes FLOPs through the output head.
    config.vocab_size += 1
    assert baseline_flops(config, 2048).per_token - ledger.per_token == 6 * 1280


def test_frozen_weights_omit_only_the_backbone_weight_gradient_products():
    config = native_config()
    full = baseline_flops(config, 2048)
    adapter = baseline_flops(config, 2048, "adapter")
    adapter_coefficients = 377_487_360 // 2 + 122_880 // 2 + 768 // 2
    all_coefficients = 1_185_342_592
    assert full.linear_per_token - adapter.linear_per_token == 2 * (
        all_coefficients - adapter_coefficients
    )
    assert full.attention_per_token == adapter.attention_per_token
    target_seconds = 128 * 2048 * full.per_token / (16 * 2.25e15 * 0.5)
    assert full.mfu(128 * 2048, target_seconds, 16) == pytest.approx(0.5)
    assert target_seconds == pytest.approx(0.12145864404718934)


def test_loaded_runtime_span_matches_portable_serialized_window():
    serialized = native_config()
    loaded = native_config()
    loaded._serialized_sliding_window = loaded.sliding_window
    loaded.sliding_window += 1
    assert baseline_flops(loaded, 2048) == baseline_flops(serialized, 2048)
    assert causal_pairs(2048, loaded.sliding_window) == 1_574_400


def test_loaded_runtime_window_override_counts_current_mask():
    config = native_config()
    config._serialized_sliding_window = config.sliding_window
    config.sliding_window = 17
    length = 37
    local = sum(
        key <= query and key > query - 17 for query in range(length) for key in range(length)
    )
    global_ = sum(key <= query for query in range(length) for key in range(length))
    ledger = baseline_flops(config, length)
    assert ledger.attention_per_token == 12 * 10 * 128 * 2 * (36 * local + 12 * global_) / length


@pytest.mark.parametrize("length,runtime_window", [(2048, None), (37, 17)])
def test_actual_publisher_mask_matches_loaded_window_ledger(length, runtime_window):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    snapshot = Path("/mnt/oss/models/limite-1b-base-cc612bafcd4a")
    if not snapshot.exists():
        pytest.skip("verified native publisher snapshot is not installed")
    from archlab.architectures.limite_loader import upstream_classes

    config_class, model_class = upstream_classes(snapshot)
    config = config_class(**json.loads((snapshot / "config.json").read_text()))
    config._attn_implementation = "sdpa"
    assert config._serialized_sliding_window == 1024
    assert config.sliding_window == 1025
    if runtime_window is not None:
        config.sliding_window = runtime_window
    upstream = sys.modules[model_class.__module__]
    mask = upstream.create_sliding_window_causal_mask(
        config=config,
        inputs_embeds=torch.empty((1, length, 0), dtype=torch.bfloat16),
        attention_mask=None,
        past_key_values=None,
        position_ids=torch.arange(length)[None],
    )
    local = int(mask[0, 0].sum())
    global_ = length * (length + 1) // 2
    globals_ = len(config.global_layers)
    pair_count = 2 * (globals_ * global_ + (config.num_hidden_layers - globals_) * local)
    assert baseline_flops(config, length).attention_per_token == (
        12 * config.num_attention_heads * config.head_dim * pair_count / length
    )
    if runtime_window is None:
        assert local == 1_574_400
        reloaded = config_class(**config.to_dict())
        assert baseline_flops(reloaded, length) == baseline_flops(config, length)


def test_invalid_measurements_fail_instead_of_fabricating_mfu():
    ledger = baseline_flops(native_config(), 2048)
    with pytest.raises(ValueError):
        ledger.mfu(1, 0, 1)
    with pytest.raises(ValueError):
        causal_pairs(0)
