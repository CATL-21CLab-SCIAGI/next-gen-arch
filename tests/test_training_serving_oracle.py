"""CPU negative controls for the independent serving-precision admission oracle."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_spec = spec_from_file_location(
    "agentic_serving_oracle", Path(__file__).parent / "integration/test_agentic_training.py")
_oracle = module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


def _metric(mean, p95, maximum=.3, kl=.001):
    return dict(absolute=dict(mean=mean, maximum=maximum), p95_absolute=p95, k3_kl_estimate=kl)


def _case():
    # Fixed-token diagnosis: the old TileLang-only cache baseline omitted the
    # TileLang-versus-FA2 reduction rounding in native serving.
    return dict(
        response_logratio=_metric(.029309, .126882),
        native_cache_vs_replay_same_tokens=_metric(.022896, .079977),
        serving_precision_cache_vs_learner_replay=_metric(.028236, .104503),
        vllm_vs_serving_precision_cache=_metric(.022995, .080314),
        serving_precision_cache_vs_replay=_metric(.020462, .076176),
    )


def test_precision_change_uses_independent_native_oracle():
    case = _case()
    assert case["response_logratio"]["p95_absolute"] > (
        1.25 * case["native_cache_vs_replay_same_tokens"]["p95_absolute"] + .01)
    _oracle._assert_response_contract([case], "fixture")


@pytest.mark.parametrize("comparison", ["response_logratio", "vllm_vs_serving_precision_cache"])
@pytest.mark.parametrize("field,value", [
    ("mean", .2), ("p95_absolute", .2), ("maximum", .501),
    ("k3_kl_estimate", .00501), ("p95_absolute", float("nan")),
])
def test_either_mismatched_policy_fails(comparison, field, value):
    case = deepcopy(_case())
    metric = case[comparison]
    if field in ("mean", "maximum"):
        metric["absolute"][field] = value
    else:
        metric[field] = value
    with pytest.raises(AssertionError):
        _oracle._assert_response_contract([case], "fixture")


def test_bad_serving_cannot_hide_inside_learner_precision_gap():
    case = _case()
    case["serving_precision_cache_vs_learner_replay"] = _metric(10., 10.)
    case["vllm_vs_serving_precision_cache"] = _metric(.05, .13)
    with pytest.raises(AssertionError):
        _oracle._assert_response_contract([case], "fixture")


def test_oracle_restores_shared_configs_and_forwards_on_error():
    class Attention:
        def forward(self):
            return "publisher"

    config = SimpleNamespace(_attn_implementation="sdpa")
    attentions = [Attention(), Attention()]
    custom_forwards = [lambda: "tilelang", lambda: "backbone"]
    for attention, forward in zip(attentions, custom_forwards, strict=True):
        attention.config = config
        attention.forward = forward
        attention.qkv_proj = SimpleNamespace(weight=torch.ones(2, dtype=torch.float32))
        attention._inference_qkv_weight = torch.ones(2, dtype=torch.bfloat16)
    actor = SimpleNamespace(model=SimpleNamespace(
        adapters=[SimpleNamespace(native=attentions[0])],
        base=SimpleNamespace(config=config, layers=[SimpleNamespace(self_attn=attentions[1])]),
    ))
    with pytest.raises(RuntimeError, match="control stopped"):
        with _oracle._native_serving_precision(actor, dict(backend="FLASH_ATTN", flash_attn_version=2)):
            assert config._attn_implementation == "flash_attention_2"
            assert all(attention.forward() == "publisher" for attention in attentions)
            raise RuntimeError("control stopped")
    assert config._attn_implementation == "sdpa"
    assert all(attention.forward is forward for attention, forward in
               zip(attentions, custom_forwards, strict=True))
    assert all(torch.equal(attention.qkv_proj.weight, torch.ones(2, dtype=torch.float32))
               for attention in attentions)
    attentions[0]._inference_qkv_weight = torch.ones(2, dtype=torch.float32)
    with pytest.raises(ValueError, match="BF16 cached QKV"):
        with _oracle._native_serving_precision(actor, dict(backend="FLASH_ATTN", flash_attn_version=2)):
            pytest.fail("FP32 projection arithmetic admitted")
    with pytest.raises(ValueError, match="explicit FlashAttention-2"):
        with _oracle._native_serving_precision(actor, dict(backend="FLASH_ATTN", flash_attn_version=3)):
            pytest.fail("unsupported oracle admitted")


def test_serving_oracle_retains_keys_past_local_window():
    pytest.importorskip("transformers")
    from transformers import DynamicCache

    class Actor:
        window = 1024

        def __call__(self, input_ids, past_key_values, use_cache):
            assert use_cache and isinstance(past_key_values, DynamicCache)
            keys = input_ids[:, None, :, None].float()
            for cache in (past_key_values, past_key_values.archlab_native_preludes):
                cache.update(keys, keys, 0)
            self.cache = past_key_values
            return SimpleNamespace(logits=torch.zeros(1, 1, 3), past_key_values=past_key_values)

    actor = Actor()
    ids = torch.zeros(1, 1102, dtype=torch.long)
    values = _oracle._cached_response_logprobs(actor, ids, 1100, [1, 2], full_history=True)
    assert len(values) == 2 and all(torch.isfinite(torch.tensor(values)))
    assert actor.cache.get_seq_length() == 1101
    assert actor.cache.archlab_native_preludes.get_seq_length() == 1101
    assert actor.window == 1024


def _conversion_case():
    case = _case()
    case["conversion_cache"] = dict(logits_relative_l2=.0713, cached_argmax_agreement=1.)
    case["original_base_control_cache_vs_replay"] = _metric(.04, .12, maximum=.4778)
    checks = [dict(kind=kind, layer=layer, relative_l2=.0003)
              for kind in ("prefill_last_query_vs_decode", "paged_decode_vs_contiguous_history")
              for layer in range(48)]
    evidence = dict(conversion_raw_reduction=[dict(
        owners=list(range(96)), live_slots=[dict(exact_values_preserved=True)] * 96,
        live_slot_byte_ranges_disjoint=True, raw_checks=checks)])
    return case, evidence


def test_conversion_release_contract_is_explicitly_control_calibrated():
    case, evidence = _conversion_case()
    # The historical full-vocabulary .02 check remains a failed diagnostic.
    assert case["conversion_cache"]["logits_relative_l2"] > .02
    _oracle._assert_response_contract([case], evidence)


@pytest.mark.parametrize("failure", ["control", "argmax", "raw", "missing_layer", "cache_alias", "cache_value"])
def test_conversion_control_and_raw_cache_failures_are_not_admitted(failure):
    case, evidence = _conversion_case()
    audit = evidence["conversion_raw_reduction"][0]
    if failure == "control":
        case["original_base_control_cache_vs_replay"] = _metric(0., 0.)
    elif failure == "argmax":
        case["conversion_cache"]["cached_argmax_agreement"] = .99
    elif failure == "raw":
        audit["raw_checks"][0]["relative_l2"] = .00101
    elif failure == "missing_layer":
        audit["raw_checks"].pop()
    elif failure == "cache_alias":
        audit["live_slot_byte_ranges_disjoint"] = False
    else:
        audit["live_slots"][0]["exact_values_preserved"] = False
    with pytest.raises(AssertionError):
        _oracle._assert_response_contract([case], evidence)


def test_conversion_oracle_keeps_tilelang_and_absolute_cached_positions():
    attention = SimpleNamespace(conversion_config=SimpleNamespace(backend="tilelang"), forward=object())
    marker = attention.forward

    class Actor:
        model = SimpleNamespace(conversion_config=True, layers=[SimpleNamespace(self_attn=attention)])

        def __init__(self):
            self.positions = []

        def __call__(self, input_ids, *, use_cache, position_ids, attention_mask,
                     logits_to_keep, past_key_values=None):
            assert use_cache and logits_to_keep == 1
            assert attention_mask == dict(full_attention=None, sliding_attention=None)
            self.positions.append(position_ids.tolist())
            if past_key_values is not None:
                assert past_key_values is self
            return SimpleNamespace(logits=torch.zeros(1, 1, 3), past_key_values=self)

    actor = Actor()
    with _oracle._native_serving_precision(actor, dict(backend="FLASH_ATTN", flash_attn_version=2)):
        values = _oracle._cached_response_logprobs(actor, torch.zeros(1, 1102, dtype=torch.long),
                                                 1100, [1, 2], full_history=True)
    assert len(values) == 2 and attention.forward is marker
    assert actor.positions == [[list(range(1100))], [[1100]]]


def test_native_conversion_layer_gradients_have_separate_blocks():
    spec = spec_from_file_location("megatron_admission", Path(__file__).parent /
                                  "integration/test_megatron_agentic_training.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    for index in range(48):
        assert module._block(f"policy.model.layers.{index}.self_attn.short_kv_delta") == f"layers.{index}"
    assert module._block("policy.model.base.layers.2.self_attn.o_proj.weight") == "base.layers.2"
    assert module._block("policy.model.adapters.2.native.o_proj.weight") == "adapters.2"
    assert module._block("policy.lm_head.weight") == "embedding_and_head"
