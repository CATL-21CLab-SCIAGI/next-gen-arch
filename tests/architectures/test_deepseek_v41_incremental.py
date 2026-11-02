"""Nonzero trained branches, window eviction, and request isolation."""

import pytest
import torch

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter
from archlab.architectures.deepseek_v41_incremental import IncrementalV41Adapter
from archlab.architectures.deepseek_v41_normal_adapter import V41NormalAttentionAdapter


def make_adapter(variant, short=3, long=7):
    config = V41AdapterConfig(width=16, streams=2, query_heads=2, kv_heads=1,
                              head_dim=16, short_window=short, long_window=long)
    cls = V41NormalAttentionAdapter if variant == "normal" else V41SimplicialAdapter
    adapter = cls(config, backend="reference").eval()
    with torch.no_grad():
        # Zero initialization would make a broken attention implementation pass.
        adapter.output.weight.normal_(std=0.2)
        adapter.read_logits.normal_()
        adapter.write_logits.normal_()
    return adapter


@pytest.mark.parametrize("variant", ["normal", "simplicial"])
@pytest.mark.parametrize("chunks", [[19], [1] * 19, [2, 6, 1, 10]])
def test_chunking_matches_full_nonzero_branch(variant, chunks):
    torch.manual_seed(71)
    adapter = make_adapter(variant)
    streams = torch.randn(2, 19, 2, 16)
    with torch.no_grad():
        expected = adapter(streams)
    cache, actual, position = IncrementalV41Adapter(adapter), [], 0
    for length in chunks:
        actual.append(cache(streams[:, position:position + length], start_position=position))
        position += length
    torch.testing.assert_close(torch.cat(actual, 1), expected, atol=2e-6, rtol=2e-6)
    assert cache.position == 19
    assert cache.cache_lengths == ({"k": 7, "v": 7} if variant == "normal"
                                   else {"k1": 3, "k2": 7, "v1": 3, "v2": 7})


@pytest.mark.parametrize("variant", ["normal", "simplicial"])
def test_production_windows_and_reset(variant):
    torch.manual_seed(123)
    adapter = make_adapter(variant, 32, 512)
    streams = torch.randn(1, 514, 2, 16)
    cache = IncrementalV41Adapter(adapter)
    actual = cache(streams, start_position=0)
    with torch.no_grad():
        expected = adapter(streams)
    torch.testing.assert_close(actual[:, [0, 31, 32, 511, 512, 513]],
                               expected[:, [0, 31, 32, 511, 512, 513]], atol=2e-6, rtol=2e-6)
    assert max(cache.cache_lengths.values()) == 512
    with pytest.raises(ValueError, match="noncontiguous"):
        cache(streams[:, :1], start_position=0)
    cache.reset()
    torch.testing.assert_close(cache(streams[:, :4], start_position=0), expected[:, :4],
                               atol=2e-6, rtol=2e-6)


def test_separate_requests_and_validation():
    adapter = make_adapter("simplicial")
    first, second = IncrementalV41Adapter(adapter), IncrementalV41Adapter(adapter)
    first(torch.randn(1, 3, 2, 16), start_position=0)
    assert second.position == 0 and second.cache_lengths == {}
    with pytest.raises(ValueError, match="batch/device/dtype"):
        first(torch.randn(2, 1, 2, 16), start_position=3)
    assert first.position == 3
    adapter.train()
    with pytest.raises(ValueError, match="eval mode"):
        second(torch.randn(1, 1, 2, 16), start_position=0)


def test_failed_projection_rolls_back_partial_cache(monkeypatch):
    adapter = make_adapter("simplicial")
    cache = IncrementalV41Adapter(adapter)
    cache(torch.randn(1, 2, 2, 16), start_position=0)
    previous = {key: value.clone() for key, value in cache._cache.items()}

    def fail(*args):
        raise RuntimeError("simulated output projection failure")

    monkeypatch.setattr(adapter.output, "forward", fail)
    with pytest.raises(RuntimeError, match="projection failure"):
        cache(torch.randn(1, 1, 2, 16), start_position=2)
    assert cache.position == 2
    for key, value in previous.items():
        torch.testing.assert_close(cache._cache[key], value, rtol=0, atol=0)
