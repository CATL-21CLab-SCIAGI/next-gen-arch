"""The native config already includes the query in its local key span."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("prefix_length", [3, 1024, 1027])
def test_fixed_cache_matches_native_inclusive_window_without_extra_slot(monkeypatch, prefix_length):
    class DynamicCache:
        def __init__(self, *, config):
            pass

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(DynamicCache=DynamicCache))
    path = Path(__file__).parents[1] / "src/archlab/architectures/limite_decode.py"
    spec = importlib.util.spec_from_file_location("_native_limite_decode_window", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Serialized history width1024 is converted to runtime span1025 by
    # publisher LimiteConfig, and DynamicSlidingWindowLayer retains1024past.
    span = 1025
    start = max(0, prefix_length - span + 1)
    keys = torch.arange(start, prefix_length, dtype=torch.float32)[None, None, :, None]
    source = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=keys)],
                             get_seq_length=lambda: prefix_length)
    config = SimpleNamespace(sliding_window=span, global_layers=[])
    position = torch.tensor([prefix_length])
    cache = module.DecodeCache(source, config, 18432, position)
    assert cache.buffers[0][0].transpose(1, 2).is_contiguous()
    token = torch.tensor([[[[prefix_length]]]], dtype=torch.float32)
    updated, _ = cache.update(token, token, 0)
    mask = cache.decode_masks()["sliding_attention"].reshape(-1)
    actual = updated[0, 0, :, 0][mask]
    assert cache.local_capacity == span
    assert mask.sum() == min(prefix_length + 1, span)
    assert torch.equal(actual, torch.arange(start, prefix_length + 1, dtype=torch.float32))


def test_interleaved_cache_global_updates_and_compaction_preserve_values(monkeypatch):
    class DynamicCache:
        def __init__(self, *, config):
            pass

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(DynamicCache=DynamicCache))
    from archlab.architectures.limite_decode_state import cache_rows

    path = Path(__file__).parents[1] / "src/archlab/architectures/limite_decode.py"
    spec = importlib.util.spec_from_file_location("_native_limite_interleaved_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keys = torch.arange(3 * 2 * 7 * 4, dtype=torch.float32).reshape(3, 2, 7, 4)
    short = torch.arange(3 * 7 * 2 * 4, dtype=torch.float32).reshape(3, 7, 2, 4)
    source = SimpleNamespace(
        layers=[SimpleNamespace(keys=keys, values=keys + 1),
                SimpleNamespace(keys=keys[:, :, -4:], values=keys[:, :, -4:] + 1)],
        get_seq_length=lambda: 7, archlab_short={0: [short, short + 1]},
    )
    config = SimpleNamespace(sliding_window=5, global_layers=[0])
    position = torch.tensor([7])
    cache = module.DecodeCache(source, config, 16, position)
    pointers = [[tensor.data_ptr() for tensor in pair] for pair in cache.buffers]
    for step in range(3):
        position.fill_(7 + step)
        token = torch.full((3, 2, 1, 4), 1000. + step)
        global_keys, _ = cache.update(token, token + 1, 0)
        local_keys, _ = cache.update(token, token + 1, 1)
        cache.update_short(token.transpose(1, 2), (token + 1).transpose(1, 2), 0)
        assert torch.equal(global_keys[:, :, 7 + step:8 + step], token)
        assert torch.equal(local_keys[:, :, -1:], token)
        assert global_keys.transpose(1, 2).is_contiguous()
        assert local_keys.transpose(1, 2).is_contiguous()
    selected = cache_rows(cache, torch.tensor([2, 0]), length=10)
    migrated = module.DecodeCache(selected, config, 16, torch.tensor([10]))
    assert torch.equal(migrated.buffers[0][0][:, :, :10], cache.buffers[0][0][[2, 0], :, :10])
    assert torch.equal(migrated.buffers[1][0], cache.buffers[1][0][[2, 0]])
    assert torch.equal(migrated.archlab_short[0][0][:, -10:], cache.archlab_short[0][0][[2, 0], -10:])
    assert pointers == [[tensor.data_ptr() for tensor in pair] for pair in cache.buffers]
