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
    token = torch.tensor([[[[prefix_length]]]], dtype=torch.float32)
    updated, _ = cache.update(token, token, 0)
    mask = cache.decode_masks()["sliding_attention"].reshape(-1)
    actual = updated[0, 0, :, 0][mask]
    assert cache.local_capacity == span
    assert mask.sum() == min(prefix_length + 1, span)
    assert torch.equal(actual, torch.arange(start, prefix_length + 1, dtype=torch.float32))
