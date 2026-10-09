"""Startup graphs are retained; async misses never enter a CUDA capture."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from archlab.rl.limite_actor import _decode_capacities


@pytest.fixture
def decode(monkeypatch):
    class DynamicCache:
        def __init__(self, *, config):
            pass

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(DynamicCache=DynamicCache))
    path = Path(__file__).parents[1] / "src/archlab/architectures/limite_decode.py"
    spec = importlib.util.spec_from_file_location("_limite_actor_capture_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("budget,expected", [(16384, (17408, 18432)), (256, (1024, 2048, 3072))])
def test_startup_keys_cover_every_qualified_prompt_and_compacted_batch(budget, expected):
    capacities = _decode_capacities(budget, 2048, 1024, 131072)
    assert capacities == expected
    for prompt in range(1, 2049):
        rounded = ((prompt + budget + 1023) // 1024) * 1024
        assert rounded in capacities


def test_startup_capacity_clamps_to_context_and_rejects_overrun():
    assert _decode_capacities(16384, 1600, 1024, 18000) == (17408, 18000)
    with pytest.raises(ValueError, match="context"):
        _decode_capacities(16384, 2048, 1024, 18000)


def test_frozen_pool_reuses_startup_graphs_and_never_captures_new_keys(decode, monkeypatch):
    created = []

    class Decoder:
        def __init__(self, model, source, token, capacity, *, capture=True):
            self.graph = object() if capture else None
            self.resets = []
            created.append((token.shape[0], capacity, capture))

        def reset(self, source, token):
            self.resets.append(source)

    monkeypatch.setattr(decode, "GraphDecoder", Decoder)
    pool = decode.GraphDecoderPool(SimpleNamespace(config=SimpleNamespace(max_position_embeddings=131072)),
                                   max_entries=8)
    startup = {}
    for batch in range(1, 5):
        for capacity in (17408, 18432):
            startup[(batch, capacity)] = pool.get("startup", torch.zeros(batch, 1), capacity)
    pool.freeze_capture()
    captured_seconds = pool.capture_seconds
    for capacity in (19456, 20480, 21504):
        fallback = pool.get("unexpected", torch.zeros(4, 1), capacity)
        assert fallback.graph is None
    for (batch, capacity), original in startup.items():
        assert pool.get("real source", torch.zeros(batch, 1), capacity) is original
        assert original.resets == ["real source"]
    assert len(pool.entries) == 8 and pool.eager_misses == 3
    assert pool.capture_seconds == captured_seconds
    assert all(capture for _, _, capture in created[:8])
    assert not any(capture for _, _, capture in created[8:])


def test_uncaptured_decoder_preserves_fixed_cache_history_without_cuda_calls(decode, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("async fallback must not touch CUDA capture/synchronization")

    for name in ("CUDAGraph", "Stream", "graph", "synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, forbidden)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(sliding_window=5, global_layers=[0])

        def forward(self, input_ids, past_key_values, **kwargs):
            token = input_ids[:, None, :, None].float()
            for index in range(2):
                past_key_values.update(token, token, index)
            masks = past_key_values.decode_masks()
            global_keys = past_key_values.buffers[0][0]
            local_keys = past_key_values.buffers[1][0]
            total = (global_keys * masks["full_attention"][..., None]).sum()
            total += (local_keys * masks["sliding_attention"][..., None]).sum()
            return SimpleNamespace(logits=total.reshape(1, 1, 1))

    keys = torch.arange(3.0)[None, None, :, None]
    source = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=keys)] * 2,
                             get_seq_length=lambda: 3)
    model = Model().eval()
    with torch.no_grad():
        decoder = decode.GraphDecoder(model, source, torch.tensor([[3]]), 10, capture=False)
        pointers = [pair[0].data_ptr() for pair in decoder.cache.buffers]
        for position in range(3, 7):
            actual = decoder(torch.tensor([[position]]), position)
            expected = sum(range(position + 1)) + sum(range(max(0, position - 4), position + 1))
            assert actual.item() == expected
            assert [pair[0].data_ptr() for pair in decoder.cache.buffers] == pointers
        decoder.reset(source, torch.tensor([[3]]))
        assert decoder(torch.tensor([[3]]), 3).item() == 12
