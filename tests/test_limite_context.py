"""Static native-context boundaries independent of an installed publisher."""

from types import SimpleNamespace

import pytest
import torch

from archlab.architectures.limite_context import (
    NativeTrainingContextCache,
    native_training_context_key,
)


def test_preparation_retains_exact_native_masks_and_guards_cold_capture():
    ids = torch.arange(8).view(2, 4)
    cache = NativeTrainingContextCache()
    calls = []
    native_masks = {"full_attention": None, "sliding_attention": torch.ones(4, 4, dtype=torch.bool)}

    def build(shape, positions, mask, past):
        assert shape.shape == (2, 4, 0)
        assert shape.dtype == torch.bfloat16
        assert mask is None and past is None
        calls.append(positions)
        return native_masks

    with pytest.raises(RuntimeError, match="enable static"):
        cache.prepare(ids, ("first",), torch.bfloat16, build)
    cache.enable()
    entry = cache.prepare(ids, ("first",), torch.bfloat16, build)
    assert entry.masks is native_masks
    assert entry.positions is calls[0]
    assert cache.prepare(ids + 1, ("first",), torch.bfloat16, build) is entry
    assert len(calls) == 1
    cache.lock()
    with pytest.raises(RuntimeError, match="before CUDA capture"):
        cache.prepare(ids, ("second",), torch.bfloat16, build)
    assert len(calls) == 1
    cache.enable(False)
    assert cache.entry is None and not cache.enabled and not cache.locked


@pytest.mark.parametrize(
    "change",
    [
        "batch",
        "length",
        "device",
        "dtype",
        "training",
        "base_training",
        "mode",
        "backend",
        "window",
        "types",
        "globals",
    ],
)
def test_context_key_tracks_every_native_geometry_and_mode_boundary(change):
    ids = SimpleNamespace(shape=(2, 4), device=torch.device("cpu"))
    config = SimpleNamespace(
        _attn_implementation="sdpa",
        sliding_window=3,
        layer_types=["sliding_attention", "full_attention"],
        global_layers=[1],
    )
    base = SimpleNamespace(
        config=config,
        training=True,
        embed_tokens=SimpleNamespace(weight=torch.empty(0, dtype=torch.bfloat16)),
    )
    training, mode = True, "full"
    before = native_training_context_key(ids, base, training, mode)
    if change == "batch":
        ids.shape = (1, 4)
    elif change == "length":
        ids.shape = (2, 5)
    elif change == "device":
        ids.device = torch.device("cuda", 1)
    elif change == "dtype":
        base.embed_tokens.weight = torch.empty(0)
    elif change == "training":
        training = False
    elif change == "base_training":
        base.training = False
    elif change == "mode":
        mode = "adapter"
    elif change == "backend":
        config._attn_implementation = "eager"
    elif change == "window":
        config.sliding_window += 1
    elif change == "types":
        config.layer_types.reverse()
    elif change == "globals":
        config.global_layers = [0]
    assert native_training_context_key(ids, base, training, mode) != before
