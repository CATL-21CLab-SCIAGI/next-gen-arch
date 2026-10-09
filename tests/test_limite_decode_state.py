from types import SimpleNamespace

import pytest
import torch
from torch import nn

from archlab.architectures.limite_decode_state import InferenceBuffers, cache_rows
from archlab.architectures.limite_gqa import _DecodeInterface


class Folded(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4.0))
        self.register_buffer("_inference_weight", None, persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self._inference_weight = None if mode else self.weight.detach()
        return self


def test_folded_buffers_own_storage_and_refresh_after_policy_update():
    model = Folded().eval()
    buffers = InferenceBuffers()
    with torch.no_grad():
        buffers.synchronize(model)
        pointer = model._inference_weight.data_ptr()
        assert pointer != model.weight.data_ptr()
        model.train()
        model.weight.add_(10)
        model.eval()
        buffers.synchronize(model)
    assert model._inference_weight.data_ptr() == pointer
    assert torch.equal(model._inference_weight, model.weight)
    model._inference_weight.zero_()
    assert model.weight.tolist() == [10, 11, 12, 13]
    model.train()
    with pytest.raises(ValueError, match="eval"):
        buffers.synchronize(model)


def test_cache_row_compaction_preserves_local_global_and_short_histories():
    key = torch.arange(3 * 2 * 8 * 4).reshape(3, 2, 8, 4)
    local = key[:, :, -5:]
    short = torch.arange(3 * 16 * 4).reshape(3, 16, 4)
    source = SimpleNamespace(buffers=[[key, key + 1], [local, local + 1]], global_layers={0},
                             archlab_short={0: [short, short + 1]}, get_seq_length=lambda: 8)
    source.archlab_native_preludes = SimpleNamespace(layers=[SimpleNamespace(keys=key, values=key)],
                                                    get_seq_length=lambda: 8)
    rows = torch.tensor([2, 0])
    result = cache_rows(source, rows, length=7)
    assert result.get_seq_length() == 7
    assert torch.equal(result.layers[0].keys, key[rows, :, :7])
    assert torch.equal(result.layers[1].keys, local[rows])
    assert torch.equal(result.archlab_short[0][0], short[rows, -7:])
    assert torch.equal(result.archlab_native_preludes.layers[0].keys, key[rows])
    with pytest.raises(ValueError, match="nonempty"):
        cache_rows(source, rows, length=0)


def test_masked_decode_gqa_matches_expansion_and_retains_training_interface():
    calls = []

    def original(*args, **kwargs):
        calls.append(True)
        return "publisher"

    interface = _DecodeInterface(SimpleNamespace(get_interface=lambda name, fallback: original))
    reduction = interface.get_interface("sdpa", None)
    q, k, v = torch.randn(2, 10, 1, 8), torch.randn(2, 2, 9, 8), torch.randn(2, 2, 9, 8)
    mask = torch.arange(9).reshape(1, 1, 1, -1) < 7
    module = SimpleNamespace(training=False, _archlab_decode_gqa=True)
    result, _ = reduction(module, q, k, v, mask, scaling=.25)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(5, 1), v.repeat_interleave(5, 1), attn_mask=mask, scale=.25)
    torch.testing.assert_close(result, expected.transpose(1, 2))
    module.training = True
    assert reduction(module, q, k, v, mask) == "publisher"
    module.training = False
    assert reduction(module, q.expand(-1, -1, 2, -1), k, v, mask) == "publisher"
    assert len(calls) == 2
    assert interface.get_interface("flash_attention_2", None) is original
