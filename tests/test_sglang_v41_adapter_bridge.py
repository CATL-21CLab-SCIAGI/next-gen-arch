from types import SimpleNamespace

import pytest
import torch
from test_deepseek_v41_incremental import make_adapter

from archlab.serving.sglang_v41_adapter_bridge import (
    RequestAdapterCaches,
    install_adapter_boundary,
)


def test_interleaved_request_slots_and_reuse_preserve_independent_prefixes():
    adapter = make_adapter("simplicial")
    pool = RequestAdapterCaches(adapter, max_slots=2)
    a, b, c = (torch.randn(1, n, 2, 16) for n in (5, 4, 2))
    pool.apply(torch.cat((a[0, :3], b[0, :2])), slots=[0, 1], lengths=[3, 2],
               positions=[0, 1, 2, 0, 1])
    result = pool.apply(torch.cat((b[0, 2:3], a[0, 3:4])), slots=[1, 0],
                        lengths=[1, 1], positions=[2, 3])
    with torch.no_grad():
        expected = torch.cat((adapter(b)[0, 2:3], adapter(a)[0, 3:4]))
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)
    actual = pool.apply(c[0], slots=[0], lengths=[2], positions=[0, 1])
    with torch.no_grad():
        torch.testing.assert_close(actual, adapter(c)[0], atol=2e-6, rtol=2e-6)
    assert pool.caches[1].position == 3
    with pytest.raises(ValueError, match="prefix"):
        pool.apply(a[0, :1], slots=[0], lengths=[1], positions=[4])
    with pytest.raises(ValueError, match="slot"):
        pool.apply(a[0, :1], slots=[2], lengths=[1], positions=[0])


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hc_pre_from_prev_sublayer = True
        self.post_attention_layernorm = torch.nn.Identity()
        self.bypass = False

    def _hc_post_with_combine(self, x, residual, post, comb, pre, batch, norm=None):
        return x, x.sum(-2), x.sum(-2)

    def forward_hc_pre_from_prev(self, positions, hidden_states, input_ids, forward_batch,
                                input_ids_global, prev_pre, **kwargs):
        if self.bypass:
            return hidden_states
        return self._hc_post_with_combine(hidden_states, None, None, None, None,
                                         forward_batch, norm=self.post_attention_layernorm)


def test_boundary_adapts_before_ffn_and_discards_stale_fused_inputs():
    layer, adapter = Layer(), make_adapter("normal")
    install_adapter_boundary(layer, adapter, tp_size=8)
    batch = SimpleNamespace(req_pool_indices=torch.tensor([0]), extend_seq_lens_cpu=[3],
                            forward_mode=SimpleNamespace(is_decode=lambda: False,
                                                         is_extend_without_speculative=lambda: True))
    streams = torch.randn(3, 2, 16)
    actual, combined, normalized = layer.forward_hc_pre_from_prev(
        torch.arange(3), streams, None, batch, None, None)
    with torch.no_grad():
        torch.testing.assert_close(actual, adapter(streams.unsqueeze(0))[0], atol=2e-6, rtol=2e-6)
    assert combined is None and normalized is None
    layer.bypass = True
    with pytest.raises(RuntimeError, match="bypassed"):
        layer.forward_hc_pre_from_prev(torch.arange(3), streams, None, batch, None, None)


def test_rejects_tp4_fused_path():
    with pytest.raises(ValueError, match="TP8"):
        install_adapter_boundary(Layer(), make_adapter("normal"), tp_size=4)


def test_packaged_preview_boundary_adapts_only_the_attention_expansion():
    class DirectLayer(torch.nn.Module):
        hc_pre_from_prev_sublayer = True

        def hc_post(self, x, residual, post, comb):
            return x

        def forward_hc_pre_from_prev(self, positions, hidden_states, input_ids, forward_batch,
                                    input_ids_global, prev_pre):
            hidden = self.hc_post(hidden_states, None, None, None)
            self.ffn_input = hidden.clone()
            return self.hc_post(hidden, None, None, None), None

    layer, adapter = DirectLayer(), make_adapter("simplicial")
    install_adapter_boundary(layer, adapter, tp_size=8)
    batch = SimpleNamespace(req_pool_indices=torch.tensor([0]), extend_seq_lens_cpu=[4],
                            forward_mode=SimpleNamespace(is_decode=lambda: False,
                                                         is_extend_without_speculative=lambda: True))
    streams = torch.randn(4, 2, 16)
    actual, _ = layer.forward_hc_pre_from_prev(torch.arange(4), streams, None, batch, None, None)
    with torch.no_grad():
        expected = adapter(streams.unsqueeze(0))[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(layer.ffn_input, expected, atol=2e-6, rtol=2e-6)
