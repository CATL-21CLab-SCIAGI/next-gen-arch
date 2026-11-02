"""Numerical oracle for the container-owned cache reduction API."""

from types import SimpleNamespace

import pytest
import torch

from archlab.architectures.limite_gqa import _DecodeInterface


def reduction(fallback):
    return _DecodeInterface(SimpleNamespace(get_interface=lambda *_: fallback)).get_interface('sdpa', None)


def test_training_keeps_publisher_reduction():
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return 'publisher', None
    module = SimpleNamespace(training=True, _archlab_decode_gqa=True)
    q = torch.empty(1, 4, 1, 8)
    assert reduction(original)(module, q, q, q, None) == ('publisher', None)
    assert len(calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='container CUDA qualification')
@pytest.mark.parametrize('is_global', [True, False])
def test_flash_cache_valid_lengths_and_graph_replay(is_global):
    pytest.importorskip('flash_attn')
    module = SimpleNamespace(training=False, _archlab_decode_gqa=True,
                             _archlab_decode_backend='flash_attn_kvcache', is_global=is_global)
    call = reduction(lambda *_args, **_kwargs: pytest.fail('unexpected fallback'))
    q = torch.randn(3, 8, 1, 64, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(3, 2, 1025, 64, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    positions = torch.arange(1025, device='cuda')
    mask = (positions < 7 if is_global else positions >= 1018)[None, None, None]
    for _ in range(2):
        call(module, q, k, v, mask)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, _ = call(module, q, k, v, mask)
    for count in (7, 103, 1025):
        mask.copy_((positions < count if is_global else positions >= 1025 - count)[None, None, None])
        graph.replay()
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), attn_mask=mask, enable_gqa=True,
        ).transpose(1, 2)
        torch.testing.assert_close(actual.float(), expected, rtol=.01, atol=.005)
