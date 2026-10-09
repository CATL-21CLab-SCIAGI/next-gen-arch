"""Numerical oracle for the container-owned cache reduction API."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from archlab.architectures import limite_gqa
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


@pytest.mark.parametrize('training,query_length', [(True, 1), (False, 2)])
def test_math_decode_selection_preserves_training_and_prefill(monkeypatch, training, query_length):
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return 'publisher', None
    monkeypatch.setattr(limite_gqa, 'sdpa_kernel', lambda *_: pytest.fail('decode context leaked'))
    module = SimpleNamespace(training=training, _archlab_decode_gqa=True,
                             _archlab_decode_backend='sdpa_math')
    q = torch.empty(1, 4, query_length, 8)
    assert reduction(original)(module, q, q, q, None) == ('publisher', None)
    assert len(calls) == 1


@pytest.mark.parametrize('backend,expected_context', [('sdpa_math', True), ('sdpa', False)])
def test_only_explicit_math_decode_opens_math_context(monkeypatch, backend, expected_context):
    active = []
    @contextmanager
    def context(selected):
        assert selected == SDPBackend.MATH
        active.append(True)
        try:
            yield
        finally:
            active.pop()
    monkeypatch.setattr(limite_gqa, 'sdpa_kernel', context)
    def score(q, k, v, **kwargs):
        assert bool(active) == expected_context
        assert kwargs['enable_gqa'] is True
        return q
    monkeypatch.setattr(limite_gqa.F, 'scaled_dot_product_attention', score)
    module = SimpleNamespace(training=False, _archlab_decode_gqa=True,
                             _archlab_decode_backend=backend)
    q = torch.empty(1, 4, 1, 8)
    result, probabilities = reduction(lambda *_args, **_kwargs: pytest.fail('fallback'))(
        module, q, q, q, None, scaling=.3,
    )
    assert result.shape == (1, 1, 4, 8)
    assert probabilities is None
    assert not active


@pytest.mark.skipif(not torch.cuda.is_available(), reason='container CUDA qualification')
def test_math_decode_context_overrides_outer_backend_and_captures():
    module = SimpleNamespace(training=False, _archlab_decode_gqa=True,
                             _archlab_decode_backend='sdpa_math')
    call = reduction(lambda *_args, **_kwargs: pytest.fail('unexpected fallback'))
    q = torch.randn(3, 8, 1, 64, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(3, 2, 67, 64, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = (torch.arange(67, device='cuda') < 53)[None, None, None]
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=.17, enable_gqa=True,
        ).transpose(1, 2).contiguous()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for _ in range(2):
            actual, _ = call(module, q, k, v, mask, scaling=.17)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, _ = call(module, q, k, v, mask, scaling=.17)
        graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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
