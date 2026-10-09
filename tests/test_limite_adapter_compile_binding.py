"""Project-local attention substitution remains valid under graph capture."""

from types import MethodType

import pytest
import torch
from torch import nn

from archlab.architectures.limite_adapter import (
    _AdapterInterface,
    _attention_forward_with_reduction,
)


def _native_reduce(x):
    return 2 * x


def _adapter_reduce(x):
    return 3 * x


def _other_reduce(x):
    return -5 * x


ALL_ATTENTION_FUNCTIONS = _AdapterInterface(_native_reduce)


class _Attention(nn.Module):
    def __init__(self, reduction=None):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.25))
        if reduction is not None:
            self.forward = MethodType(
                _attention_forward_with_reduction(type(self).forward, reduction), self
            )

    def forward(self, x):
        reduce = ALL_ATTENTION_FUNCTIONS.get_interface("native", None)
        return reduce(x * self.weight)


class _MixedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [_Attention(), *[_Attention(_adapter_reduce) for _ in range(48)]]
        )

    def forward(self, x):
        return sum(layer(x) for layer in self.layers)


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_local_reduction_compiles_with_original_and_repeated_adapters(backend):
    model = _MixedAttention()
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    expected = model(x)
    expected.sum().backward()
    expected_dx = x.grad.clone()
    expected_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    x.grad = None

    compiled = torch.compile(model, backend=backend, fullgraph=True)
    actual = compiled(x)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)
    for p, expected_grad in zip(model.parameters(), expected_grads, strict=True):
        torch.testing.assert_close(p.grad, expected_grad, rtol=0, atol=0)

    # The two adapters share the namespace receiving compiled globals, while
    # other reductions and the original publisher keep their own behavior.
    assert model.layers[1].forward.__func__ is model.layers[2].forward.__func__
    bound = model.layers[1].forward.__func__
    other = _attention_forward_with_reduction(_Attention.forward, _other_reduce)
    assert bound.__globals__ is model.layers[-1].forward.__func__.__globals__
    assert bound.__globals__ is not other.__globals__
    assert bound.__code__ != other.__code__
    assert bound.__code__ != _Attention.forward.__code__
    assert bound.__code__.co_code == _Attention.forward.__code__.co_code
    torch.testing.assert_close(_Attention()(x), 2 * x * 1.25, rtol=0, atol=0)
    torch.testing.assert_close(_Attention(_other_reduce)(x), -5 * x * 1.25, rtol=0, atol=0)
    assert ALL_ATTENTION_FUNCTIONS.reduction is _native_reduce
