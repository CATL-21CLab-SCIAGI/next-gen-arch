import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from archlab.architectures.prelude_loop_coda import LoopLayout, boundary
from archlab.automodel.deepseek_v41_loop import install_loop, set_repetitions


class Block(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4) / 8)
        self.index = index
        self.engram = None
        self.ffn = SimpleNamespace(gate=SimpleNamespace(aux_loss_coeff=0.01))
        self.attn = SimpleNamespace(indexer=SimpleNamespace())

    def forward(self, hidden, pre_mix, state, **kwargs):
        # The index assertion catches accidental carry of final-core CSA2 state
        # into the next core, which must re-enter with prelude ownership.
        assert state == self.index
        output = checkpoint(lambda x: x + (x @ self.weight).tanh(), hidden, use_reentrant=False)
        return output, pre_mix, state + 1


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({str(i): Block(i) for i in range(3)})

    def forward(self, hidden):
        mix, state = hidden.new_zeros(1), 0
        for layer in self.layers.values():
            hidden, mix, state = layer(hidden, mix, state)
        return hidden


@pytest.mark.parametrize("recursions", [1, 2, 4, 12])
def test_tied_forward_and_full_gradients_match_independent_unroll(recursions):
    torch.manual_seed(7)
    model = nn.Module()
    model.model = Decoder()
    oracle = copy.deepcopy(model)
    names = list(model.state_dict())
    install_loop(model, layout=LoopLayout(1, 1, 1))
    set_repetitions(model, recursions)
    x = torch.randn(2, 3, 2, 4, requires_grad=True)
    reference_input = x.detach().clone().requires_grad_()
    actual = model.model(x)
    anchor = reference_input + (reference_input @ oracle.model.layers["0"].weight).tanh()
    expected = anchor
    for _ in range(recursions):
        expected = expected + (expected @ oracle.model.layers["1"].weight).tanh()
        # Independent literal implementation of the reference equation.
        expected = (
            expected
            * (expected.square().mean(-1, keepdim=True) + torch.finfo(torch.float32).eps).rsqrt()
            + anchor / 2**0.5
        )
    expected = expected + (expected @ oracle.model.layers["2"].weight).tanh()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_input.grad, rtol=1e-5, atol=2e-5)
    for p, reference in zip(model.parameters(), oracle.parameters(), strict=True):
        torch.testing.assert_close(p.grad, reference.grad, rtol=1e-5, atol=2e-5)
    assert list(model.state_dict()) == names
    assert len(list(model.model.layers.values())) == 3
    assert model.model.layers["1"].attn.indexer._archlab_loop_visits == recursions


def test_two_live_forward_graphs_keep_separate_anchors():
    model = nn.Module()
    model.model = Decoder()
    install_loop(model, layout=LoopLayout(1, 1, 1))
    x, y = torch.randn(2, 4, requires_grad=True), torch.randn(2, 4, requires_grad=True)
    first, second = model.model(x), model.model(y)
    (first.sum() + second.sum()).backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
    assert model.model.layers.executing is False


def test_boundary_is_finite_on_zero_and_preserves_storage_dtype():
    h = torch.zeros(2, 4, dtype=torch.bfloat16)
    e = torch.ones_like(h)
    result = boundary(h, e)
    assert result.dtype == h.dtype and torch.isfinite(result).all()
    torch.testing.assert_close(result.float(), torch.full((2, 4), 2**-0.5), atol=0.003, rtol=0)
