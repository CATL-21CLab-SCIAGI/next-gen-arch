"""Compilation keeps the same trainable tensors, checkpoint keys and updates."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from archlab.automodel.limite_adapter_common import loss_sum
from archlab.automodel.limite_adapter_compilation import configure_compilation


class _Projection(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype))


class _Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(11, 4, dtype=torch.bfloat16)
        self.base = nn.Module()
        self.base.layers = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.adapter_config = {"variant": "normal"}
        for _ in range(4):
            layer = nn.Module()
            layer.self_attn = _Projection(4, 4, dtype=torch.bfloat16)
            layer.mlp = _Projection(4, 4, dtype=torch.bfloat16)
            self.base.layers.append(layer)
            adapter = nn.Module()
            adapter.native = _Projection(4, 4, dtype=torch.float32)
            self.adapters.append(adapter)

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        for layer, adapter in zip(self.base.layers, self.adapters, strict=True):
            x = layer.mlp(layer.self_attn(adapter.native(x)))
        return SimpleNamespace(last_hidden_state=x)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Body()
        self.head = nn.Linear(4, 11, bias=False, dtype=torch.bfloat16)

    def _softcapped_logits(self, x):
        return 3.0 * torch.sigmoid(self.head(x).float() / 2.0)


def test_regional_compilation_preserves_optimizer_identity_and_next_update():
    compiler_config = getattr(torch.compiler, "config", None)
    if not all(
        hasattr(compiler_config, name)
        for name in ("recompile_limit", "accumulated_recompile_limit")
    ):
        pytest.skip("the configured container compiler API is unavailable")
    torch.manual_seed(19)
    eager = _Model()
    actual = copy.deepcopy(eager)
    keys = list(actual.state_dict())
    parameters = list(actual.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    reference_optimizer = torch.optim.AdamW(eager.parameters(), lr=1e-3)
    old_limit = torch.compiler.config.recompile_limit
    old_accumulated = torch.compiler.config.accumulated_recompile_limit
    try:
        part, _ = configure_compilation(actual, "strict-regional", backend="aot_eager")
        assert keys == list(actual.state_dict())
        assert all(a is b for a, b in zip(parameters, actual.parameters(), strict=True))
        assert all(a is b for a, b in zip(parameters, optimizer.param_groups[0]["params"], strict=True))
        ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[2, 3, 4, -100]])
        expected_loss = loss_sum(eager, ids, labels, chunk=2)
        actual_loss = loss_sum(actual, ids, labels, chunk=2, head_part=part)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        expected_loss.backward()
        actual_loss.backward()
        for a, b in zip(actual.parameters(), eager.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        optimizer.step()
        reference_optimizer.step()
        for a, b in zip(actual.parameters(), eager.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            for name in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    optimizer.state[a][name], reference_optimizer.state[b][name], rtol=0, atol=0
                )
    finally:
        torch.compiler.config.recompile_limit = old_limit
        torch.compiler.config.accumulated_recompile_limit = old_accumulated


def test_default_compilation_is_inert_and_invalid_modes_fail():
    model = _Model()
    original = model.model.base.layers[0].self_attn.forward.__func__
    assert configure_compilation(model) == (None, {"mode": "none"})
    assert model.model.base.layers[0].self_attn.forward.__func__ is original
    with pytest.raises(ValueError, match="invalid"):
        configure_compilation(model, "whole")
    model.model.adapter_config["variant"] = "simplicial"
    with pytest.raises(ValueError, match="only for normal"):
        configure_compilation(model, "strict-regional")
