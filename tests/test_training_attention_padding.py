"""FA4-only installs reuse upstream padding without changing masks or gradients."""

import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _module(name, filename):
    path = Path(__file__).resolve().parents[1] / "third_party/verl/verl/utils" / filename
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _without_fa2(monkeypatch, *, missing="flash_attn"):
    helpers = _module("upstream_torch_padding", "npu_flash_attn_utils.py")
    monkeypatch.setitem(sys.modules, "verl.utils.device",
                        SimpleNamespace(is_torch_npu_available=lambda **kwargs: False))
    monkeypatch.setitem(sys.modules, "verl.utils.npu_flash_attn_utils", helpers)
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "flash_attn.bert_padding":
            raise ModuleNotFoundError("Missing test dependency", name=missing)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    return _module("upstream_attention_dispatch", "attention_utils.py")


@pytest.mark.parametrize("missing", ["flash_attn", "flash_attn.bert_padding"])
def test_fa4_only_padding_preserves_values_masks_and_backward(monkeypatch, missing):
    padding = _without_fa2(monkeypatch, missing=missing)
    values = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).requires_grad_()
    mask = torch.tensor([[0, 1, 1, 1], [1, 0, 1, 0]])
    packed, indices, cumulative, maximum, used = padding.unpad_input(values, mask)
    restored = padding.pad_input(packed, indices, 2, 4)
    torch.testing.assert_close(restored, values.detach() * mask[..., None])
    assert cumulative.tolist() == [0, 3, 5] and maximum == 3 and used.tolist() == [3, 2]
    restored.sum().backward()
    torch.testing.assert_close(values.grad, mask[..., None].expand_as(values).float())


def test_missing_fa2_transitive_dependency_is_not_hidden(monkeypatch):
    padding = _without_fa2(monkeypatch, missing="unexpected_dependency")
    with pytest.raises(ModuleNotFoundError, match="Missing test dependency"):
        padding._get_attention_functions()
