from types import SimpleNamespace

import pytest
import torch

from archlab.automodel.deepseek_v41_official_reference import _co_resident_reserve, reference_hidden


class _Reference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(12, 4, dtype=torch.bfloat16)
        self.norm = torch.nn.Identity()
        self.max_seq_len = 8
        self.calls = []

    def forward(self, tokens, start_pos):
        self.calls.append(start_pos)
        hidden = self.norm(self.embed(tokens))
        torch.rand(3)  # Native generation sampling must not disturb training RNG.
        return hidden[:, -1]


def test_capture_uses_full_native_forward_and_restores_process_state():
    model = _Reference()
    tokens = torch.tensor([[1, 4, 8]])
    before = torch.random.get_rng_state()
    dtype = torch.get_default_dtype()
    actual = reference_hidden(model, tokens)
    torch.testing.assert_close(actual, model.embed(tokens))
    assert model.calls == [0]
    assert not actual.requires_grad
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.get_default_dtype() == dtype
    assert not model.norm._forward_hooks


def test_capture_removes_hook_after_native_forward_failure():
    model = _Reference()
    def fail(*args, **kwargs):
        raise RuntimeError("native error")
    model.forward = fail
    before = torch.get_default_dtype()
    with pytest.raises(RuntimeError, match="native error"):
        reference_hidden(model, torch.tensor([[1, 2]]))
    assert torch.get_default_dtype() == before
    assert not model.norm._forward_hooks


@pytest.mark.parametrize("allocated,initial_other,expected_other", [(215, 50, 55), (200, 50, 50)])
def test_capacity_keeps_other_live_models_in_the_reserve(monkeypatch, allocated, initial_other, expected_other):
    gib = 2**30
    parameter = SimpleNamespace(device=torch.device("cuda"), numel=lambda: 160 * gib // 2, element_size=lambda: 2)
    model = SimpleNamespace(parameters=lambda: iter([parameter]), buffers=lambda: iter([]))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: allocated * gib)
    reserve, report = _co_resident_reserve(model, other_allocated_bytes=initial_other * gib, workspace_gib=24)
    assert reserve == expected_other + 24
    assert report["co_resident_tensor_gib"] == expected_other
