from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from archlab.automodel import limite_adapter_rl as rl


class MixedPolicy(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.model = nn.Module()
        self.model.trainable_mode = mode
        self.model.adapters = nn.Linear(2, 2, bias=False)
        self.model.base = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.model.scale = nn.Parameter(torch.ones((), dtype=torch.float32))
        if mode == "adapter":
            self.model.base.requires_grad_(False)
            self.model.scale.requires_grad_(False)
        self.archlab_base_snapshot_sha256 = "publisher"


def test_full_rl_parent_requires_exact_completed_full_warmup():
    marker = {"tokens": 10_000_000_000}
    receipt = dict(tokens=marker["tokens"], trainable_mode="full")
    rl.check_warmup_parent(marker, receipt, "full")
    with pytest.raises(ValueError, match="counts differ"):
        rl.check_warmup_parent(marker, dict(receipt, tokens=2_000_000_000), "full")
    with pytest.raises(ValueError, match="full-weight warmup"):
        rl.check_warmup_parent(marker, dict(receipt, trainable_mode="adapter"), "full")
    with pytest.raises(ValueError, match="exactly 10B"):
        rl.check_warmup_parent({"tokens": 2}, {"tokens": 2}, "adapter")
    rl.check_warmup_parent({"tokens": 2}, {"tokens": 2}, "full", correctness_fixture=True)


def test_full_identity_records_origin_and_all_parameters_are_trainable(monkeypatch):
    model = MixedPolicy("full")
    monkeypatch.setattr(rl, "frozen_fingerprint", lambda _: pytest.fail("full weights are trainable"))
    assert rl.identity_contract(model, {"base_snapshot_sha256": "publisher"}) == {
        "trainable_mode": "full", "base_snapshot_sha256": "publisher"
    }
    model.model.base.requires_grad_(False)
    with pytest.raises(ValueError, match="every model parameter"):
        rl.identity_contract(model, {"base_snapshot_sha256": "publisher"})


def test_adapter_identity_still_detects_frozen_weight_changes():
    model = MixedPolicy("adapter")
    receipt = {"frozen_sha256": rl.frozen_fingerprint(model)}
    identity = rl.identity_contract(model, receipt)
    with torch.no_grad():
        model.model.adapters.weight.add_(1)
    assert rl.identity_contract(model, receipt) == identity
    with torch.no_grad():
        model.model.base.weight.add_(1)
    with pytest.raises(ValueError, match="frozen base"):
        rl.identity_contract(model, receipt)


@pytest.mark.parametrize("mode,masters", [("adapter", False), ("full", True)])
def test_optimizer_preserves_mixed_dtypes_and_selects_full_masters(monkeypatch, mode, masters):
    model = MixedPolicy(mode)
    captured = {}

    def optimizer(parameters, **kwargs):
        captured.update(kwargs, parameters=list(parameters))
        return captured

    monkeypatch.setitem(sys.modules, "archlab.optimizers.rl_adam", SimpleNamespace(SignalFusedAdam=optimizer))
    assert rl.optimizer_for_model(model)["master_weights"] is masters
    assert {id(parameter) for parameter in captured["parameters"]} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert model.model.base.weight.dtype == torch.bfloat16
    assert model.model.scale.dtype == torch.float32


def test_mixed_gradient_evidence_and_flat_advantage_have_zero_signal():
    model = MixedPolicy("full")
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    evidence = rl.gradient_evidence(model)
    assert evidence["adapter_gradient_norm"] > 0
    assert evidence["backbone_gradient_norm"] > 0
    assert evidence["backbone_gradient_tensors"] == 2
    # Flat normalized rewards produce zero advantages in the same surrogate.
    for parameter in model.parameters():
        parameter.grad.zero_()
    assert rl.gradient_evidence(model)["backbone_gradient_norm"] == 0
    model.model.scale.grad.fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        rl.gradient_evidence(model)


def test_deferred_reduction_occurs_only_at_final_accumulated_backward(monkeypatch):
    model = MixedPolicy("full")
    calls = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(rl, "synchronize_gradients", lambda parameters: calls.append(list(parameters)))
    rl.finish_backward(model.parameters(), "deferred", False)
    rl.finish_backward(model.parameters(), "bucketed", True)
    assert calls == []
    rl.finish_backward(model.parameters(), "deferred", True)
    assert calls == [list(model.parameters())]


def test_checkpoint_oracle_checks_scalar_masters_moments_and_dtypes():
    state = {"step": 3, "state": [torch.ones(()), torch.ones(4, dtype=torch.float32)]}
    copied = {"step": 3, "state": [tensor.clone() for tensor in state["state"]]}
    rl.assert_state_equal(state, copied)
    copied["state"][1][0] += 1
    with pytest.raises(AssertionError, match="values differ"):
        rl.assert_state_equal(state, copied)
    copied["state"][1] = state["state"][1].bfloat16()
    with pytest.raises(AssertionError, match="dtype differs"):
        rl.assert_state_equal(state, copied)


def test_flat_optimizer_probe_does_not_alias_live_masters_or_step():
    state = {"state": {0: {"step": 3, "master_param": torch.ones(1024)}}}
    probe = rl.optimizer_probe(state)
    state["state"][0]["master_param"][0] += 1
    state["state"][0]["step"] += 1
    assert probe["state"][0]["step"] == 3
    assert probe["state"][0]["master_param"]["values"][0] == 1
    assert probe["state"][0]["master_param"]["shape"] == (1024,)
