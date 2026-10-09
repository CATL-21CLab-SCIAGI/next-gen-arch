from types import SimpleNamespace

import pytest
import torch
from torch import nn

from archlab.automodel import limite_native_rl_qualification as qualification


def test_decode_gate_requires_distribution_greedy_cache_and_finite_oracles():
    report = dict(mean_weighted_error=.001, mean_kl=.00001,
                  compaction_max_weighted_error=.001, compaction_max_kl=.00001,
                  greedy_ids_exact=True, pool_reuse_same_graph=True, finite_logits=True)
    assert qualification.decode_admitted(report)
    for key, value in (
        ("mean_weighted_error", .02), ("mean_kl", .001),
        ("compaction_max_weighted_error", .02), ("compaction_max_kl", .001),
        ("greedy_ids_exact", False), ("pool_reuse_same_graph", False), ("finite_logits", False),
    ):
        assert not qualification.decode_admitted(report | {key: value})


def test_full_vocabulary_importance_control_measures_behavior_clip_mass_and_support():
    identical = torch.tensor([[1., 2., 3.]])
    result = qualification.importance_distribution_control(identical, identical)
    assert result["passed"] and result["max_actor_clip_mass"] == 0
    assert result["min_effective_sample_fraction"] == pytest.approx(1.)
    actor = torch.tensor([[.9, .1]]).log()
    native = torch.tensor([[.99, .01]]).log()
    result = qualification.importance_distribution_control(actor, native)
    assert result["max_actor_clip_mass"] == pytest.approx(.1)
    assert result["max_native_clip_mass"] == pytest.approx(.01)
    assert result["max_normalization_error"] < 1e-6
    assert not result["passed"]
    # A tiny vocabulary tail can have a large ratio while its exact clipped
    # mass is small. Selected-token safeguards remain a separate admission.
    result = qualification.importance_distribution_control(
        torch.tensor([[.99, .01]]).log(), torch.tensor([[.999, .001]]).log(),
    )
    assert result["passed"] and result["max_actor_clip_mass"] == pytest.approx(.01)
    assert not qualification.importance_distribution_control(
        torch.full_like(identical, float("nan")), identical,
    )["passed"]


def importance_row(**changes):
    return dict(finite_full_support=True, mean_actor_clip_mass=.01, max_actor_clip_mass=.01,
                mean_native_clip_mass=.02, max_native_clip_mass=.02,
                min_effective_sample_fraction=.99, max_normalization_error=1e-7,
                passed=True) | changes


def test_importance_summary_weights_actual_token_opportunities_and_keeps_worst_case_diagnostics():
    rows = [importance_row(), importance_row(
        mean_actor_clip_mass=.09, max_actor_clip_mass=.09,
        mean_native_clip_mass=.10, max_native_clip_mass=.10, passed=False,
    )]
    result = qualification.summarize_importance_controls(rows, [4, 1])
    assert result["passed"]
    assert result["mean_actor_clip_mass"] == pytest.approx(.026)
    assert result["mean_native_clip_mass"] == pytest.approx(.036)
    assert result["active_row_token_comparisons"] == 5
    assert result["max_native_clip_mass"] == .10
    assert not result["per_conditional_max_bound_passed"]
    assert result["comparisons"] == rows
    # A sustained excess fails the same 5% clipped-token bound.
    assert not qualification.summarize_importance_controls([rows[1], rows[1]], [4, 1])["passed"]


@pytest.mark.parametrize('changes', [
    dict(mean_actor_clip_mass=.051, max_actor_clip_mass=.051),
    dict(mean_native_clip_mass=.051, max_native_clip_mass=.051),
    dict(min_effective_sample_fraction=.949),
    dict(max_normalization_error=1.1e-5),
    dict(finite_full_support=False),
    dict(mean_actor_clip_mass=float('nan')),
    dict(max_native_clip_mass=float('inf')),
    dict(min_effective_sample_fraction=float('nan')),
    dict(max_normalization_error=float('nan')),
    dict(mean_actor_clip_mass=-.01),
    dict(mean_native_clip_mass=.02, max_native_clip_mass=.01),
])
def test_importance_summary_rejects_corrupted_or_unhealthy_controls(changes):
    assert not qualification.summarize_importance_controls([importance_row(**changes)], [1])["passed"]


@pytest.mark.parametrize('rows,counts', [
    ([], []), ([importance_row()], []), ([importance_row()], [1, 2]),
    ([importance_row()], [0]), ([importance_row()], [-1]),
    ([importance_row()], [1.5]), ([importance_row()], [True]),
])
def test_importance_summary_rejects_missing_or_invalid_active_row_counts(rows, counts):
    with pytest.raises(ValueError, match='active-row count'):
        qualification.summarize_importance_controls(rows, counts)


def test_gradient_oracle_catches_missing_zero_and_nonfinite_tensors():
    model = nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    report = qualification.gradient_report(model)
    assert report["all_trainable_gradients_finite"] and report["all_trainable_gradients_nonzero"]
    assert report["gradient_norm"] > 0
    model.weight.grad = None
    assert qualification.gradient_report(model)["missing"] == ["weight"]
    model.weight.grad = torch.zeros_like(model.weight)
    assert qualification.gradient_report(model)["zero"] == ["weight"]
    model.weight.grad.fill_(float("nan"))
    report = qualification.gradient_report(model)
    assert report["nonfinite"] == ["weight"]
    assert not report["all_trainable_gradients_finite"] and not report["all_trainable_gradients_nonzero"]


def test_fingerprint_checks_every_parameter_value_and_dtype():
    model = nn.Linear(2, 2)
    before = qualification.parameter_fingerprint(model)
    with torch.no_grad():
        model.weight[-1, -1] += 1
    assert qualification.parameter_fingerprint(model) != before
    before = qualification.parameter_fingerprint(model)
    model.bfloat16()
    assert qualification.parameter_fingerprint(model) != before


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 2)
        self.layers = nn.ModuleList([nn.Linear(2, 2)])

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.layers[0](self.embedding(input_ids)))


class ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBackbone()
        self.head = nn.Linear(2, 8)

    def _softcapped_logits(self, hidden):
        return self.head(hidden)


def test_backward_oracle_reuses_chunked_native_loss_and_restores_gradients(monkeypatch):
    model = ToyPolicy().eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    grads = {name: parameter.grad for name, parameter in model.named_parameters()}
    original = qualification.parameter_fingerprint(model)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1024**3)
    actual_loss_sum = qualification.loss_sum
    shapes = []

    def loss_sum(policy, ids, targets, **kwargs):
        shapes.append((ids.shape, targets.shape, kwargs, int((targets == -100).sum())))
        return actual_loss_sum(policy, ids, targets, **kwargs)

    monkeypatch.setattr(qualification, "loss_sum", loss_sum)
    report = qualification.backward_oracle(model, torch.tensor([[1, 2, 3]]), 12, 4)
    assert report["passed"] and report["backward_length"] == 12 and report["response_tokens"] == 8
    assert report["weights_unchanged"] and report["gradients_restored"] and report["training_mode_restored"]
    assert shapes == [(torch.Size([1, 12]), torch.Size([1, 12]),
                       dict(chunk=512, checkpoint_head=True), 4)]
    assert qualification.parameter_fingerprint(model) == original
    assert not model.training
    assert all(parameter.grad is grads[name] for name, parameter in model.named_parameters())


def test_backward_exception_restores_existing_gradients_and_mode(monkeypatch):
    model = ToyPolicy().eval()
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    grad = parameter.grad
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate admission failure")

    monkeypatch.setattr(qualification, "loss_sum", fail)
    with pytest.raises(RuntimeError, match="deliberate admission failure"):
        qualification.backward_oracle(model, torch.tensor([[1, 2, 3]]), 12, 4)
    assert parameter.grad is grad and not model.training
