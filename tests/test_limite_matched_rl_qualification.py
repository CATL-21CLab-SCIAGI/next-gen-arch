from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from archlab.automodel import limite_matched_rl_qualification as qualification
from archlab.automodel.checkpoint_oracle import assert_state_equal
from archlab.rl.limite_checkpoint import capture_rng
from archlab.rl.limite_scoring import enable_chunked_policy_scores
from archlab.rl.limite_update_oracle import first_adam_updates


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(11, 4)
        self.layer = nn.Linear(4, 4)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.layer(self.embedding(input_ids)).tanh())


class ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBackbone()
        self.head = nn.Linear(4, 11)
        self.config = SimpleNamespace(max_position_embeddings=32)
        self.archlab_native_checkpoint = True

    def _softcapped_logits(self, hidden):
        return 3 * (self.head(hidden) / 3).tanh()

    def forward(self, input_ids, logits_to_keep=0, **kwargs):
        hidden = self.model(input_ids).last_hidden_state
        return SimpleNamespace(logits=self._softcapped_logits(hidden[:, -logits_to_keep:]))


def test_score_gate_rejects_biased_nonfinite_and_clipped_policy_ratios():
    expected = torch.tensor([[-1., -2.]])
    assert qualification.score_error(expected, expected)["passed"]
    assert not qualification.score_error(expected + .11, expected)["passed"]
    assert not qualification.score_error(expected + 1., expected)["passed"]
    assert not qualification.score_error(torch.full_like(expected, float("nan")), expected)["passed"]


def test_gradient_gate_checks_all_parameters_and_rejects_wrong_gradients():
    model = ToyPolicy()
    expected = {}
    for name, parameter in model.named_parameters():
        parameter.grad = torch.ones_like(parameter)
        expected[name] = parameter.grad.clone()
    assert qualification.gradient_error(model, expected)["passed"]
    parameter = model.model.layer.weight
    parameter.grad.mul_(1.5)
    assert not qualification.gradient_error(model, expected)["passed"]
    parameter.grad = None
    assert qualification.gradient_error(model, expected)["missing"] == ["model.layer.weight"]


def test_noise_admission_requires_equivalent_update_for_cancelling_scalar_gradients():
    model = nn.Linear(4, 1)
    with torch.no_grad():
        model.weight.fill_(1.)
        model.bias.fill_(1.)
    parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    expected = dict(weight=torch.ones_like(model.weight), bias=torch.full_like(model.bias, 1e-4))
    repeats = [dict(weight=expected["weight"].clone(), bias=expected["bias"] + noise)
               for noise in (1e-5, -1e-5)]
    reference_updates = first_adam_updates(parameters, expected)[0]
    repeat_updates = [first_adam_updates(parameters, gradients)[0] for gradients in repeats]
    actual = dict(weight=expected["weight"].clone(), bias=expected["bias"] * 2)
    for name, parameter in model.named_parameters():
        parameter.grad = actual[name]
    assert not qualification.gradient_error(model, expected)["passed"]
    update_report = qualification.update_error(first_adam_updates(parameters, actual)[0],
                                                reference_updates, repeat_updates)
    assert update_report["passed"]
    assert qualification.gradient_error(model, expected, repeat_gradients=repeats,
                                         update_report=update_report)["passed"]
    # A sign flip in the same low-magnitude scalar changes Adam's real update;
    # it must fail even though its contribution to global gradient error is tiny.
    actual["bias"].neg_()
    update_report = qualification.update_error(first_adam_updates(parameters, actual)[0],
                                                reference_updates, repeat_updates)
    assert not update_report["passed"]
    assert not qualification.gradient_error(model, expected, repeat_gradients=repeats,
                                             update_report=update_report)["passed"]


def test_replay_oracle_catches_alignment_and_restores_gradients_rng_and_mode():
    torch.manual_seed(7)
    reference = ToyPolicy().eval()
    optimized = deepcopy(reference)
    enable_chunked_policy_scores(optimized, chunk_size=3)
    for model in (reference, optimized):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
    gradients = [{name: parameter.grad for name, parameter in model.named_parameters()}
                 for model in (reference, optimized)]
    rng = capture_rng()
    report = qualification.replay_oracle(reference, optimized, torch.tensor([[1, 2, 3]]), 13)
    assert report["passed"]
    assert report["repeat_reference_backwards"] == 3
    assert report["prompt_tokens"] == 7 and report["completion_tokens"] == 6
    assert report["gradients"]["relative_l2"] < 1e-5
    assert_state_equal(capture_rng(), rng)
    for model, prior in zip((reference, optimized), gradients, strict=True):
        assert not model.training
        assert all(parameter.grad is prior[name] for name, parameter in model.named_parameters())
    with torch.no_grad():
        optimized.head.bias[1].add_(2.)
    assert not qualification.replay_oracle(reference, optimized, torch.tensor([[1, 2, 3]]), 13)["passed"]


def test_full_context_stress_uses_actual_limit_and_preserves_state():
    torch.manual_seed(11)
    model = ToyPolicy().eval()
    enable_chunked_policy_scores(model, chunk_size=3)
    prior = next(model.parameters())
    prior.grad = torch.ones_like(prior)
    previous_gradient = prior.grad
    before, rng = qualification.parameter_fingerprint(model), capture_rng()
    report = qualification.full_context_oracle(model, torch.tensor([[1, 2, 3]]), 32, 5)
    assert report["passed"] and report["sequence_length"] == 32 and report["completion_tokens"] == 27
    assert report["weights_unchanged"] and report["gradients_restored"] and report["training_mode_restored"]
    assert qualification.parameter_fingerprint(model) == before
    assert prior.grad is previous_gradient and not model.training
    assert_state_equal(capture_rng(), rng)
    with pytest.raises(ValueError, match="native context"):
        qualification.full_context_oracle(model, torch.tensor([[1, 2, 3]]), 33, 5)


def test_stress_failure_restores_existing_gradients_mode_and_rng(monkeypatch):
    model = ToyPolicy().eval()
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    gradient, rng = parameter.grad, capture_rng()

    def fail(*args):
        torch.rand(3)
        raise RuntimeError("deliberate failure")

    monkeypatch.setattr(qualification, "_replay_scores", fail)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        qualification.full_context_oracle(model, torch.tensor([[1, 2, 3]]), 32, 5)
    assert parameter.grad is gradient and not model.training
    assert_state_equal(capture_rng(), rng)
