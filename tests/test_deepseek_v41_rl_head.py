from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from archlab.automodel.deepseek_v41_rl_head import install_rl_head, selected_log_probs


@pytest.mark.parametrize("chunk_size", [1, 4, 100])
def test_signed_upstream_matches_dense_hidden_and_head_gradients(chunk_size):
    generator = torch.Generator().manual_seed(831)
    hidden = torch.randn(2, 7, 5, generator=generator, requires_grad=True)
    weight = torch.randn(17, 5, generator=generator, requires_grad=True)
    labels = torch.randint(0, 17, (2, 7), generator=generator)
    labels[0, :3] = -100
    labels[1, -2:] = -100
    upstream = torch.randn(2, 7, generator=generator)
    upstream[:, ::2] *= -3
    expected = F.linear(hidden, weight).log_softmax(-1).gather(
        -1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    expected = expected.masked_fill(labels == -100, 0)
    actual = selected_log_probs(hidden, labels, weight, chunk_size)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual, (hidden, weight), upstream)
    expected_grads = torch.autograd.grad(expected, (hidden, weight), upstream)
    for observed, oracle in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(observed, oracle, rtol=3e-6, atol=2e-6)
    assert torch.equal(actual_grads[0][labels == -100], torch.zeros_like(actual_grads[0][labels == -100]))


def test_signed_sequence_policy_loss_and_ignored_batch():
    torch.manual_seed(18)
    hidden = torch.randn(3, 6, 4, requires_grad=True)
    weight = torch.randn(11, 4, requires_grad=True)
    labels = torch.randint(0, 11, (3, 6))
    labels[:, :2] = -100
    labels[2] = -100
    advantage = torch.tensor([1.5, -1.5, 99.])
    logp = selected_log_probs(hidden, labels, weight, 3)
    loss = -(logp.sum(-1) * advantage).mean()
    dense = F.linear(hidden, weight).log_softmax(-1)
    selected = dense.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    expected = -(selected.masked_fill(labels == -100, 0).sum(-1) * advantage).mean()
    torch.testing.assert_close(loss, expected)
    for actual, oracle in zip(torch.autograd.grad(loss, (hidden, weight)),
                              torch.autograd.grad(expected, (hidden, weight)), strict=True):
        torch.testing.assert_close(actual, oracle, atol=2e-6, rtol=3e-6)


def test_all_ignored_has_zero_values_and_derivatives():
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(7, 4, requires_grad=True)
    output = selected_log_probs(hidden, torch.full((2, 3), -100), weight, 2)
    output.backward(torch.randn_like(output))
    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(hidden.grad) == 0
    assert torch.count_nonzero(weight.grad) == 0


@pytest.mark.parametrize("train_hidden,train_weight", [(True, False), (False, True)])
def test_one_trainable_input_and_no_persistent_vocabulary_logits(train_hidden, train_weight):
    hidden = torch.randn(2, 8, 4, requires_grad=train_hidden)
    weight = torch.randn(19, 4, requires_grad=train_weight)
    labels = torch.randint(0, 19, (2, 8))
    shapes = []

    def pack(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = selected_log_probs(hidden, labels, weight, 3)
    assert shapes == [(2, 8, 4), (2, 8), (19, 4), (16,)]
    result.sum().backward()
    assert (hidden.grad is not None) == train_hidden
    assert (weight.grad is not None) == train_weight


def test_install_keeps_existing_ce_and_parameters():
    head = torch.nn.Linear(4, 11, bias=False)
    original_loss = lambda hidden, labels: hidden.sum()  # noqa: E731
    head.loss = original_loss
    model = SimpleNamespace(lm_head=head)
    original_weight = head.weight
    receipt = install_rl_head(model)
    assert receipt["method"] == "rl_log_probs"
    assert head.loss is original_loss and head.weight is original_weight
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    labels = torch.randint(0, 11, (2, 3))
    head.rl_log_probs(hidden, labels).sum().backward()
    assert head.weight.grad is not None and hidden.grad is not None
    with pytest.raises(ValueError, match="already installed"):
        install_rl_head(model)


def test_validation_and_autocast_precision():
    hidden, weight = torch.randn(2, 3, 4), torch.randn(7, 4)
    labels = torch.randint(0, 7, (2, 3))
    expected = selected_log_probs(hidden, labels, weight, 2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = selected_log_probs(hidden, labels, weight, 2)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="outside"):
        selected_log_probs(hidden, torch.full_like(labels, 7), weight)
    with pytest.raises(ValueError, match="positive"):
        selected_log_probs(hidden, labels, weight, 0)
