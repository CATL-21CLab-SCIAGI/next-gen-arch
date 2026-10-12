"""Native sigmoid head values, gradients and bounded retained vocabulary state."""
from __future__ import annotations

import pytest
import torch

from archlab.architectures.limite_head import chunked_logits_processor


def processor(logits, *, label, temperature):
    temperature.clamp_(min=.5)  # Match package processors' in-place normalization.
    logp = (logits / temperature[..., None]).log_softmax(-1)
    return {"log_probs": logp.gather(-1, label[..., None]).squeeze(-1),
            "entropy": -(logp.exp() * logp).sum(-1)}


@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_values_native_softcap_gradients_and_mutable_processor_inputs(chunk):
    torch.manual_seed(1)
    hidden = torch.randn(2, 7, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(11, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 11, (2, 7))
    temperature = torch.rand(2, 7, dtype=torch.float64)
    args = dict(label=labels, temperature=temperature)
    advantages = torch.randn(2, 7, dtype=torch.float64)

    def head(x):
        return 23 * ((x @ weight.T + 5) / 7.5).sigmoid()

    expected = processor(head(hidden), label=labels, temperature=temperature.clone())
    want = torch.autograd.grad((expected["log_probs"] * advantages).sum()
                               + .1 * expected["entropy"].sum(), (hidden, weight))
    before = temperature.clone()
    actual = chunked_logits_processor(head, hidden, processor, args, chunk_size=chunk)
    got = torch.autograd.grad((actual["log_probs"] * advantages).sum()
                              + .1 * actual["entropy"].sum(), (hidden, weight))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-12, atol=1e-12)
    for actual_grad, expected_grad in zip(got, want, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)
    assert torch.equal(temperature, before)


def test_backward_does_not_retain_full_sequence_vocabulary_logits():
    hidden = torch.randn(1, 103, 8, requires_grad=True)
    weight = torch.randn(127, 8, requires_grad=True)
    saved = []
    projected = []

    def head(x):
        projected.append(x.shape[1])
        return 23 * ((x @ weight.T + 5) / 7.5).sigmoid()

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = chunked_logits_processor(
            head, hidden, processor,
            dict(label=torch.zeros(1, 103, dtype=torch.long), temperature=torch.ones(1, 103)),
            chunk_size=16,
        )
    assert not any(shape and shape[-1] == 127 for shape in saved)
    output["log_probs"].sum().backward()
    assert hidden.grad.isfinite().all() and weight.grad.isfinite().all()
    assert max(projected) <= 16


def test_bad_geometry_fails_before_projection():
    with pytest.raises(ValueError, match="arguments"):
        chunked_logits_processor(None, torch.zeros(1, 5, 7), processor,
                                 {"label": torch.zeros(1, 4, dtype=torch.long)})
