import pytest
import torch
from torch.nn import functional as F

from archlab.automodel.deepseek_v41_loss import FrozenHeadCrossEntropy


@pytest.mark.parametrize("chunk", [1, 3, 128])
def test_chunked_loss_and_input_gradient_match_pytorch(chunk):
    torch.manual_seed(992)
    hidden = torch.randn(2, 5, 7, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    weight = torch.randn(23, 7)
    labels = torch.randint(0, 23, (2, 5))
    labels[:, 1] = -100
    actual = FrozenHeadCrossEntropy.apply(hidden, labels, weight, chunk)
    expected = F.cross_entropy(F.linear(other, weight).flatten(0, 1), labels.flatten(), reduction="sum")
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, other.grad)


def test_empty_targets_still_participate_in_backward():
    hidden = torch.randn(1, 5, 7, requires_grad=True)
    loss = FrozenHeadCrossEntropy.apply(hidden, torch.full((1, 5), -100), torch.randn(23, 7), 3)
    loss.backward()
    assert loss == 0 and torch.equal(hidden.grad, torch.zeros_like(hidden))
