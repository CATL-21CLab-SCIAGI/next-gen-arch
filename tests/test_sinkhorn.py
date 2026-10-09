import torch

from archlab.optimizers.sinkhorn import sinkhorn_step


def oracle(weight, gradient, momentum, lr, dtype):
    momentum = (.95 * momentum.float() + .05 * gradient).to(dtype)
    update = .95 * momentum.float() + .05 * gradient
    norms = update.norm(dim=1)
    update[norms <= .001 * norms.mean()] = 0
    for i in range(11):
        update = update / (update.norm(dim=1 if i % 2 == 0 else 0, keepdim=True) + 1e-20)
    return weight - lr * .18 * weight.shape[1] ** .5 * update, momentum


def test_sinkhorn_chunked_matches_dense_and_zero_rows():
    torch.manual_seed(72)
    weight = torch.randn(19, 11)
    expected = weight.clone()
    momentum = torch.zeros_like(weight, dtype=torch.bfloat16)
    reference_momentum = momentum.clone()
    for _ in range(5):
        gradient = torch.randn_like(weight)
        gradient[0] = 0
        gradient[1] *= 1e-8
        expected, reference_momentum = oracle(expected, gradient, reference_momentum, 3e-6, torch.bfloat16)
        sinkhorn_step(weight, gradient, momentum, lr=3e-6, chunk_rows=3)
        torch.testing.assert_close(weight, expected, atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(momentum, reference_momentum, atol=0, rtol=0)


def test_zero_update_is_finite_and_preserves_weights():
    weight = torch.randn(7, 5)
    original = weight.clone()
    sinkhorn_step(weight, torch.zeros_like(weight), torch.zeros_like(weight, dtype=torch.bfloat16), lr=3e-6)
    torch.testing.assert_close(weight, original, atol=0, rtol=0)
