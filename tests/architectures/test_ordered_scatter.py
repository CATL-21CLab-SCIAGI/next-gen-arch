import pytest
import torch

from archlab.architectures.ordered_scatter import ordered_permutation_sum


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_ordered_permutation_matches_original_sum_and_gradient(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires the existing GPU runtime")
    torch.manual_seed(226)
    tokens, slots, hidden = 41, 6, 32
    positions = torch.full((tokens, slots), -1, device=device, dtype=torch.long)
    active = torch.randperm(tokens * slots, device=device)[:173]
    positions.view(-1)[active] = torch.arange(active.numel(), device=device)
    values = torch.randn(active.numel(), hidden, device=device, dtype=dtype, requires_grad=True)
    candidate = values.detach().clone().requires_grad_()
    expected = torch.zeros(tokens, hidden, device=device, dtype=torch.float32)
    for slot in range(slots):
        rows = torch.where(positions[:, slot] >= 0)[0]
        expected = expected.index_add(0, rows, values[positions[rows, slot]].float())
    actual = ordered_permutation_sum(candidate, positions, chunk_rows=3)
    cotangent = torch.randn_like(expected)
    expected.backward(cotangent)
    actual.backward(cotangent)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(candidate.grad, values.grad, rtol=0, atol=0)


def test_permutation_contract_rejects_missing_or_repeated_sources():
    values = torch.randn(2, 4, requires_grad=True)
    for indices in ([0, 0], [0, -1], [0, 2], [0, -2]):
        with pytest.raises(ValueError, match="permutation"):
            ordered_permutation_sum(values, torch.tensor([indices]))
    empty = torch.empty(0, 4, requires_grad=True)
    output = ordered_permutation_sum(empty, torch.full((3, 2), -1))
    output.sum().backward()
    assert not output.count_nonzero() and empty.grad.shape == empty.shape


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the existing B300 runtime")
def test_production_shape_reduction_memory_and_exact_gradient():
    import json
    import time

    torch.manual_seed(908)
    tokens, slots, width = 65536, 6, 5120
    positions = torch.arange(tokens * slots, device="cuda").reshape(tokens, slots)
    values = torch.randn(tokens * slots, width, device="cuda", dtype=torch.bfloat16)
    cotangent = torch.randn(tokens, width, device="cuda", dtype=torch.float32)

    def original(x):
        output = torch.zeros(tokens, width, device="cuda", dtype=torch.float32)
        rows = torch.arange(tokens, device="cuda")
        for slot in range(slots):
            output = output.index_add(0, rows, x[positions[:, slot]].float())
        return output

    def measure(function):
        x = values.clone().requires_grad_()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        initial = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        output = function(x)
        output.backward(cotangent)
        torch.cuda.synchronize()
        return (
            output.detach(),
            x.grad,
            {
                "peak_additional_gib": (torch.cuda.max_memory_allocated() - initial) / 2**30,
                "seconds": time.perf_counter() - began,
            },
        )

    reference, reference_gradient, first = measure(original)
    actual, actual_gradient, second = measure(
        lambda x: ordered_permutation_sum(x, positions, validate=False)
    )
    assert torch.equal(actual, reference) and torch.equal(actual_gradient, reference_gradient)
    assert second["peak_additional_gib"] < first["peak_additional_gib"]
    print(
        json.dumps(
            {
                "scope": "expert combination only; worst-case single-owner dispatch; no model update",
                "timing": "single cold smoke; not a steady-state throughput result",
                "tokens": tokens,
                "slots": slots,
                "width": width,
                "reference": first,
                "bounded": second,
                "forward_and_backward_exact": True,
            }
        )
    )
