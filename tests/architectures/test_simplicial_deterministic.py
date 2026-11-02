"""Independent joint-attention oracle and deterministic-gradient regression."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the frozen CUDA runtime")
@pytest.mark.parametrize("length,short,long", [(9, 3, 5), (33, 1, 1), (137, 4, 17)])
def test_deterministic_simplicial_matches_oracle_and_repeats(length, short, long):
    from archlab.architectures.simplicial_attention import reference_simplicial
    from archlab.architectures.simplicial_deterministic import deterministic_simplicial_attention

    torch.manual_seed(912)
    originals = [torch.randn(1, length, heads, 16, device="cuda") * .5 for heads in (4, 2, 2, 2, 2)]
    cotangent = torch.randn_like(originals[0])
    results = []
    for fn in (reference_simplicial, *[deterministic_simplicial_attention] * 3):
        inputs = [value.clone().requires_grad_() for value in originals]
        output = fn(*inputs, short, long)
        output.backward(cotangent)
        results.append((output.detach(), *[value.grad for value in inputs]))
    for actual, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
    for repeated in results[2:]:
        assert all(torch.equal(value, expected) for value, expected in zip(repeated, results[1], strict=True))
    if short == long == 1:
        assert all(not value.count_nonzero() for value in results[1][1:4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the frozen CUDA runtime")
def test_deterministic_core_does_not_change_original_atomic_api_guard():
    from archlab.architectures.simplicial_attention import simplicial_attention
    from archlab.architectures.simplicial_deterministic import deterministic_simplicial_attention

    old, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    values = [torch.randn(1, 3, heads, 16, device="cuda", requires_grad=True) for heads in (2, 1, 1, 1, 1)]
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="nondeterministic"):
            simplicial_attention(*values, 2, 3)
        deterministic_simplicial_attention(*values, 2, 3).sum().backward()
        assert all(value.grad.isfinite().all() for value in values)
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn)
