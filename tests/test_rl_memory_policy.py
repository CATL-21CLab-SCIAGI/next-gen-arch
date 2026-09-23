"""Numerical oracles for the opt-in replay memory policy in the DLC runtime."""

from types import SimpleNamespace

import pytest
import torch

from archlab.automodel.deepseek_v41_rl_memory_policy import (
    inplace_native_expert_function,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the existing B300 runtime")
@pytest.mark.parametrize("first_expert", [0, 2])
def test_inplace_expert_sum_has_exact_outputs_and_all_input_gradients(first_expert):
    from archlab.automodel.deepseek_v41_official_moe import _native_up_grouped_down

    torch.manual_seed(788)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    owner = SimpleNamespace(config=SimpleNamespace(swiglu_limit=7.0))
    tokens, hidden, middle, experts = 128, 64, 32, 4
    x = torch.randn(tokens, hidden, device="cuda", requires_grad=True)
    up = (
        torch.randn(experts, hidden, middle * 2, device="cuda", dtype=torch.bfloat16) * 0.03
    ).requires_grad_()
    down = (
        torch.randn(experts, middle, hidden, device="cuda", dtype=torch.bfloat16) * 0.03
    ).requires_grad_()
    scores = torch.randn(tokens, experts + first_expert, device="cuda").softmax(-1)
    weights, indices = scores.topk(3)
    weights.requires_grad_()
    mask = torch.arange(tokens, device="cuda") % 5 != 0
    originals = (x, weights, up, down)
    copies = tuple(t.detach().clone().requires_grad_() for t in originals)
    first = _native_up_grouped_down(
        owner, x, mask, weights, indices, up, down, experts, first_expert
    )
    second = inplace_native_expert_function()(
        owner, copies[0], mask, copies[1], indices, copies[2], copies[3], experts, first_expert
    )
    upstream = torch.randn_like(first)
    first.backward(upstream)
    second.backward(upstream)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for original, candidate in zip(originals, copies, strict=False):
        assert original.grad is not None
        torch.testing.assert_close(original.grad, candidate.grad, rtol=0, atol=0)
