"""Numerical gates for container-owned fusions used by the DP-only recipe."""

import os

import pytest
import torch

from archlab.megatron.qwen38_flash_next_sample import select_token


def test_sampling_greedy_and_nucleus_keep_the_highest_probability_token():
    logits = torch.tensor([[0.0, 4.0, -1.0]])
    assert select_token(logits, temperature=0, top_p=1).item() == 1
    assert select_token(logits, temperature=1, top_p=0.1).item() == 1
    with pytest.raises(RuntimeError, match="nonfinite"):
        select_token(torch.tensor([[float("nan")]]), temperature=0, top_p=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the frozen CUDA runtime")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_fused_permutation_preserves_outputs_and_token_probability_gradients(dtype):
    utils = pytest.importorskip("megatron.core.transformer.moe.moe_utils")
    torch.manual_seed(83)
    tokens = torch.randn(64, 384, device="cuda", dtype=dtype)
    routing = torch.zeros(64, 64, device="cuda", dtype=torch.bool)
    routing.scatter_(1, torch.rand(64, 64, device="cuda").topk(3, dim=1).indices, True)
    probabilities = torch.rand(64, 64, device="cuda", dtype=dtype) * routing
    weight = torch.randn_like(tokens)

    def evaluate(fused):
        x = tokens.detach().clone().requires_grad_()
        p = probabilities.detach().clone().requires_grad_()
        permuted, permuted_p, indices, _, _ = utils.permute(
            x, routing, probs=p, num_out_tokens=192, fused=fused
        )
        expert_output = torch.nn.functional.silu(permuted) * permuted_p.unsqueeze(-1)
        restored = utils.unpermute(
            expert_output, indices, x.shape, routing_map=routing, fused=fused
        )
        (restored * weight).float().sum().backward()
        return restored.detach(), x.grad, p.grad

    reference, actual = evaluate(False), evaluate(True)
    tolerance = (
        dict(atol=0.04, rtol=0.02) if dtype == torch.bfloat16 else dict(atol=2e-5, rtol=2e-5)
    )
    for expected, observed in zip(reference, actual, strict=True):
        torch.testing.assert_close(observed, expected, **tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the frozen CUDA runtime")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_fused_router_preserves_top3_scores_and_gradient(dtype):
    utils = pytest.importorskip("megatron.core.transformer.moe.moe_utils")
    torch.manual_seed(84)
    # Distinct representable scores avoid making tie-breaking part of the oracle.
    values = torch.stack([torch.randperm(64, device="cuda") for _ in range(64)]).to(dtype) / 16
    weights = torch.randn_like(values)

    def evaluate(fused):
        logits = values.detach().clone().requires_grad_()
        probs, mapping = utils.topk_routing_with_score_function(logits, 3, fused=fused)
        (probs * weights).float().sum().backward()
        return probs.detach(), mapping, logits.grad

    expected, actual = evaluate(False), evaluate(True)
    assert torch.equal(actual[1], expected[1])
    for index in (0, 2):
        torch.testing.assert_close(actual[index], expected[index], atol=0.004, rtol=0.02)


@pytest.mark.skipif(
    not torch.cuda.is_available() or "RANK" not in os.environ,
    reason="run under single-rank torchrun in the frozen CUDA runtime",
)
def test_native_fused_cross_entropy_preserves_loss_and_gradients():
    from megatron.core import parallel_state
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

    created_group = not torch.distributed.is_initialized()
    if created_group:
        torch.distributed.init_process_group("nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    try:
        group = torch.distributed.group.WORLD
        assert group.size() == 1
        torch.manual_seed(85)
        initial = torch.randn(16, 2, 1024, device="cuda", dtype=torch.bfloat16)
        targets = torch.randint(0, 1024, (16, 2), device="cuda")
        reference = initial.detach().clone().requires_grad_()
        actual = initial.detach().clone().requires_grad_()
        expected_loss = vocab_parallel_cross_entropy(reference, targets)
        actual_loss = fused_vocab_parallel_cross_entropy(actual, targets, group)
        torch.testing.assert_close(actual_loss, expected_loss, atol=2e-5, rtol=2e-5)
        expected_loss.mean().backward()
        actual_loss.mean().backward()
        torch.testing.assert_close(actual.grad, reference.grad, atol=0.0003, rtol=0.02)
    finally:
        parallel_state.destroy_model_parallel()
        if created_group:
            torch.distributed.destroy_process_group()
