"""Numerical oracles for the opt-in replay memory policy in the DLC runtime."""

from types import SimpleNamespace

import pytest
import torch

from archlab.automodel.deepseek_v41_rl_memory_policy import (
    inplace_native_expert_function,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the existing B300 runtime")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_hc_host_roundtrip_is_exact_and_deduplicates_saved_views(dtype):
    import copy
    from types import MethodType

    from archlab.architectures.deepseek_v41_math import hc_split_sinkhorn
    from archlab.automodel.deepseek_v41_full_boundaries import trainable_hc_forward
    from archlab.automodel.deepseek_v41_rl_memory_policy import (
        hc_offload_statistics,
        install_hc_activation_offload,
    )

    torch.manual_seed(908)
    reference = torch.nn.Module()
    reference.streams, reference.iterations = 4, 20
    reference.eps, reference.norm_eps = 1e-6, 1e-6
    reference.fn = torch.nn.Parameter(torch.randn(24, 4 * 16, device="cuda") * 0.01)
    reference.scale = torch.nn.Parameter(torch.ones(3, device="cuda"))
    reference.base = torch.nn.Parameter(torch.zeros(24, device="cuda"))
    reference._archlab_native_hc = hc_split_sinkhorn
    reference.forward = MethodType(trainable_hc_forward, reference)
    candidate = copy.deepcopy(reference)
    before = {name: id(p) for name, p in candidate.named_parameters()}
    install_hc_activation_offload(candidate)
    x = torch.randn(2, 17, 4, 16, device="cuda", dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = reference(x), candidate(y)
    for key in ("pre", "post", "comb"):
        torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=0, atol=0)
    cotangents = [torch.randn_like(getattr(a, key)) for key in ("pre", "post", "comb")]
    torch.autograd.backward([a.pre, a.post, a.comb], cotangents)
    torch.autograd.backward([b.pre, b.post, b.comb], cotangents)
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for (name, p), (_, q) in zip(
        reference.named_parameters(), candidate.named_parameters(), strict=True
    ):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        assert id(q) == before[name] and q.device.type == "cuda"
    assert hc_offload_statistics(candidate) == {
        "tensor_copies": 1,
        "copied_bytes": x.numel() * x.element_size(),
    }
    with torch.no_grad():
        candidate(x)
    assert hc_offload_statistics(candidate)["tensor_copies"] == 1


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
