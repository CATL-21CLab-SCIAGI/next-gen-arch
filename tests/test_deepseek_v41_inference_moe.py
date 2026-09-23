import torch
from torch.nn import functional as F

from archlab.architectures.deepseek_v41_inference_moe import (
    expert_output,
    owned_experts,
    router_weights,
)


def test_correction_bias_selects_experts_without_scaling_probabilities():
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    gate = torch.zeros(4, 8, dtype=torch.bfloat16)
    probability, ids = router_weights(x, gate, torch.tensor([100., -100., 20., 0.]),
                                      top_k=2, route_scale=1.5)
    assert ids.tolist() == [[0, 2], [0, 2]]
    torch.testing.assert_close(probability, torch.full((2, 2), .75), rtol=0, atol=0)


def test_expert_probability_is_applied_before_bf16_activation_rounding():
    torch.manual_seed(97)
    x = torch.randn(7, 16).bfloat16()
    gate_up = torch.randn(24, 16).bfloat16()
    down = torch.randn(16, 12).bfloat16()
    probability = torch.rand(7, 1)
    actual = expert_output(x, gate_up, down, limit=10., probability=probability)
    gate = (x.float() @ gate_up[:12].float().T).bfloat16().float().clamp(max=10.)
    up = (x.float() @ gate_up[12:].float().T).bfloat16().float().clamp(-10., 10.)
    activation = (F.silu(gate) * up * probability).bfloat16()
    expected = (activation.float() @ down.float().T).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    wrong = (expert_output(x, gate_up, down, limit=10.).float() * probability).bfloat16()
    assert not torch.equal(actual, wrong)


def test_ep_partitions_cover_same_routed_experts_with_fp32_subtotals():
    torch.manual_seed(101)
    x = torch.randn(5, 16).bfloat16()
    gate_up = torch.randn(4, 24, 16).bfloat16()
    down = torch.randn(4, 16, 12).bfloat16()
    probabilities, indices = router_weights(x, torch.randn(4, 16).bfloat16(),
                                            torch.randn(4), top_k=3, route_scale=1.5)
    whole = owned_experts(x, probabilities, indices, gate_up, down, first_expert=0, limit=10.)
    first = owned_experts(x, probabilities, indices, gate_up[:2], down[:2], first_expert=0, limit=10.)
    second = owned_experts(x, probabilities, indices, gate_up[2:], down[2:], first_expert=2, limit=10.)
    assert whole.dtype == first.dtype == second.dtype == torch.float32
    torch.testing.assert_close(first + second, whole, rtol=0, atol=0)
