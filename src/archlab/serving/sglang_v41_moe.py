"""Precision-preserving MoE execution over SGLang-owned BF16 EP weights."""

from __future__ import annotations

from types import MethodType

import torch

from archlab.architectures.deepseek_v41_inference_moe import (
    expert_output,
    owned_experts,
    router_weights,
)


def install_native_moe(moe, *, tp_rank, tp_size, all_reduce):
    c = moe.config
    if (tp_size not in (8, 32) or c.n_routed_experts % tp_size or moe.num_fused_shared_experts
            or not moe._shared_expert_tp1 or moe.is_hash
            or c.scoring_func != "sqrtsoftplus" or not c.norm_topk_prob
            or c.n_shared_experts != 1 or c.swiglu_limit <= 0):
        raise ValueError("native MoE requires EP8/EP32, one replicated shared expert, and V4.1 routing")
    if "forward" in moe.__dict__:
        raise ValueError("MoE execution was already extended")
    local_experts = c.n_routed_experts // tp_size

    @torch.inference_mode()
    def forward(self, hidden_states, forward_batch=None, gemm_output_zero_allocator=None,
                input_ids=None, input_ids_global=None, skip_shared_experts=False):
        if skip_shared_experts or hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
            raise ValueError("native MoE requires complete BF16 text-token inputs")
        if tp_size == 32:
            from sglang.srt.runtime_context import get_forward
            if get_forward().mlp_reduce_scatter:
                raise ValueError("native EP32 requires gathered inputs and exactly one FP32 all-reduce")
        if hidden_states.shape[0] == 0:
            return hidden_states
        gate_up, down = self.experts.w13_weight, self.experts.w2_weight
        if (gate_up.shape != (local_experts, 2 * c.moe_intermediate_size, c.hidden_size)
                or down.shape != (local_experts, c.hidden_size, c.moe_intermediate_size)
                or gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16):
            raise ValueError("SGLang expert layout/precision differs from the reviewed native path")
        probability, indices = router_weights(hidden_states, self.gate.weight,
                                              self.gate.e_score_correction_bias,
                                              top_k=c.num_experts_per_tok,
                                              route_scale=c.routed_scaling_factor)
        total = owned_experts(hidden_states, probability, indices, gate_up, down,
                              first_expert=tp_rank * local_experts, limit=c.swiglu_limit)
        total = all_reduce(total)
        if total.dtype != torch.float32:
            raise ValueError("expert reduction must preserve FP32 subtotals")
        shared = expert_output(hidden_states, self.shared_experts.gate_up_proj.weight,
                               self.shared_experts.down_proj.weight, limit=c.swiglu_limit)
        return (total + shared.float()).to(hidden_states.dtype)

    moe.forward = MethodType(forward, moe)
