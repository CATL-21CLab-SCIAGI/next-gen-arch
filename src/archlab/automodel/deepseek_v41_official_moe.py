"""Reference sum precision around the official V4.1 grouped expert kernels.

The released model adds routed BF16 expert outputs and its shared expert in
FP32, then rounds once. AutoModel's standard torch dispatcher already computes
the routed subtotal in FP32, but casts it before the shared addition. This
project-owned orchestration retains that subtotal through the shared addition.
Gate/up projections use native separate BF16 linears and eager FP32 activation;
the grouped down kernel and official autograd gather are retained. No upstream
source, parameter, checkpoint key, or runtime package is changed.
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f
from torch.distributed.tensor import DTensor
from torch.nn import functional as F


class _NonMutatingVarlenGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local, group, lengths, max_length):
        from nemo_automodel.components.moe.experts import _AllGatherConcatVarlenFn

        return _AllGatherConcatVarlenFn.forward(ctx, local, group, lengths, max_length)

    @staticmethod
    def backward(ctx, grad_output):
        # Incoming gradients may alias another branch's gradient. A distributed
        # in-place sum must operate on storage owned by this backward only.
        reduced = grad_output.contiguous().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=ctx.group)
        start = sum(ctx.gathered_lens[:ctx.rank])
        return reduced.narrow(0, start, ctx.gathered_lens[ctx.rank]).clone(), None, None, None


def _native_up_grouped_down(self, x, token_mask, weights, indices, up, down,
                            local_experts, first_expert):
    """Match native BF16 gate/up linears; retain the official grouped down kernel.

    The grouped gate/up GEMM changes values at BF16 rounding boundaries for
    this checkpoint's decoded FP4 weights. Separate F.linear calls reproduce
    its native projection values. Expert parameters and EP/FSDP ownership stay
    in the official module; only its computation dispatch is adapted.
    """
    from nemo_automodel.components.moe.experts import _permute_tokens_for_grouped_mm

    ids, slots, probs, counts, offs = _permute_tokens_for_grouped_mm(
        indices, weights, token_mask, local_experts, first_expert, return_slot_ids=True)
    if ids.numel() == 0:
        # The upstream empty-owner branch preserves the required autograd path.
        return self._forward_grouped_mm(x.to(up.dtype), token_mask, weights, indices, up, down,
                                        None, None, local_experts, first_expert)
    middle = up.shape[-1] // 2
    activations = []
    first = 0
    for expert, count in enumerate(counts.tolist()):
        last = first + count
        if count:
            selected = x[ids[first:last]].to(up.dtype)
            gate = F.linear(selected, up[expert, :, :middle].T.contiguous()).float()
            value = F.linear(selected, up[expert, :, middle:].T.contiguous()).float()
            gate = gate.clamp(max=self.config.swiglu_limit)
            value = value.clamp(min=-self.config.swiglu_limit, max=self.config.swiglu_limit)
            activated = F.silu(gate) * value
            activated = activated * probs[first:last, None]
            activations.append(activated.to(up.dtype))
        first = last
    # BF16 grouped down matched native F.linear bit-for-bit on identical
    # activations in the real-layer stage diagnostic.
    output = torch._grouped_mm(torch.cat(activations), down, offs)
    # Native MoE sums each token's experts in ascending expert-ID order. Avoid
    # unordered atomic collisions at BF16 halfway cases before the final cast.
    # Each selected token occurs at most once in an individual ordered slot.
    positions = torch.full(indices.shape, -1, dtype=torch.int64, device=x.device)
    positions.view(-1)[slots] = torch.arange(ids.numel(), device=x.device)
    ordered_positions = positions.gather(1, indices.argsort(dim=-1))
    result = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(indices.shape[1]):
        token_ids = torch.where(ordered_positions[:, slot] >= 0)[0]
        result = result.index_add(0, token_ids, output[ordered_positions[token_ids, slot]].float())
    return result


def _plain_gather(local, group, lengths, max_length):
    if local.shape[0] < max_length:
        padding = local.new_zeros((max_length - local.shape[0], *local.shape[1:]))
        local = torch.cat((local, padding))
    parts = [torch.empty_like(local) for _ in lengths]
    dist.all_gather(parts, local, group=group)
    return torch.cat([part[:length] for part, length in zip(parts, lengths, strict=True)])


def _fp32_grouped_experts_forward(self, x, token_mask, weights, indices):
    """Keep official EP ownership with native projection and FP32 sum precision."""
    if isinstance(x, DTensor) or x.ndim != 2:
        raise ValueError("FP32 MoE combination requires a local [tokens, hidden] tensor")
    projection = self.gate_and_up_projs
    if isinstance(projection, DTensor):
        mesh = projection.device_mesh
        if mesh.ndim != 1:
            raise ValueError("expert FSDP must unshard before the remaining one-dimensional EP placement")
        ep_size, ep_rank, group = mesh.size(), mesh.get_local_rank(), mesh.get_group()
        up = projection.to_local()
        down = self.down_projs.to_local()
    else:
        ep_size, ep_rank, group = 1, 0, None
        up, down = projection, self.down_projs
    if self.n_routed_experts % ep_size:
        raise ValueError("routed expert count must divide the EP mesh")
    local_experts = self.n_routed_experts // ep_size
    if up.shape[0] != local_experts or down.shape[0] != local_experts:
        raise ValueError("expert weights do not have their complete local EP shapes")
    up, down = up.to(x.dtype), down.to(x.dtype)
    # BF16 values are represented exactly in this FP32 carrier. Per-expert
    # linears still receive BF16 inputs, but their branch gradients and the
    # EP all-reduce now accumulate in FP32 before one cast to the original x.
    # This avoids repeated BF16 rounding while many expert branches converge.
    x = x.float()
    local_tokens = x.shape[0]
    if ep_size > 1:
        count = torch.tensor([local_tokens], device=x.device, dtype=torch.int64)
        counts = [torch.empty_like(count) for _ in range(ep_size)]
        dist.all_gather(counts, count, group=group)
        lengths = [int(value.item()) for value in counts]
        max_length = max(lengths)
        gather = getattr(self, "_archlab_ep_gather", _NonMutatingVarlenGather)
        x = gather.apply(x, group, lengths, max_length)
        weights = gather.apply(weights.float(), group, lengths, max_length)
        indices = _plain_gather(indices, group, lengths, max_length)
        token_mask = _plain_gather(token_mask, group, lengths, max_length)
    if x.shape[0] == 0:
        raise ValueError("MoE requires at least one physical token in its EP group")
    # Official permutation/grouped down and native gate/up precision retain
    # probability placement, empty-owner gradients, and FP32 scatter sum.
    routed = _native_up_grouped_down(self, x, token_mask, weights, indices, up, down,
                                   local_experts, ep_rank * local_experts)
    if routed.dtype != torch.float32:
        raise TypeError("the official grouped expert subtotal must remain FP32")
    if ep_size > 1:
        # Preserve the upstream empty-owner link and collective backward order.
        routed = routed + x.sum(dtype=torch.float32) * 0.0
        routed = dist_nn_f.all_reduce(routed, op=dist.ReduceOp.SUM, group=group)
        routed = routed.narrow(0, sum(lengths[:ep_rank]), local_tokens).contiguous()
    return routed


def _cast_complete_moe(module, args, kwargs, output):
    inputs = kwargs.get("x", args[0] if args else None)
    if not isinstance(inputs, torch.Tensor) or not isinstance(output, torch.Tensor):
        raise TypeError("expected tensor input/output at the complete MoE boundary")
    if output.dtype != torch.float32:
        raise TypeError("shared expert must be added while the routed subtotal is FP32")
    return output.to(inputs.dtype)


def install_official_fp32_moe(model):
    """Install after official torch-dispatcher construction/loading/FSDP wrapping.

    Only decoder MoE numerics are adapted. The official MoE.forward still owns
    routing, the shared expert and their addition; its output hook performs the
    single final cast. All registered parameter objects and keys are retained.
    """
    from nemo_automodel.components.moe.experts import GroupedExperts
    from nemo_automodel.components.moe.layers import MoE

    if getattr(model, "_archlab_v41_fp32_moe_installed", False):
        raise ValueError("official FP32 MoE combination is already installed")
    selected = [(name, module) for name, module in model.named_modules() if isinstance(module, MoE)]
    if not selected:
        raise ValueError("no official MoE modules found")
    for name, module in selected:
        expert = module.experts
        if (module.backend.dispatcher != "torch" or module.backend.experts != "torch_mm"
                or not isinstance(expert, GroupedExperts) or not expert.use_torch_mm or expert.use_mxfp8):
            raise ValueError(f"{name}: FP32 sum adaptation requires official torch dispatcher and torch_mm experts")
        config = expert.config
        if (config.apply_router_weight_after_down or config.expert_bias
                or config.expert_activation != "swiglu" or config.swiglu_limit <= 0
                or config.n_shared_experts != 1 or module.shared_expert_gate is not None
                or module.fc1_latent_proj is not None or module.fc2_latent_proj is not None):
            raise ValueError(f"{name}: MoE geometry differs from native V4.1 pre-down weighting/shared expert")
        if "forward" in expert.__dict__:
            raise ValueError(f"{name}: expert forward has already been adapted")
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError("freeze the official base before adapting its MoE precision")
    original = dict(model.named_parameters())
    for _, module in selected:
        module.experts.forward = MethodType(_fp32_grouped_experts_forward, module.experts)
        module.register_forward_hook(_cast_complete_moe, with_kwargs=True)
    after = dict(model.named_parameters())
    if original.keys() != after.keys() or any(after[name] is not parameter for name, parameter in original.items()):
        raise RuntimeError("MoE precision adaptation changed parameter objects or checkpoint keys")
    model._archlab_v41_fp32_moe_installed = True
    return {"implementation": "project-native-up-grouped-down-ordered-fp32-moe-v3",
            "modules": [name for name, _ in selected], "dispatcher": "torch", "experts": "torch_mm",
            "gate_up_compute": "native-separate-contiguous-bfloat16-linear",
            "activation": "eager-float32-clamped-probability-weighted-swiglu",
            "down_compute": "torch-grouped-mm",
            "input_gradient_accumulation": "float32-local-and-expert-parallel",
            "expert_accumulation_order": "ascending-expert-id",
            "gather_backward": "private-clone-before-inplace-allreduce",
            "routed_subtotal_dtype": "float32", "shared_add_dtype": "float32",
            "final_output_dtype": "input_dtype", "router_weight_placement": "before_down_projection",
            "original_parameters_preserved": True}
