"""Project-owned EP adapters around the unmodified native expert equations.

Attention is replicated, not tensor-parallel. Different ranks may supply
different tokens. All-to-all has an explicit input-gradient inverse; the frozen
Engram tables use row-sharded lookups. This module does not create a mesh or
claim a qualified 32-rank training configuration.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


class _Exchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, send_sizes, recv_sizes, group):
        ctx.send_sizes, ctx.recv_sizes, ctx.group = send_sizes, recv_sizes, group
        result = values.new_empty((sum(recv_sizes), *values.shape[1:]))
        dist.all_to_all_single(result, values.contiguous(), recv_sizes, send_sizes, group=group)
        return result

    @staticmethod
    def backward(ctx, gradient):
        result = gradient.new_empty((sum(ctx.send_sizes), *gradient.shape[1:]))
        dist.all_to_all_single(result, gradient.contiguous(), ctx.send_sizes, ctx.recv_sizes, group=ctx.group)
        return result, None, None, None


def exchange_layout(destinations, group):
    size = dist.get_world_size(group)
    counts = torch.bincount(destinations, minlength=size)
    incoming = torch.empty_like(counts)
    dist.all_to_all_single(incoming, counts, group=group)
    return destinations.argsort(stable=True), counts.tolist(), incoming.tolist()


def exchange(values, send_sizes, recv_sizes, group):
    return _Exchange.apply(values, send_sizes, recv_sizes, group)


class _ExchangeWithWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weights, send, recv, group):
        ctx.send, ctx.recv, ctx.group = send, recv, group
        return exchange(values, send, recv, group), exchange(weights, send, recv, group)

    @staticmethod
    def backward(ctx, dx, dw):
        # One autograd node fixes the ordering of the two reverse collectives
        # even when ranks execute different sets of routed experts.
        return (exchange(dx, ctx.recv, ctx.send, ctx.group),
                exchange(dw, ctx.recv, ctx.send, ctx.group), None, None, None)


def expert_parallel_classes(reference, group):
    """Return constructors with native checkpoint names and no TP assumptions."""
    size, rank = dist.get_world_size(group), dist.get_rank(group)

    class ExpertParallelMoE(nn.Module):
        def __init__(self, layer_id, args):
            super().__init__()
            experts, active = args.get_moe_config(layer_id)
            if experts % size:
                raise ValueError("expert count must divide EP size")
            self.dim, self.n_activated_experts = args.dim, active
            self.local_experts = experts // size
            self.first = rank * self.local_experts
            self.gate = reference.Gate(layer_id, args)
            dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
            self.experts = nn.ModuleList([
                reference.Expert(args.dim, args.moe_inter_dim, dtype=dtype, swiglu_limit=args.swiglu_limit)
                if self.first <= i < self.first + self.local_experts else None
                for i in range(experts)
            ])
            if args.n_shared_experts != 1:
                raise ValueError("exactly one native shared expert is required")
            self.shared_experts = reference.Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)

        def forward(self, x, image_mask=None):
            shape = x.shape
            x = x.reshape(-1, self.dim)
            weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
            expert_ids = indices.flatten()
            order, send, recv = exchange_layout(expert_ids // self.local_experts, group)
            token_ids = torch.arange(x.shape[0], device=x.device).repeat_interleave(self.n_activated_experts)[order]
            received_ids = exchange(expert_ids[order], send, recv, group)
            received_x, received_weights = _ExchangeWithWeights.apply(
                x[token_ids], weights.flatten()[order, None], send, recv, group)
            # Connect empty receivers to BOTH collective graphs. Every rank
            # must participate in the same reverse collectives, even with zero
            # tokens assigned to its local experts.
            result = received_x.float() * 0 + received_weights * 0
            for i in range(self.first, self.first + self.local_experts):
                selected = torch.where(received_ids == i)[0]
                if selected.numel():
                    result = result.index_copy(0, selected,
                                              self.experts[i](received_x[selected], received_weights[selected]).float())
            returned = exchange(result, recv, send, group)
            # Match the native serial expert-ID addition order. Atomic
            # index_add would introduce forward nondeterminism and make exact
            # zero-adapter identity checks unreliable near BF16 boundaries.
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(order.numel(), device=order.device)
            expert_order = indices.argsort(dim=-1)
            original_slots = torch.arange(x.shape[0], device=x.device) * self.n_activated_experts
            routed = torch.zeros_like(x, dtype=torch.float32)
            for k in range(self.n_activated_experts):
                routed = routed + returned[inverse[original_slots + expert_order[:, k]]]
            return (routed + self.shared_experts(x)).to(x.dtype).reshape(shape)

    class RowShardedEngram(nn.Module):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__()
            if embedding_dim % 32:
                raise ValueError("native Engram requires group-32 scales")
            self.num_embeddings = num_embeddings
            self.rows = (num_embeddings + size - 1) // size
            self.embedding_dim = embedding_dim
            self.weight = nn.Parameter(torch.empty(self.rows, embedding_dim, dtype=torch.float8_e4m3fn),
                                       requires_grad=False)
            self.scale = nn.Parameter(torch.empty(self.rows, embedding_dim // 32, dtype=torch.float8_e8m0fnu),
                                      requires_grad=False)

        @torch.no_grad()
        def forward(self, indices):
            ids = indices.flatten()
            if bool(((ids < 0) | (ids >= self.num_embeddings)).any()):
                raise ValueError("Engram lookup outside the original table")
            order, send, recv = exchange_layout(ids // self.rows, group)
            received = exchange(ids[order], send, recv, group) - rank * self.rows
            if self.weight.dtype == torch.bfloat16 and self.scale is None:
                values = self.weight[received]
            else:
                values = self.weight[received].float()
                scales = self.scale[received].float().repeat_interleave(32, -1)
                values = (values * scales).bfloat16()
            returned = exchange(values, recv, send, group)
            output = torch.empty_like(returned)
            output[order] = returned
            return output.reshape(*indices.shape, self.embedding_dim)

    return ExpertParallelMoE, RowShardedEngram
