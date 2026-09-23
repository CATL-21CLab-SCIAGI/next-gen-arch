"""V4.1 inference MoE equations with the qualified training precision policy."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def router_weights(x, weight, bias, *, top_k, route_scale):
    with torch.autocast(x.device.type, enabled=False):
        scores = F.softplus(F.linear(x.float(), weight.float())).sqrt()
        indices = (scores + bias.float()).topk(top_k, dim=-1).indices
        probabilities = scores.gather(1, indices)
        if top_k > 1:
            probabilities = probabilities / (probabilities.sum(-1, keepdim=True) + 1e-20)
        return probabilities * route_scale, indices


def expert_output(x, gate_up, down, *, limit, probability=None):
    middle = gate_up.shape[0] // 2
    if gate_up.shape[0] != middle * 2 or down.shape != (x.shape[-1], middle):
        raise ValueError("expert matrix geometry differs")
    with torch.autocast(x.device.type, enabled=False):
        gate = F.linear(x, gate_up[:middle]).float().clamp(max=limit)
        up = F.linear(x, gate_up[middle:]).float().clamp(min=-limit, max=limit)
        activated = F.silu(gate) * up
        if probability is not None:
            activated = activated * probability.float()
        return F.linear(activated.to(x.dtype), down)


def owned_experts(x, probabilities, indices, gate_up, down, *, first_expert, limit):
    """Return an FP32 subtotal, ordered by expert ID, before the EP reduction."""
    if gate_up.ndim != 3 or down.ndim != 3 or gate_up.shape[0] != down.shape[0]:
        raise ValueError("expected stacked local expert weights")
    total = torch.zeros_like(x, dtype=torch.float32)
    last = first_expert + gate_up.shape[0]
    for expert in indices.unique(sorted=True).tolist():
        if not first_expert <= expert < last:
            continue
        rows, slots = torch.where(indices == expert)
        local = expert - first_expert
        values = expert_output(x[rows], gate_up[local], down[local], limit=limit,
                               probability=probabilities[rows, slots, None])
        # A top-k list cannot contain the same expert twice for one token.
        total.index_add_(0, rows, values.float())
    return total
