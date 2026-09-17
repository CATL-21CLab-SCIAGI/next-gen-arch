"""Report-configured Muon for complete, already-reduced adapter matrices.

This deliberately excludes the frozen speedrun optimizer's NorMuon, cautious
decay and Polar Express changes. The execution adapter owns DP gradient
reduction; no local FSDP shard may be passed as though it were a complete head.
Parameters and momentum stay FP32; projection compute may use BF16 autocast.
"""

from __future__ import annotations

import math

import torch


def orthogonalized_direction(gradient):
    """Eight fast Newton–Schulz steps and two stabilization steps per matrix."""
    if gradient.ndim < 2 or gradient.dtype != torch.float32:
        raise ValueError("complete FP32 matrices are required")
    transpose = gradient.shape[-2] > gradient.shape[-1]
    x = gradient.mT if transpose else gradient
    norm = x.norm(dim=(-2, -1), keepdim=True)
    # Guard exact zero without adding an epsilon that changes nonzero updates.
    x = x / torch.where(norm > 0, norm, torch.ones_like(norm))
    for a, b, c in [(3.4445, -4.775, 2.0315)] * 8 + [(2., -1.5, .5)] * 2:
        xx = x @ x.mT
        x = a * x + (b * xx + c * (xx @ xx)) @ x
    return x.mT if transpose else x


class HeadwiseMuon(torch.optim.Optimizer):
    """Group `head_dim` selects consecutive row blocks; None is one full matrix."""

    def __init__(self, params, lr=1e-5, momentum=.95, weight_decay=.1, update_rms=.18):
        values = (lr, momentum, weight_decay, update_rms)
        if any(not math.isfinite(x) for x in values) or lr < 0 or not 0 <= momentum < 1 or weight_decay < 0 or update_rms <= 0:
            raise ValueError("invalid Muon hyperparameters")
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                     update_rms=update_rms, head_dim=None))
        seen = set()
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) in seen or p.ndim != 2 or p.dtype != torch.float32 or not p.requires_grad:
                    raise ValueError("unique trainable complete FP32 matrix parameters are required")
                if hasattr(p, "to_local"):
                    raise ValueError("orthogonalizing distributed shards is not supported")
                seen.add(id(p))
                head_dim = group["head_dim"]
                if head_dim is not None and (type(head_dim) is not int or head_dim < 1 or p.shape[0] % head_dim):
                    raise ValueError("head_dim must divide matrix rows")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse or p.grad.dtype != torch.float32:
                    raise ValueError("dense globally reduced FP32 gradients are required")
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                m = state["momentum_buffer"]
                m.mul_(group["momentum"]).add_(p.grad)
                direction = p.grad + group["momentum"] * m
                head_dim = group["head_dim"]
                matrices = direction if head_dim is None else direction.unflatten(0, (-1, head_dim))
                update = orthogonalized_direction(matrices)
                scale = group["update_rms"] * math.sqrt(max(matrices.shape[-2:]))
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape_as(p), alpha=-group["lr"] * scale)
        return loss
