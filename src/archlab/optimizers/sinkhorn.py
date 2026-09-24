"""DeepSeek V4.1 Algorithm 1, with row-sharded GPU-resident momentum.

Only column statistics and the mean row norm are reduced across row owners.
BF16 momentum storage is an explicit local adaptation of the report.
"""

import math

import torch
import torch.distributed as dist


@torch.no_grad()
def sinkhorn_step(master, grad, momentum, *, lr, group=None, beta=0.95,
                  iterations=11, threshold=1e-3, eps=1e-20, correction=0.18,
                  chunk_rows=4096, momentum_scale=None):
    if master.ndim != 2 or iterations < 1 or iterations % 2 != 1:
        raise ValueError("Sinkhorn requires a matrix and an odd positive iteration count")
    rows, columns = master.shape
    distributed = group is not None and dist.get_world_size(group) > 1
    old_scale = momentum_scale.clone() if momentum_scale is not None else 1.
    if momentum_scale is not None:
        maximum = torch.zeros((), device=master.device)
        for first in range(0, rows, chunk_rows):
            sl = slice(first, first + chunk_rows)
            updated = momentum[sl].float().mul(old_scale * beta).add_(grad[sl].float(), alpha=1-beta)
            maximum = torch.maximum(maximum, updated.abs().amax())
        # Use one scale across row shards so stored rounding is shard invariant.
        if distributed:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        momentum_scale.copy_(torch.exp2(torch.ceil(torch.log2(maximum.clamp_min(2.**-110)))))
    new_scale = momentum_scale if momentum_scale is not None else 1.
    # Keep only scalar row/column factors and one bounded FP32 row chunk.
    row_norm = torch.empty(rows, device=master.device, dtype=torch.float32)
    for first in range(0, rows, chunk_rows):
        sl = slice(first, first + chunk_rows)
        updated = momentum[sl].float().mul(old_scale * beta).add_(grad[sl].float(), alpha=1-beta)
        momentum[sl].copy_(updated / new_scale)
        # Use the stored value for reproducible continuation from checkpoints.
        direction = momentum[sl].float().mul(new_scale * beta).add_(grad[sl].float(), alpha=1-beta)
        row_norm[sl] = direction.norm(dim=1)
    totals = torch.stack((row_norm.sum(), row_norm.new_tensor(rows)))
    if distributed:
        dist.all_reduce(totals, group=group)
    row_scale = (row_norm > threshold * totals[0] / totals[1].clamp_min(1)).float()
    col_scale = torch.ones(columns, device=master.device, dtype=torch.float32)
    for iteration in range(iterations):
        if iteration % 2 == 0:
            for first in range(0, rows, chunk_rows):
                sl = slice(first, first + chunk_rows)
                direction = momentum[sl].float().mul(new_scale * beta).add_(grad[sl].float(), alpha=1-beta)
                norm = (direction * col_scale * row_scale[sl, None]).norm(dim=1)
                row_scale[sl] /= norm + eps
        else:
            squared = torch.zeros_like(col_scale)
            for first in range(0, rows, chunk_rows):
                sl = slice(first, first + chunk_rows)
                direction = momentum[sl].float().mul(new_scale * beta).add_(grad[sl].float(), alpha=1-beta)
                squared += (direction * row_scale[sl, None] * col_scale).square().sum(dim=0)
            if distributed:
                dist.all_reduce(squared, group=group)
            norm = squared.sqrt()
            col_scale = torch.where(norm > 0, col_scale / (norm + eps), torch.ones_like(col_scale))
    for first in range(0, rows, chunk_rows):
        sl = slice(first, first + chunk_rows)
        direction = momentum[sl].float().mul(new_scale * beta).add_(grad[sl].float(), alpha=1-beta)
        master[sl].add_(direction * row_scale[sl, None] * col_scale,
                        alpha=-lr * correction * math.sqrt(columns))
