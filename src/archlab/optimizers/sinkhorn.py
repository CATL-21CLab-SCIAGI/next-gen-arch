"""DeepSeek-V4.1 Algorithm 1, with row-sharded Sinkhorn scaling vectors.

Source: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf
Sections 2.5 and 3.1.3. This is not mHC's Sinkhorn-Knopp kernel. Official
training repositories expose no reusable implementation of this optimizer;
the public Viby implementation uses MLX rather than PyTorch/NCCL.
"""

import math

import torch
import torch.distributed as dist

from archlab.optimizers.sharded_adafactor import ShardedAdafactor, local_tensor, stochastic_bfloat16


def sinkhorn_direction(update, *, global_rows=None, group=None, steps=11, tau=1e-3, eps=1e-20):
    """Return Algorithm 1's RMS-normalized direction without iterative matrix writes.

    Column statistics and the masking threshold cover the complete row-sharded
    matrix. A zero update remains exactly zero, including empty feature columns.
    """
    if update.ndim != 2 or steps < 1 or steps % 2 != 1:
        raise ValueError("Sinkhorn requires a matrix and a positive odd iteration count")
    if not 0 <= tau or not eps > 0 or (global_rows is not None and global_rows < 1):
        raise ValueError("invalid Sinkhorn threshold, epsilon or row count")
    squared = update.float().square()
    rho = squared.sum(-1).sqrt()
    total = rho.sum()
    if group is not None:
        dist.all_reduce(total, group=group)
    rows = update.shape[0] if global_rows is None else global_rows
    row = (rho > tau * total / rows).to(torch.float32)
    column = torch.ones(update.shape[1], dtype=torch.float32, device=update.device)
    for iteration in range(steps):
        if iteration % 2 == 0:
            norm = (squared @ column.square()).sqrt() * row
            row = torch.where(norm > 0, row / (norm + eps), 0.0)
        else:
            partial = squared.T @ row.square()
            if group is not None:
                dist.all_reduce(partial, group=group)
            norm = partial.sqrt() * column
            column = torch.where(norm > 0, column / (norm + eps), 0.0)
    return update.float() * row[:, None] * column[None, :] * math.sqrt(update.shape[1])


class EngramSinkhornAdafactor(ShardedAdafactor):
    """Paper Sinkhorn for Engram tables; retain the control optimizer elsewhere.

    The experiment changes only the requested table group: beta=.95, gamma=.18,
    K=11, tau=1e-3, epsilon=1e-20, 5x table LR, and no table weight decay.
    FP32 momentum is the sole full-sized persistent table optimizer state.
    BF16 parameter storage uses the same unbiased stochastic rounding as the
    control optimizer; serialized RNG state makes continuation reproducible.
    All parameters remain in the original group/order for checkpoint support.
    """

    def __init__(self, named_parameters, lr=1e-4, table_lr_scale=1.0, **kwargs):
        named = list(named_parameters)
        super().__init__([p for _, p in named], lr=lr, **kwargs)
        if not math.isfinite(table_lr_scale) or not 0 < table_lr_scale <= 1:
            raise ValueError("invalid Sinkhorn table learning-rate scale")
        self.param_groups[0]["table_lr_scale"] = table_lr_scale
        self.table_ids = {id(p) for name, p in named if ".engram.embed.weight" in name and p.ndim == 2}
        if len(self.table_ids) != 2:
            raise ValueError("expected exactly two Engram embedding tables")

    @torch.no_grad()
    def step(self, closure=None):
        group = self.param_groups[0]
        parameters = group["params"]
        group["params"] = [p for p in parameters if id(p) not in self.table_ids]
        try:
            if group["params"]:
                result = super().step(closure)
            else:
                result = None
                if closure is not None:
                    with torch.enable_grad():
                        result = closure()
                self.last_metrics = {"updated_parameter_tensors": 0, "changed_local_elements": 0}
        finally:
            group["params"] = parameters
        changed = torch.zeros((), device=parameters[0].device, dtype=torch.int64)
        table_updates = 0
        for p in parameters:
            if id(p) not in self.table_ids or p.grad is None:
                continue
            value, owners = self._geometry(p)
            gradient = local_tensor(p.grad).float()
            state = self.state[p]
            if not state:
                state.update(step=0, momentum=torch.zeros_like(value, dtype=torch.float32))
            momentum = state["momentum"]
            momentum.mul_(0.95).add_(gradient, alpha=0.05)
            update = momentum.mul(0.95).add_(gradient, alpha=0.05)
            direction = sinkhorn_direction(update, global_rows=p.shape[0], group=owners)
            candidate = value.float() - 0.18 * 5 * group["lr"] * group["table_lr_scale"] * direction
            updated = (stochastic_bfloat16(candidate)
                       if value.dtype == torch.bfloat16 and group["stochastic_rounding"]
                       else candidate.to(value.dtype))
            changed += (updated != value).sum()
            value.copy_(updated)
            state["step"] += 1
            table_updates += 1
        self.last_metrics["changed_local_elements"] += int(changed)
        self.last_metrics["updated_parameter_tensors"] += table_updates
        return result


# Preserve the separately qualified resident RL optimizer/storage contract.
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
