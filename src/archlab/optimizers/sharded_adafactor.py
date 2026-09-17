"""GPU-resident Adafactor with FP32 factors and stochastic BF16 storage.

Implements the paper's epsilon-in-statistics formulation, relative parameter
scaling, time-dependent second moment, and RMS update clipping. Row-sharded
matrices reduce column statistics over their owner mesh; expert banks factor
each expert matrix separately, reducing across expert-FSDP row shards. There are no full-size master weights
or momentum tensors. Dense temporary tensors are bounded by ``chunk_elements``.
"""
from __future__ import annotations

import math
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard


def stochastic_bfloat16(value, *, generator=None):
    if value.dtype != torch.float32:
        raise TypeError("stochastic rounding takes FP32 values")
    noise = torch.randint(0, 65536, value.shape, dtype=torch.int32,
                          device=value.device, generator=generator)
    bits = value.contiguous().view(torch.int32)
    return ((bits + noise) >> 16).to(torch.int16).view(torch.bfloat16)


def local_tensor(value):
    return value.to_local() if isinstance(value, DTensor) else value


class ShardedAdafactor(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, *, eps=1e-30, parameter_floor=1e-3,
                 beta2_decay=-0.8, clip_threshold=1., chunk_elements=16*1024*1024,
                 stochastic_rounding=True):
        if not 0 < lr <= 1 or eps <= 0 or not -1 <= beta2_decay < 0:
            raise ValueError("invalid Adafactor hyperparameters")
        super().__init__(params, dict(lr=lr, eps=eps, parameter_floor=parameter_floor,
            beta2_decay=beta2_decay, clip_threshold=clip_threshold,
            chunk_elements=chunk_elements, stochastic_rounding=stochastic_rounding))
        self.last_metrics = {}

    @staticmethod
    def _geometry(p):
        value = local_tensor(p)
        group = None
        if value.ndim not in (1, 2, 3):
            raise ValueError(f"unsupported parameter shape: {value.shape}")
        if isinstance(p, DTensor):
            allowed = {0, 1} if p.ndim == 3 else {0}
            if any(not isinstance(s, Shard) or s.dim not in allowed for s in p.placements):
                raise ValueError(f"Adafactor requires unique batch/row shards: {p.placements}")
            row_dim = 1 if p.ndim == 3 else 0
            row_axes = [axis for axis, placement in enumerate(p.placements) if placement.dim == row_dim]
            if len(row_axes) > 1:
                raise ValueError("multiple mesh axes sharding a matrix row are not supported")
            if row_axes:
                group = p.device_mesh.get_group(row_axes[0])
        if p.ndim > 1 and value.shape[-1] != p.shape[-1]:
            raise ValueError("a factorized row must have its complete columns")
        return value, group

    @staticmethod
    def _sum(value, group):
        if group is not None:
            dist.all_reduce(value, group=group)
        return value

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        changed = torch.zeros((), device=self.param_groups[0]["params"][0].device, dtype=torch.int64)
        updated = 0
        for options in self.param_groups:
            for p in options["params"]:
                if p.grad is None:
                    continue
                value, group = self._geometry(p)
                grad = local_tensor(p.grad)
                if grad.is_sparse or grad.shape != value.shape:
                    raise ValueError("expected dense gradients with the local parameter shape")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    if p.ndim > 1:
                        state["row"] = torch.zeros((*value.shape[:-1], 1), device=value.device, dtype=torch.float32)
                        col_shape = (*value.shape[:-2], 1, value.shape[-1])
                        state["col"] = torch.zeros(col_shape, device=value.device, dtype=torch.float32)
                    else:
                        state["variance"] = torch.zeros_like(value, dtype=torch.float32)
                state["step"] += 1
                mix = state["step"] ** options["beta2_decay"]
                rho = min(options["lr"], state["step"] ** -0.5)
                eps = options["eps"]
                if p.ndim == 1:
                    g = grad.float()
                    state["variance"].lerp_(g.square().add_(eps), mix)
                    update = g * state["variance"].rsqrt()
                    squares = torch.stack((value.float().square().sum(), update.square().sum(),
                                           g.new_tensor(value.numel())))
                    self._sum(squares, group)
                    alpha = rho * max(options["parameter_floor"], math.sqrt(float(squares[0]/squares[2])))
                    alpha /= max(1., math.sqrt(float(squares[1]/squares[2])) / options["clip_threshold"])
                    candidate = value.float().add_(update, alpha=-alpha)
                    result = stochastic_bfloat16(candidate) if value.dtype == torch.bfloat16 and options["stochastic_rounding"] else candidate.to(value.dtype)
                    changed += (result != value).sum()
                    value.copy_(result)
                else:
                    columns, rows = value.shape[-1], value.shape[-2]
                    batch = 1 if p.ndim == 2 else value.shape[0]
                    v = value.view(batch, rows, columns)
                    g = grad.view_as(v)
                    row = state["row"].view(batch, rows, 1)
                    col = state["col"].view(batch, 1, columns)
                    chunk_rows = max(1, options["chunk_elements"] // columns)
                    col_sum = torch.zeros_like(col)
                    param_square = torch.zeros(batch, 1, 1, device=value.device, dtype=torch.float32)
                    for b in range(batch):
                        for first in range(0, rows, chunk_rows):
                            last = min(rows, first + chunk_rows)
                            squared = g[b:b+1, first:last].float().square().add_(eps)
                            row[b:b+1, first:last].lerp_(squared.mean(-1, keepdim=True), mix)
                            col_sum[b:b+1] += squared.sum(-2, keepdim=True)
                            param_square[b] += v[b, first:last].float().square().sum()
                    global_rows = p.shape[-2]
                    self._sum(col_sum, group)
                    self._sum(param_square, group)
                    col.lerp_(col_sum / global_rows, mix)
                    row_mean = row.sum(-2, keepdim=True)
                    self._sum(row_mean, group)
                    row_mean.div_(global_rows).clamp_min_(eps)
                    scale = (param_square / (global_rows * columns)).sqrt_().clamp_min_(options["parameter_floor"]).mul_(rho)
                    update_square = torch.zeros_like(param_square)
                    for b in range(batch):
                        for first in range(0, rows, chunk_rows):
                            last = min(rows, first + chunk_rows)
                            inverse = (row[b:b+1, first:last] / row_mean[b:b+1] * col[b:b+1]).clamp_min_(eps).rsqrt_()
                            inverse.mul_(g[b:b+1, first:last])
                            update_square[b] += inverse.square().sum()
                    self._sum(update_square, group)
                    scale.div_((update_square / (global_rows * columns)).sqrt_().div_(options["clip_threshold"]).clamp_min_(1))
                    for b in range(batch):
                        for first in range(0, rows, chunk_rows):
                            last = min(rows, first + chunk_rows)
                            update = (row[b:b+1, first:last] / row_mean[b:b+1] * col[b:b+1]).clamp_min_(eps).rsqrt_()
                            update.mul_(g[b:b+1, first:last]).mul_(scale[b:b+1])
                            old = v[b:b+1, first:last]
                            candidate = old.float() - update
                            result = stochastic_bfloat16(candidate) if old.dtype == torch.bfloat16 and options["stochastic_rounding"] else candidate.to(old.dtype)
                            changed += (result != old).sum()
                            old.copy_(result)
                updated += 1
        self.last_metrics = {"updated_parameter_tensors": updated, "changed_local_elements": int(changed)}
        return loss
