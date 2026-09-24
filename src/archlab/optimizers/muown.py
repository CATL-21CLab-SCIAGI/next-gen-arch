"""Muown's implicit row weight normalization, with checkpointable row state.

The Newton--Schulz coefficients and Jacobian follow kcc-lion/muown. The
additional direction-update multiplier is explicit, separate from the
reference optimizer's 0.2 * sqrt(max(rows, columns)) scaling.
"""

import math

import torch


def orthogonalize(matrix, steps=10, dtype=torch.bfloat16):
    x = matrix.to(dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a = x @ x.T
        b = torch.addmm(a, a, a, beta=-4.7750, alpha=2.0315)
        x = torch.addmm(x, b, x, beta=3.4445)
    return (x.T if transposed else x).float()


class Muown(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-6, momentum=0.95, betas=(0.95, 0.95),
                 eps=1e-8, ns_steps=10, update_scale=0.5, pg_collection=None,
                 momentum_dtype=None, orthogonalization_dtype=torch.bfloat16,
                 direction_observer=None):
        super().__init__(params, dict(lr=lr, momentum=momentum, betas=betas,
                                     eps=eps, ns_steps=ns_steps, update_scale=update_scale))
        self.pg_collection = pg_collection
        self.momentum_dtype = momentum_dtype
        self.orthogonalization_dtype = orthogonalization_dtype
        self.direction_observer = direction_observer

    def _parallel(self, p):
        axis = getattr(p, "partition_dim", -1)
        if not getattr(p, "tensor_model_parallel", False) or axis < 0 or self.pg_collection is None:
            return None, -1
        group = self.pg_collection.expt_tp if getattr(p, "expert_tp", False) else self.pg_collection.tp
        return group, axis

    def _row_sum(self, p, value):
        result = value.sum(1, keepdim=True)
        group, axis = self._parallel(p)
        if axis == 1 and torch.distributed.get_world_size(group) > 1:
            torch.distributed.all_reduce(result, group=group)
        return result

    def _init_group(self, group, skip_non_grad_params=True):
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue
            if not self.state[p]:
                if p.ndim != 2:
                    raise ValueError("Muown matrix groups must contain two-dimensional weights")
                norm = self._row_sum(p, p.float().square()).sqrt()
                self.state[p].update(step=0, g=norm.clone(), v_norm=norm.clamp_min(1e-12),
                                     momentum_buffer=torch.zeros_like(p, dtype=self.momentum_dtype or p.dtype),
                                     m_g=torch.zeros_like(norm), v_g=torch.zeros_like(norm))
                if self.momentum_dtype == torch.float16:
                    self.state[p]["momentum_scale"] = torch.ones((), device=p.device, dtype=torch.float32)

    def _orthogonalize(self, p, direction, steps):
        group, axis = self._parallel(p)
        size = torch.distributed.get_world_size(group) if group is not None else 1
        if size == 1:
            return orthogonalize(direction, steps, self.orthogonalization_dtype), direction.shape
        parts = [torch.empty_like(direction) for _ in range(size)]
        torch.distributed.all_gather(parts, direction, group=group)
        stride = getattr(p, "partition_stride", 1)
        if stride != 1:
            split = [part.chunk(stride, dim=axis) for part in parts]
            full = torch.cat([split[r][s] for s in range(stride) for r in range(size)], dim=axis)
        else:
            full = torch.cat(parts, dim=axis)
        result = orthogonalize(full, steps, self.orthogonalization_dtype)
        rank = torch.distributed.get_rank(group)
        local = torch.cat([part.chunk(size, dim=axis)[rank]
                           for part in result.chunk(stride, dim=axis)], dim=axis)
        return local, full.shape

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            self._init_group(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                s = self.state[p]
                s["step"] += 1
                g, vn = s["g"], s["v_norm"]
                grad = p.grad.float()
                u = p.float() / g.abs().clamp_min(1e-12) * g.sign().masked_fill(g == 0, 1)
                # Weight normalization is singular at zero; seed a direction
                # from the first gradient there, while keeping initial W=0.
                grad_norm = self._row_sum(p, grad.square()).sqrt().clamp_min(1e-12)
                u = torch.where(g == 0, grad / grad_norm, u)
                v = u * vn
                grad_g = self._row_sum(p, grad * u)
                grad_v = (g / vn) * (grad - u * grad_g)
                momentum = s["momentum_buffer"].float()
                if "momentum_scale" in s:
                    momentum.mul_(s["momentum_scale"])
                momentum.mul_(group["momentum"]).add_(grad_v)
                if "momentum_scale" in s:
                    # A power-of-two scale preserves FP16 mantissa precision
                    # without under/overflow for small/large gradient histories.
                    scale = torch.exp2(torch.ceil(torch.log2(momentum.abs().amax().clamp_min(2.**-110))))
                    s["momentum_buffer"].copy_(momentum / scale)
                    s["momentum_scale"].copy_(scale)
                else:
                    s["momentum_buffer"].copy_(momentum)
                direction = grad_v.add(momentum, alpha=group["momentum"])
                update, shape = self._orthogonalize(p, direction, group["ns_steps"])
                if self.direction_observer is not None:
                    self.direction_observer(update)
                v.add_(update, alpha=-group["lr"] * 0.2 * math.sqrt(max(shape)) * group["update_scale"])
                beta1, beta2 = group["betas"]
                s["m_g"].lerp_(grad_g, 1 - beta1)
                s["v_g"].lerp_(grad_g.square(), 1 - beta2)
                denom = (s["v_g"] / (1 - beta2 ** s["step"])).sqrt().add_(group["eps"])
                g.addcdiv_(s["m_g"], denom, value=-group["lr"] / (1 - beta1 ** s["step"]))
                vn.copy_(self._row_sum(p, v.square()).sqrt().clamp_min(1e-12))
                p.copy_(g * (v / vn))
        return loss
