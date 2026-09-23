"""World-aggregated, loss-free expert balancing with checkpointed FP32 bias."""

from collections import deque

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate


def auxiliary_scale(local_targets, global_targets):
    if global_targets <= 0 or not 0 <= local_targets <= global_targets:
        raise ValueError("invalid local/global target counts for router auxiliary loss")
    return local_targets / global_targets


def bias_correction(load, rate, *, proportional=False):
    mean = load.float().mean()
    if proportional:
        change = ((mean - load.float()) / mean.clamp_min(1e-12)).clamp(-5.0, 1.0) * rate
        return change - change.mean()
    return (mean - load).sign() * rate


@torch.no_grad()
def balance_routers(gates, rate, *, proportional=False):
    if not gates or rate <= 0:
        raise ValueError("active learned routers and positive balance rate required")
    loads = []
    for gate in gates:
        if gate._cumulative_expert_load is None:
            raise ValueError("router load was not collected")
        loads.append(gate._cumulative_expert_load.detach().float().clone())
    matrix = torch.stack(loads)
    dist.all_reduce(matrix)
    cvs = []
    dead = []
    unused_windows = []
    window_sizes = []
    for gate, load in zip(gates, matrix, strict=True):
        bias = gate.e_score_correction_bias
        if bias.dtype != torch.float32:
            raise ValueError("routing bias must retain FP32 precision")
        change = bias_correction(load, rate, proportional=proportional)
        if isinstance(bias, DTensor):
            change = DTensor.from_local(
                change,
                device_mesh=bias.device_mesh,
                placements=[Replicate()] * bias.device_mesh.ndim,
                run_check=False,
            ).redistribute(placements=bias.placements)
        bias.add_(change)
        gate._cumulative_expert_load = None
        gate.e_score_correction_bias_master = None
        cvs.append(float(load.std() / load.mean().clamp_min(1)))
        dead.append(float((load == 0).float().mean()))
        if not hasattr(gate, "_archlab_recent_expert_usage"):
            gate._archlab_recent_expert_usage = deque(maxlen=20)
        gate._archlab_recent_expert_usage.append((load > 0).detach().clone())
        unused_windows.append(
            float((~torch.stack(tuple(gate._archlab_recent_expert_usage)).any(0)).float().mean())
        )
        window_sizes.append(len(gate._archlab_recent_expert_usage))
    return {
        "router_load_cv_mean": sum(cvs) / len(cvs),
        "router_load_cv_max": max(cvs),
        "router_dead_fraction_mean": sum(dead) / len(dead),
        "router_dead_fraction_max": max(dead),
        "router_cv_per_layer": cvs,
        "router_unused_fraction_per_layer": dead,
        "router_unused_fraction_window": sum(unused_windows) / len(unused_windows),
        "router_worst_unused_fraction_window": max(unused_windows),
        "router_usage_window_updates": min(window_sizes),
    }
