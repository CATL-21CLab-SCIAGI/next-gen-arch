"""Disposable first-step FP32-master Adam oracle for RL admission.

This is an admission calculation, never a training optimizer. It preserves the
native gradient clipping dtype and FP32 subtraction rounding that an ideal
``-lr * sign(gradient)`` comparison would omit.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def first_adam_updates(parameters, gradients, *, max_grad_norm=1., learning_rate=1e-5,
                       betas=(.9, .95), epsilon=1e-8):
    """Return named first-step master deltas and clipping evidence.

    Parameters and gradients must be snapshots on the same device, with their
    original dtypes. No input is mutated. The contract is a fresh, zero-decay,
    bias-corrected Adam with FP32 masters and moments, as used by matched RL.
    Missing gradients retain their optimizer no-op semantics.
    """
    if parameters.keys() != gradients.keys():
        raise ValueError("Adam admission requires matching parameter and gradient names")
    if (not all(math.isfinite(value) and value > 0
                for value in (max_grad_norm, learning_rate, epsilon))
            or len(betas) != 2 or any(not 0 <= value < 1 for value in betas)):
        raise ValueError("invalid first-step Adam admission settings")
    active = []
    for name, parameter in parameters.items():
        gradient = gradients[name]
        if gradient is None:
            continue
        if (parameter.dtype not in (torch.bfloat16, torch.float32)
                or gradient.shape != parameter.shape or gradient.dtype != parameter.dtype
                or gradient.device != parameter.device):
            raise ValueError("Adam admission must retain native BF16/FP32 gradient geometry")
        if not bool(torch.isfinite(gradient).all() and torch.isfinite(parameter).all()):
            raise FloatingPointError("nonfinite parameter or gradient in Adam admission")
        active.append(gradient)
    if not active:
        return dict.fromkeys(parameters), dict(gradient_norm=0., clip_coefficient=1., no_step=True)
    if any(value.device != active[0].device for value in active):
        raise ValueError("Adam admission requires one device")
    # Same mixed-dtype norm geometry as torch.nn.utils.clip_grad_norm_. Its
    # BF16 per-tensor norms are rounded before the final promoted norm.
    total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(value) for value in active]))
    coefficient = (max_grad_norm / (total_norm + 1e-6)).clamp(max=1.)
    updates = {}
    beta1, beta2 = betas
    for name, parameter in parameters.items():
        gradient = gradients[name]
        if gradient is None:
            updates[name] = None
            continue
        # In-place native clipping rounds each BF16 gradient before the Fused
        # Adam kernel consumes it. A BF16 tensor multiplied by this scalar
        # retains its dtype, as with the public PyTorch clipping operation.
        clipped = (gradient * coefficient).to(gradient.dtype).float()
        moment = clipped * (1 - beta1)
        variance = clipped.square() * (1 - beta2)
        step = learning_rate * (moment / (1 - beta1)) / ((variance / (1 - beta2)).sqrt() + epsilon)
        master = parameter.detach().float()
        updates[name] = (master - step) - master
    return updates, dict(gradient_norm=float(total_norm), clip_coefficient=float(coefficient), no_step=False,
                         learning_rate=learning_rate, betas=betas, epsilon=epsilon,
                         gradient_clipping="native dtype", master_and_moment_dtype="float32")
