# ruff: noqa: I001  # Preserve the checkpoint-qualified import order.
"""Trainable HC and vocabulary-head boundaries on the pinned official model."""

from __future__ import annotations

from types import MethodType
import torch
from torch.nn import functional as F

from archlab.architectures.deepseek_v41_math import hc_split_sinkhorn


class NativeTrainableHC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixes, scale, base, streams, iterations, eps, native):
        ctx.save_for_backward(mixes, scale, base)
        ctx.geometry = streams, iterations, eps
        return native(mixes, scale, base, streams, iterations, eps)

    @staticmethod
    def backward(ctx, dpre, dpost, dcomb):
        saved = ctx.saved_tensors
        with torch.enable_grad():
            leaves = tuple(value.detach().requires_grad_() for value in saved)
            outputs = hc_split_sinkhorn(*leaves, *ctx.geometry)
            gradients = torch.autograd.grad(outputs, leaves, (dpre, dpost, dcomb))
        return *gradients, None, None, None, None


def trainable_hc_forward(self, hidden_states):
    from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41Mix

    with torch.autocast(hidden_states.device.type, enabled=False):
        flat = hidden_states.flatten(2).float()
        mixes = F.linear(flat, self.fn) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.norm_eps
        )
        # Clones retain derivatives and outlive the owner's FSDP unshard storage.
        scale, base = self.scale.clone(), self.base.clone()
        values = NativeTrainableHC.apply(
            mixes, scale, base, self.streams, self.iterations, self.eps, self._archlab_native_hc
        )
    return DeepseekV41Mix(*values)


class TrainableHeadCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, labels, weight, chunk_size=128):
        if weight.dtype != torch.float32 or chunk_size < 1:
            raise ValueError("full training retains the native FP32 vocabulary head")
        if hidden.shape[:-1] != labels.shape or hidden.shape[-1] != weight.shape[1]:
            raise ValueError("hidden/label/head shapes do not match")
        x, targets = hidden.flatten(0, -2), labels.flatten()
        valid = torch.where(targets != -100)[0]
        loss = hidden.new_zeros((), dtype=torch.float32)
        for ids in valid.split(chunk_size):
            if ids.numel():
                loss += F.cross_entropy(
                    F.linear(x[ids].float(), weight), targets[ids], reduction="sum"
                )
        ctx.save_for_backward(hidden, labels, weight, valid)
        ctx.chunk_size = chunk_size
        return loss

    @staticmethod
    def backward(ctx, dl):
        hidden, labels, weight, valid = ctx.saved_tensors
        x, targets = hidden.flatten(0, -2), labels.flatten()
        dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[2] else None
        for ids in valid.split(ctx.chunk_size):
            if ids.numel():
                values = x[ids].float()
                probabilities = F.linear(values, weight).softmax(-1)
                probabilities[torch.arange(ids.numel(), device=ids.device), targets[ids]] -= 1
                probabilities.mul_(dl)
                if dx is not None:
                    dx[ids] = (probabilities @ weight).to(dx.dtype)
                if dw is not None:
                    dw.addmm_(probabilities.T, values)
        return None if dx is None else dx.reshape_as(hidden), None, dw, None


def _head_loss(self, hidden, labels, chunk_size=128):
    return TrainableHeadCrossEntropy.apply(hidden, labels, self.weight, chunk_size)


def install_full_training_boundaries(model):
    """Call before the first forward, so FSDP initializes all trainable hooks."""
    from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method
    from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41HyperConnection

    count = 0
    for module in model.modules():
        if isinstance(module, DeepseekV41HyperConnection):
            if not hasattr(module, "_archlab_native_hc"):
                raise ValueError("install the verified native HC forward first")
            module.forward = MethodType(trainable_hc_forward, module)
            count += 1
    if count == 0:
        raise ValueError("no native HC modules found")
    model.lm_head.loss = MethodType(_head_loss, model.lm_head)
    register_fsdp_forward_method(model.lm_head, "loss")
    model.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, FSDPModule):
            # Losses are normalized once by the GLOBAL target count. Expert
            # all-reduce/gather and owner-table all-to-all already sum dWeights.
            module.set_gradient_divide_factor(1.0)
    return {
        "trainable_hc_modules": count,
        "head_loss": "chunked-full-vocabulary-FP32",
        "gradient_reductions": "sum; normalize loss by global targets",
    }
