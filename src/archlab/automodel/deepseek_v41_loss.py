"""Memory-bounded, full-vocabulary FP32 CE with a frozen native output head."""

from __future__ import annotations

import torch
from torch.nn import functional as F


class FrozenHeadCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, labels, weight, chunk_size=128):
        if weight.requires_grad or weight.dtype != torch.float32 or chunk_size < 1:
            raise ValueError("the native frozen FP32 output head and positive chunk size are required")
        if hidden.shape[:-1] != labels.shape or hidden.shape[-1] != weight.shape[1]:
            raise ValueError("hidden/label/head shapes do not match")
        x, targets = hidden.flatten(0, -2), labels.flatten()
        valid = torch.where(targets != -100)[0]
        loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for ids in valid.split(chunk_size):
            if ids.numel():
                loss += F.cross_entropy(F.linear(x[ids].float(), weight), targets[ids], reduction="sum")
        ctx.save_for_backward(hidden, labels, weight, valid)
        ctx.chunk_size = chunk_size
        return loss

    @staticmethod
    def backward(ctx, dl):
        hidden, labels, weight, valid = ctx.saved_tensors
        x, targets = hidden.flatten(0, -2), labels.flatten()
        dx = torch.zeros_like(x)
        for ids in valid.split(ctx.chunk_size):
            if ids.numel():
                probabilities = F.linear(x[ids].float(), weight).softmax(-1)
                probabilities[torch.arange(ids.numel(), device=ids.device), targets[ids]] -= 1
                dx[ids] = (probabilities @ weight * dl).to(dx.dtype)
        return dx.reshape_as(hidden), None, None, None


def frozen_head_loss(hidden, labels, head, chunk_size=128):
    return FrozenHeadCrossEntropy.apply(hidden, labels, head.weight, chunk_size)
