"""Memory-bounded policy log probabilities on the trained FP32 vocabulary head."""

from types import MethodType

import torch
from torch.nn import functional as F


class SelectedLogProbabilities(torch.autograd.Function):
    """Retain hidden states and targets, recomputing vocabulary chunks in backward.

    Labels already identify the next token at each hidden-state position. This
    boundary does not shift labels or choose a policy-gradient reduction.
    """

    @staticmethod
    def forward(ctx, hidden, labels, weight, chunk_size=128):
        if weight.dtype != torch.float32 or weight.ndim != 2:
            raise ValueError("RL retains the trained FP32 vocabulary matrix")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if (hidden.ndim < 2 or hidden.shape[:-1] != labels.shape
                or hidden.shape[-1] != weight.shape[1]):
            raise ValueError("hidden, target, and vocabulary shapes differ")
        if labels.dtype != torch.long or labels.device != hidden.device or weight.device != hidden.device:
            raise ValueError("targets must be int64 on the hidden/head device")
        targets = labels.reshape(-1)
        valid = torch.where(targets != -100)[0]
        if valid.numel() and bool(((targets[valid] < 0) | (targets[valid] >= weight.shape[0])).any()):
            raise ValueError("target outside the vocabulary")
        x = hidden.reshape(-1, hidden.shape[-1])
        result = torch.zeros_like(targets, dtype=torch.float32)
        with torch.autocast(hidden.device.type, enabled=False):
            for ids in valid.split(chunk_size):
                if ids.numel():
                    logits = F.linear(x[ids].float(), weight)
                    result[ids] = logits.log_softmax(-1).gather(1, targets[ids, None]).squeeze(1)
        ctx.save_for_backward(hidden, labels, weight, valid)
        ctx.chunk_size = chunk_size
        return result.reshape(labels.shape)

    @staticmethod
    def backward(ctx, upstream):
        hidden, labels, weight, valid = ctx.saved_tensors
        x, targets = hidden.reshape(-1, hidden.shape[-1]), labels.reshape(-1)
        gradient = upstream.reshape(-1)
        dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[2] else None
        with torch.autocast(hidden.device.type, enabled=False):
            for ids in valid.split(ctx.chunk_size):
                if ids.numel():
                    values = x[ids].float()
                    # d log p(target) / d logits = one_hot(target) - softmax.
                    dlogits = -F.linear(values, weight).softmax(-1)
                    dlogits[torch.arange(ids.numel(), device=ids.device), targets[ids]] += 1
                    dlogits.mul_(gradient[ids, None])
                    if dx is not None:
                        dx[ids] = (dlogits @ weight).to(dx.dtype)
                    if dw is not None:
                        dw.addmm_(dlogits.T, values)
        return None if dx is None else dx.reshape_as(hidden), None, dw, None


def selected_log_probs(hidden, labels, weight, chunk_size=128):
    """Return FP32 log p(labels) with zero values/gradients at labels == -100."""
    return SelectedLogProbabilities.apply(hidden, labels, weight, chunk_size)


def _head_log_probs(self, hidden, labels, chunk_size=128):
    return selected_log_probs(hidden, labels, self.weight, chunk_size)


def install_rl_head(model):
    """Add an FSDP-managed log-probability method without replacing CE or weights.

    Install once, before the first RL forward. Existing checkpoint tensor names,
    the ordinary ``lm_head.loss`` method, and parameter trainability are retained.
    """
    import inspect

    from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method

    head = model.lm_head
    if hasattr(head, "rl_log_probs"):
        raise ValueError("RL head is already installed")
    if head.weight.dtype != torch.float32:
        raise ValueError("RL requires the original FP32 vocabulary head")
    head.rl_log_probs = MethodType(_head_log_probs, head)
    if isinstance(head, FSDPModule):
        register_fsdp_forward_method(head, "rl_log_probs")
    # The resident NeMo wrapper computes logits even when hidden states are
    # requested. RL projects its selected positions separately; retaining one
    # unused position avoids a full [batch, context, vocabulary] allocation.
    if hasattr(model, "forward") and "logits_to_keep" in inspect.signature(model.forward).parameters:
        model._archlab_rl_hidden_forward_kwargs = {"logits_to_keep": 1}
    return {"method": "rl_log_probs", "dtype": "float32", "ignored_target": -100,
            "hidden_forward_kwargs": dict(getattr(model, "_archlab_rl_hidden_forward_kwargs", {})),
            "vocabulary_memory": "bounded-chunks-recomputed-in-backward"}
