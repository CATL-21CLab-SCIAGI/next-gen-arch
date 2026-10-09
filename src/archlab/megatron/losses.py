"""Explicit legacy-mean and native-token-count callback ABIs; not interchangeable."""

from __future__ import annotations

from functools import partial

import torch


def component_mean_loss(
    output_tensor: torch.Tensor,
    component_metrics: dict[str, torch.Tensor] | None = None,
):
    losses = output_tensor.reshape(-1).float()
    count = torch.tensor(losses.numel(), dtype=torch.float32, device=losses.device)
    loss_sum = losses.sum()
    report = {"lm loss": torch.stack((loss_sum.detach(), count))}
    for name, value in (component_metrics or {}).items():
        metric_count = torch.ones((), dtype=torch.float32, device=value.device)
        report[name] = torch.stack((value.float(), metric_count))
    return loss_sum / count, report


def masked_token_loss(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Return Megatron's per-token ABI: summed loss, valid-token count, metrics.

    The native finalizer divides all gradients, including MoE/MTP auxiliary
    gradients, by the global token count. A legacy two-item, pre-averaged
    callback leaves that count at zero and breaks their relative scaling.
    """
    losses = output_tensor.reshape(-1).float()
    mask = loss_mask.reshape(-1).float()
    loss_sum = (losses * mask).sum()
    count = mask.sum(dtype=torch.int64)
    return loss_sum, count, {"lm loss": torch.stack((loss_sum.detach(), count))}


def native_token_forward_step(data_iterator, model, return_schedule_plan: bool = False):
    """Native causal-token model ABI shared by Flash-Next training and pilots."""
    if return_schedule_plan:
        raise NotImplementedError("the full Qwen adapter does not use schedule plans")
    batch = next(data_iterator)
    tokens, labels = batch["tokens"], batch["labels"]
    positions = torch.arange(tokens.size(1), device=tokens.device).expand_as(tokens)
    loss_mask = labels.ne(-1).float()
    losses = model(tokens, positions, None, labels=labels, loss_mask=loss_mask)
    return losses, partial(masked_token_loss, loss_mask)
