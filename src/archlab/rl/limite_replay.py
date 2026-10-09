"""Remove masked replay tails while retaining the global DAPO denominator."""

from __future__ import annotations

import torch


def trim_replay(inputs):
    """Trim only padding, never model-generated tokens or the group normalizer."""
    result = dict(inputs)
    prompt_mask, mask = inputs["prompt_mask"], inputs["completion_mask"]
    if not bool(mask.any()):
        return result
    first = int(prompt_mask.any(0).nonzero()[0])
    last = int(mask.any(0).nonzero()[-1]) + 1
    if not bool(prompt_mask[:, first:].all()):
        # Mixed prompt lengths are legitimate in eval. Keep their left padding.
        first = 0
    for key in ("prompt_ids", "prompt_mask"):
        result[key] = inputs[key][:, first:]
    for key in (
        "completion_ids", "completion_mask", "tool_mask", "old_per_token_logps",
        "sampling_per_token_logps", "ref_per_token_logps", "importance_sampling_ratio",
    ):
        if key in inputs:
            result[key] = inputs[key][:, :last]
    if inputs["advantages"].ndim == 2:
        result["advantages"] = inputs["advantages"][:, :last]
    return result


def global_signal(inputs, device):
    """All ranks agree before skipping any DDP forward/backward collective."""
    advantages = inputs["advantages"]
    mask = inputs["completion_mask"]
    if "tool_mask" in inputs:
        mask = mask * inputs["tool_mask"]
    active_advantages = advantages[:, None] if advantages.ndim == 1 else advantages
    signal = (active_advantages * mask != 0).any()
    value = signal.to(device=device, dtype=torch.int32)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return bool(value)
