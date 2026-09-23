"""Masked group-relative policy gradients for sampled model completions.

No language-model target or gold completion is accepted here: rewards belong to
fresh generated rollouts. Rollout collection, behavior-policy identity and prompt
grouping are the caller's responsibility. See function contracts for normalization
and the distinction between a score-function gradient and an importance surrogate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def group_relative_advantages(
    rewards: Tensor, *, baseline: str = "group_standardized", eps: float = 1e-8
) -> Tensor:
    """Detached group-relative rewards for ``[prompt, sample]`` groups.

    Each group must contain at least two samples. Groups with standard deviation
    at most eps have zero advantage (and therefore no policy-gradient signal).
    Scaling before moments avoids overflow for finite large reward magnitudes.
    ``group_standardized`` subtracts group mean and divides by population std;
    ``leave_one_out`` subtracts the other samples' mean without std normalization.
    """
    if rewards.ndim != 2 or rewards.shape[0] < 1 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape [prompts, samples>=2]")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if baseline not in ("group_standardized", "leave_one_out"):
        raise ValueError("baseline must be group_standardized or leave_one_out")
    dtype = torch.float64 if rewards.dtype == torch.float64 else torch.float32
    values = rewards.detach().to(dtype)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("rewards must be finite")
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1)
    scaled = values / scale
    centered = scaled - scaled.mean(dim=-1, keepdim=True)
    std = centered.square().mean(dim=-1, keepdim=True).sqrt()
    nonconstant = std > eps / scale
    if baseline == "group_standardized":
        advantages = centered / std.clamp_min(torch.finfo(dtype).tiny)
    else:
        count = rewards.shape[-1]
        advantages = centered * (count / (count - 1)) * scale
    result = torch.where(nonconstant, advantages, torch.zeros_like(advantages))
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("group-relative advantages overflowed")
    return result


@dataclass(frozen=True)
class PolicyLoss:
    loss: Tensor
    policy_loss: Tensor
    kl: Tensor
    clip_fraction: Tensor
    mean_ratio: Tensor
    completion_tokens: int
    sequences: int
    normalization: str
    has_policy_signal: bool
    skip_update: bool
    reference_kl_available: bool

    def metrics(self) -> dict[str, float | int | str]:
        result = {
            "loss": float(self.loss.detach()),
            "policy_loss": float(self.policy_loss.detach()),
            "clip_fraction": float(self.clip_fraction.detach()),
            "mean_ratio": float(self.mean_ratio.detach()),
            "completion_tokens": self.completion_tokens,
            "sequences": self.sequences,
            "normalization": self.normalization,
            "has_policy_signal": self.has_policy_signal,
            "skip_update": self.skip_update,
            "reference_kl_available": self.reference_kl_available,
        }
        if self.reference_kl_available:
            result["sampled_kl"] = float(self.kl.detach())
        return result


def group_relative_policy_loss(
    logprobs: Tensor,
    completion_mask: Tensor,
    advantages: Tensor,
    *,
    old_logprobs: Tensor | None = None,
    reference_logprobs: Tensor | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float = 0.0,
    normalization: str = "sequence",
    max_log_ratio: float = 60.0,
) -> PolicyLoss:
    """Group-relative loss over aligned completion tokens ``[B, G, T]``.

    ``completion_mask`` is boolean or exact 0/1, excludes all prompt/padding tokens,
    and includes EOS if it was sampled. Every completion must have >=1 valid token.
    ``advantages`` is [B,G] and detached here regardless of its input provenance.

    With no old_logprobs, this is the on-policy score-function loss ``-A*log p``.
    The caller must use fresh samples from the current policy and only one update.
    With old_logprobs, use ``-ratio*A`` where ratio=exp(log p-log p_behavior), with
    optional PPO min(unclipped, clipped) surrogate. Behavior scores are detached;
    clipping requires them. Their sampling distribution must match the policy
    scores (temperature/truncation must be accounted for by the caller).

    ``sequence_sum`` sums tokens within each completion, then averages completions:
    with the score-function path this is the usual trajectory-level RLOO gradient.
    ``sequence`` averages tokens within each completion, then completions equally
    (a length-normalized variant). ``token`` averages all completion tokens.
    Token log probabilities and masks must refer to the same next-token positions;
    the caller must shift logits/targets correctly before passing them here.
    KL uses the sampled-token k3 regularizer exp(ref-current)-(ref-current)-1,
    computed with expm1 for accuracy. It is not full-vocabulary exact KL. Reference
    scores are detached. Nonzero beta requires explicit reference log probabilities.
    Excessive log ratios fail rather than silently clamping/changing the objective.
    """
    if logprobs.ndim != 3 or min(logprobs.shape) < 1 or logprobs.shape[1] < 2:
        raise ValueError("logprobs must have shape [prompts, samples>=2, tokens]")
    if not logprobs.is_floating_point():
        raise ValueError("logprobs must be floating point")
    if completion_mask.shape != logprobs.shape or completion_mask.device != logprobs.device:
        raise ValueError("completion_mask must match logprobs shape and device")
    if not bool(((completion_mask == 0) | (completion_mask == 1)).all()):
        raise ValueError("completion_mask must contain only 0 or 1")
    mask = completion_mask.bool()
    lengths = mask.sum(dim=-1)
    if bool((lengths == 0).any()):
        raise ValueError("each completion must contain at least one scored token")
    if advantages.shape != logprobs.shape[:2] or advantages.device != logprobs.device:
        raise ValueError("advantages must match prompt/sample shape and device")
    if not bool(torch.isfinite(advantages).all()):
        raise ValueError("advantages must be finite")
    if normalization not in ("sequence_sum", "sequence", "token"):
        raise ValueError("normalization must be sequence_sum, sequence or token")
    if not math.isfinite(kl_beta) or kl_beta < 0:
        raise ValueError("kl_beta must be finite and nonnegative")
    if kl_beta and reference_logprobs is None:
        raise ValueError("nonzero kl_beta requires reference_logprobs")
    if clip_epsilon is not None:
        if old_logprobs is None:
            raise ValueError("clipping requires old_logprobs from the behavior policy")
        if not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
            raise ValueError("clip_epsilon must lie strictly between 0 and 1")
    if not math.isfinite(max_log_ratio) or not 0 < max_log_ratio <= 60:
        raise ValueError("max_log_ratio must lie in (0,60]")
    dtype = torch.float64 if logprobs.dtype == torch.float64 else torch.float32

    def scores(value: Tensor, name: str, *, detach: bool) -> Tensor:
        if value.shape != logprobs.shape or value.device != logprobs.device:
            raise ValueError(f"{name} must match logprobs shape and device")
        if not value.is_floating_point() or not bool(torch.isfinite(value[mask]).all()):
            raise ValueError(f"{name} must have finite floating point completion scores")
        if bool((value[mask] > 1e-5).any()):
            raise ValueError(f"{name} must be normalized log probabilities <=0")
        value = value.detach() if detach else value
        return torch.where(mask, value.to(dtype), torch.zeros((), device=value.device, dtype=dtype))

    def bounded_difference(first: Tensor, second: Tensor, name: str) -> Tensor:
        difference = first - second
        if not bool(torch.isfinite(difference).all()) or bool((difference.abs() > max_log_ratio).any()):
            raise FloatingPointError(f"{name} exceeds max_log_ratio; reject this update")
        return difference

    def reduce(value: Tensor) -> Tensor:
        masked = torch.where(mask, value, torch.zeros((), device=value.device, dtype=dtype))
        if normalization == "sequence_sum":
            return masked.sum(dim=-1).mean()
        if normalization == "sequence":
            return (masked.sum(dim=-1) / lengths).mean()
        return masked.sum() / lengths.sum()

    current = scores(logprobs, "logprobs", detach=False)
    advantage = advantages.detach().to(dtype).unsqueeze(-1)
    clipped = torch.zeros_like(current, dtype=torch.bool)
    ratio = torch.ones_like(current)
    if old_logprobs is None:
        policy_tokens = -advantage * current
    else:
        old = scores(old_logprobs, "old_logprobs", detach=True)
        ratio = bounded_difference(current, old, "behavior log ratio").exp()
        surrogate = ratio * advantage
        if clip_epsilon is not None:
            clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
            clipped = (ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)
            surrogate = torch.minimum(surrogate, clipped_ratio * advantage)
        policy_tokens = -surrogate
    kl_tokens = torch.zeros_like(current)
    if reference_logprobs is not None:
        reference = scores(reference_logprobs, "reference_logprobs", detach=True)
        delta = bounded_difference(reference, current, "reference log ratio")
        kl_tokens = torch.expm1(delta) - delta
    policy_loss = reduce(policy_tokens)
    kl = reduce(kl_tokens)
    loss = policy_loss + kl_beta * kl
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite policy loss; reject this update")
    has_policy_signal = bool((advantages != 0).any())
    return PolicyLoss(
        loss=loss,
        policy_loss=policy_loss,
        kl=kl,
        clip_fraction=(clipped & mask).sum().to(dtype).div(lengths.sum()).detach(),
        mean_ratio=ratio[mask].mean().detach(),
        completion_tokens=int(lengths.sum()),
        sequences=lengths.numel(),
        normalization=normalization,
        has_policy_signal=has_policy_signal,
        skip_update=not has_policy_signal and kl_beta == 0,
        reference_kl_available=reference_logprobs is not None,
    )
