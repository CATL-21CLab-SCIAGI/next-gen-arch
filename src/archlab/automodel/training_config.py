"""Portable policy for the single frozen-backbone, one-pass finetuning run."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler


@dataclass(frozen=True)
class TrainingConfig:
    world_size: int = 32
    ep_size: int = 8
    sequence_length: int = 16384
    micro_batch: int = 1
    seed: int = 1234
    adapter_seed: int = 42
    initial_lr: float = 1e-7
    peak_lr: float = 1e-5
    minimum_lr: float = 1e-6
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    gradient_clip: float = 1.0
    validation_interval: int = 100
    early_validation_step: int = 10
    validation_batches: int = 4
    checkpoint_interval: int = 1000

    def __post_init__(self) -> None:
        counts = (self.world_size, self.ep_size, self.sequence_length, self.micro_batch,
                  self.warmup_steps, self.validation_interval, self.early_validation_step,
                  self.validation_batches, self.checkpoint_interval)
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("training dimensions and intervals must be positive integers")
        if self.world_size % self.ep_size or not 2 <= self.ep_size <= 8:
            raise ValueError("use the qualified divisible DP/node-local EP layout")
        if self.micro_batch != 1:
            raise ValueError("this first launch uses the qualified microbatch one, without accumulation")
        scales = (self.initial_lr, self.peak_lr, self.minimum_lr, self.epsilon, self.gradient_clip)
        if any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("learning rates, epsilon and gradient clipping must be finite and positive")
        if self.initial_lr > self.peak_lr or self.minimum_lr > self.peak_lr:
            raise ValueError("initial/minimum LR must not exceed peak LR")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight decay must be finite and nonnegative")
        if not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1:
            raise ValueError("Adam betas must be strictly between zero and one")

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.AdamW:
        """Capture only already-installed, already-sharded trainable parameters."""
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=self.initial_lr, betas=(self.beta1, self.beta2),
                                 eps=self.epsilon, weight_decay=self.weight_decay, foreach=False)

    def build_scheduler(self, optimizer: torch.optim.Optimizer, *, total_steps: int) -> OptimizerParamScheduler:
        """Reuse the backend's linear warm-up and cosine decay, one step per data cursor."""
        from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler

        if total_steps <= self.warmup_steps:
            raise ValueError("the one-pass budget must exceed warm-up")
        return OptimizerParamScheduler(
            optimizer, init_lr=self.initial_lr, max_lr=self.peak_lr, min_lr=self.minimum_lr,
            lr_warmup_steps=self.warmup_steps, lr_decay_steps=total_steps, lr_decay_style="cosine",
            start_wd=self.weight_decay, end_wd=self.weight_decay, wd_incr_steps=total_steps,
            wd_incr_style="constant", use_checkpoint_opt_param_scheduler=False)
