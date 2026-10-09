"""Preserve warmup adapter AdamW while adding native-dtype backbone masters."""

from __future__ import annotations

import math

import torch

FORMAT = "archlab-limite-split-adamw-v1"


def warmup_learning_rates(step, tokens, schedule):
    """Continue the original adapter schedule and the persisted full-phase clock."""
    target = schedule["target_tokens"]
    origin_step = schedule["full_weight_start_step"]
    origin_tokens = schedule["full_weight_start_tokens"]
    adapter = (
        schedule["adapter_peak_lr"]
        * min(1.0, (step + 1) / schedule["adapter_warmup_steps"])
        * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * tokens / target)))
    )
    phase_progress = (tokens - origin_tokens) / (target - origin_tokens)
    backbone = (
        schedule["backbone_peak_lr"]
        * min(1.0, (step - origin_step + 1) / schedule["backbone_warmup_steps"])
        * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * phase_progress)))
    )
    return adapter, backbone


class FullWeightWarmupAdamW:
    """Composition of unchanged Torch adapter AdamW and container TE AdamW."""

    def __init__(self, adapter, backbone):
        self.adapter = adapter
        self.backbone = backbone

    @property
    def param_groups(self):
        return self.adapter.param_groups + self.backbone.param_groups

    def zero_grad(self, set_to_none=True):
        self.adapter.zero_grad(set_to_none=set_to_none)
        self.backbone.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.adapter.step()
        self.backbone.step()

    def state_dict(self):
        return {
            "format": FORMAT,
            "adapter": self.adapter.state_dict(),
            "backbone": self.backbone.state_dict(),
        }

    def load_state_dict(self, state):
        if state.get("format") == FORMAT:
            self.adapter.load_state_dict(state["adapter"])
            self.backbone.load_state_dict(state["backbone"])
        elif "state" in state and "param_groups" in state:
            # The native 2B checkpoint has no backbone training history.
            self.adapter.load_state_dict(state)
        else:
            raise ValueError("unsupported full-weight warmup optimizer state")


def build_warmup_optimizer(model, *, lr=1e-4, backbone_lr=1e-5, betas=(0.9, 0.95), weight_decay=0):
    adapter_parameters = list(model.model.adapters.parameters())
    adapter = torch.optim.AdamW(
        adapter_parameters, lr=lr, betas=betas, weight_decay=weight_decay, fused=True
    )
    if getattr(model.model, "trainable_mode", "adapter") == "adapter":
        return adapter
    from transformer_engine.pytorch.optimizers import FusedAdam

    adapter_ids = {id(parameter) for parameter in adapter_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in adapter_ids
    ]
    if not backbone_parameters:
        raise ValueError("full-weight warmup requires trainable backbone parameters")
    backbone = FusedAdam(
        backbone_parameters,
        lr=backbone_lr,
        betas=betas,
        weight_decay=weight_decay,
        adam_w_mode=True,
        master_weights=True,
        master_weight_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
    )
    return FullWeightWarmupAdamW(adapter, backbone)
