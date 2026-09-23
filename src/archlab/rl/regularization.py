"""Bounded length pressure for already-correct group-relative rollouts."""

import math
import torch


DEFAULT_LENGTH_PENALTY = {
    "enabled": False, "minimum_pass_rate": .25, "reference_quantile": .5,
    "maximum_deduction": .1, "tolerated_excess": .1, "saturation_excess": 1., "exponent": 1.,
}


def validate_length_penalty(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULT_LENGTH_PENALTY):
        raise ValueError("length penalty must declare every supported field")
    if type(value["enabled"]) is not bool:
        raise ValueError("length penalty enabled must be boolean")
    for name in set(value) - {"enabled"}:
        x = value[name]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError(f"length penalty {name} must be a finite number")
    if not (0 <= value["minimum_pass_rate"] <= 1 and 0 < value["reference_quantile"] <= 1
            and 0 <= value["maximum_deduction"] < 1 and value["tolerated_excess"] >= 0
            and value["saturation_excess"] > value["tolerated_excess"] and value["exponent"] >= 1):
        raise ValueError("invalid correctness-preserving length penalty bounds")


def successful_length_rewards(rewards, lengths, config):
    """Keep failures at zero and correct answers above zero; leave hard groups alone."""
    validate_length_penalty(config)
    if (rewards.ndim != 2 or lengths.shape != rewards.shape or lengths.device != rewards.device
            or not bool(((rewards == 0) | (rewards == 1)).all())
            or not bool(torch.isfinite(lengths).all()) or bool((lengths <= 0).any())):
        raise ValueError("length regularization requires binary group rewards and positive lengths")
    adjusted = rewards.detach().float().clone()
    eligible = penalized = 0
    if config["enabled"]:
        for index in range(rewards.shape[0]):
            success = rewards[index] == 1
            if int(success.sum()) < 2 or float(success.float().mean()) <= config["minimum_pass_rate"]:
                continue
            eligible += 1
            reference = torch.quantile(lengths[index, success].float(), config["reference_quantile"])
            excess = lengths[index].float() / reference - 1
            ramp = ((excess - config["tolerated_excess"]) /
                    (config["saturation_excess"] - config["tolerated_excess"])).clamp(0, 1)
            deduction = config["maximum_deduction"] * ramp.pow(config["exponent"]) * success
            adjusted[index] -= deduction
            penalized += int((deduction > 0).sum())
    return adjusted, {"eligible_groups": eligible, "penalized_successes": penalized,
                      "total_deduction": float((rewards - adjusted).sum())}
