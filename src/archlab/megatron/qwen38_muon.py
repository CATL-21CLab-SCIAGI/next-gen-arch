"""Qwen3.8-specific parameter splitting on Megatron's native distributed Muon."""

from __future__ import annotations

import math
from typing import Any

import torch

POLAR_EXPRESS_COEFFICIENTS = (
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
)
FROBENIUS_EPSILON = 1e-14
MUON_MOMENTUM = 0.95
MUON_EXTRA_SCALE = 0.2


def _validate_local_matrix_metadata(parameter: torch.Tensor, tp_size: int) -> None:
    """Accept TE's partition metadata only when the TP shard is the whole matrix."""
    if tp_size != 1:
        raise ValueError("the Qwen3.8 adapter currently requires TP=EP=1")
    partition_dim = getattr(parameter, "partition_dim", None)
    if partition_dim not in (None, -1, 0, 1):
        raise ValueError(f"invalid 2D parameter partition dimension: {partition_dim}")


def polar_express_zeroth_power(
    matrices: torch.Tensor,
    *,
    use_bfloat16_matmul: bool,
) -> torch.Tensor:
    """Eight-step Polar Express orthogonalization over a batch of matrices."""
    if matrices.ndim != 3:
        raise ValueError("Muon logical matrices must have shape [count, rows, columns]")
    transposed = matrices.size(-2) > matrices.size(-1)
    value = matrices.mT if transposed else matrices
    norm = value.square().sum(dim=(-2, -1), keepdim=True).sqrt()
    value = value / norm.clamp_min(FROBENIUS_EPSILON)
    if use_bfloat16_matmul:
        value = value.to(torch.bfloat16)
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
        gram = value @ value.mT
        polynomial = b * gram + c * (gram @ gram)
        value = a * value + polynomial @ value
    value = value.float()
    return value.mT if transposed else value


def muon_recipe_contract() -> dict[str, Any]:
    return {
        "implementation": "Megatron layer-wise distributed Muon with Qwen3.8 adapter",
        "momentum": MUON_MOMENTUM,
        "nesterov": True,
        "coefficient_schedule": "polar_express",
        "newton_schulz_steps": len(POLAR_EXPRESS_COEFFICIENTS),
        "frobenius_normalization_epsilon": FROBENIUS_EPSILON,
        "scale": "0.2 * sqrt(max(rows, columns)) per logical matrix",
        "fp32_matmul_precision": "medium (BF16 Newton-Schulz matmuls)",
        "fused_parameter_splitting": {
            "attention_q": "per head",
            "gdn_qkv": "per q/k/v head",
            "expert_fc1": "gate/up halves",
        },
        "scalar_optimizer": "AdamW; Adam for no-decay n-gram tables",
        "distributed": {
            "optimizer": "Megatron LayerWiseDistributedOptimizer",
            "parameter_layout": "longest-processing-time whole-parameter placement",
            "gradient_communication": "reduce-scatter overlapped with backward",
            "adam_state": "Megatron distributed optimizer",
        },
    }


def install_qwen38_muon_adapter(config) -> None:
    """Install a process-local adapter without altering container packages."""
    from megatron.core.optimizer import emerging_optimizers
    from megatron.core.optimizer.optimizer_config import ParamKey, ParamPredicate
    from megatron.core.utils import get_pg_size

    entry = emerging_optimizers._EMERGING_OPTIMIZERS["muon"]
    if getattr(entry.optimizer_cls, "_archlab_qwen38_muon", False):
        return
    native_muon = emerging_optimizers.TensorParallelMuon

    class Qwen38TensorParallelMuon(native_muon):
        _archlab_qwen38_muon = True

        def __init__(self, *args, **kwargs):
            expected = {
                "momentum": MUON_MOMENTUM,
                "nesterov": True,
                "coefficient_type": "polar_express",
                "num_ns_steps": len(POLAR_EXPRESS_COEFFICIENTS),
                "scale_mode": "spectral",
                "extra_scale_factor": MUON_EXTRA_SCALE,
                "fp32_matmul_prec": "medium",
            }
            drift = {
                key: (kwargs.get(key), value)
                for key, value in expected.items()
                if kwargs.get(key) != value
            }
            if drift:
                raise ValueError(f"Qwen3.8 Muon recipe drift: {drift}")
            super().__init__(*args, **kwargs)
            self._qwen_use_bfloat16_matmul = kwargs["fp32_matmul_prec"] == "medium"
            self._qwen_scale_mode = kwargs["scale_mode"]
            self._qwen_extra_scale_factor = kwargs["extra_scale_factor"]

        def orthogonalize(
            self,
            parameter: torch.Tensor,
            gradient: torch.Tensor,
            **group_kwargs: Any,
        ) -> torch.Tensor:
            if gradient.ndim != 2:
                raise ValueError("Qwen3.8 Muon requires 2D physical parameters")
            tp_size = 1
            if self.pg_collection:
                tp_group = (
                    self.pg_collection.expt_tp
                    if getattr(parameter, "expert_tp", False)
                    else self.pg_collection.tp
                )
                tp_size = get_pg_size(tp_group)
            # Transformer Engine labels column/row-parallel weights with
            # partition_dim even at TP=1.  Such a tensor is still the complete
            # physical matrix, so its Qwen-specific logical splits are valid.
            _validate_local_matrix_metadata(parameter, tp_size)
            if self._qwen_scale_mode != "spectral":
                raise ValueError("Qwen3.8 requires spectral Muon scaling")

            split_rows = int(group_kwargs.get("archlab_muon_split_rows", gradient.size(0)))
            if gradient.size(0) % split_rows:
                raise ValueError(
                    f"cannot split {tuple(gradient.shape)} into {split_rows}-row matrices"
                )
            logical = gradient.view(-1, split_rows, gradient.size(1))
            update = polar_express_zeroth_power(
                logical,
                use_bfloat16_matmul=self._qwen_use_bfloat16_matmul,
            )
            scale = self._qwen_extra_scale_factor * math.sqrt(
                max(split_rows, gradient.size(1))
            )
            return (update * scale).view_as(gradient)

    split_rows = {
        config.linear_key_dim,
        config.attention_head_dim,
        config.moe_intermediate_size,
        config.shared_expert_intermediate_size,
    }
    overrides = dict(entry.default_param_overrides)
    overrides[ParamKey(attr="archlab_no_weight_decay")] = {"wd_mult": 0.0}
    for rows in sorted(split_rows):
        predicate = ParamPredicate(
            name=f"archlab_muon_split_rows_{rows}",
            fn=lambda parameter, expected=rows: getattr(
                parameter, "archlab_muon_split_rows", None
            )
            == expected,
        )
        overrides[ParamKey(predicate=predicate)] = {"archlab_muon_split_rows": rows}

    entry.optimizer_cls = Qwen38TensorParallelMuon
    entry.default_param_overrides = overrides
