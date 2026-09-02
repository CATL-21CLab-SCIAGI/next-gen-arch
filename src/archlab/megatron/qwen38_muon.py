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


def _combine_grad_norms(norms: list[float | torch.Tensor]) -> float | torch.Tensor:
    """Combine optimizer-local L2 norms without synchronizing CUDA to the host."""
    tensor = next((value for value in norms if isinstance(value, torch.Tensor)), None)
    if tensor is None:
        return math.sqrt(sum(float(value) ** 2 for value in norms))
    values = [
        value if isinstance(value, torch.Tensor) else tensor.new_tensor(float(value))
        for value in norms
    ]
    return torch.stack(values).square().sum().sqrt()


def _canonical_optimizer_step(steps: list[int | torch.Tensor]) -> int | torch.Tensor | None:
    """Validate equal chained-optimizer counters by value, not tensor identity."""
    if not steps:
        return None
    values = [int(step.item()) if isinstance(step, torch.Tensor) else int(step) for step in steps]
    if len(set(values)) != 1:
        raise ValueError(f"chained optimizer step counters diverged: {values}")
    return steps[0]


def _filter_and_reorder_optimizer_groups(
    current_groups: list[dict[str, Any]],
    loaded_groups: list[dict[str, Any]],
    identifier_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Match checkpoint groups one-to-one, including duplicate identifier keys.

    Megatron's fresh-container implementation stores only one loaded group per
    identifier tuple in a dict.  Muon's logical matrix splits deliberately
    produce several groups with the same scheduler identifiers, so that code
    aliases every duplicate to the last group and breaks optimizer resume.
    Preserve the checkpoint order within each identifier tuple instead.
    """

    if len(current_groups) != len(loaded_groups):
        raise ValueError(
            "optimizer parameter group count changed: "
            f"{len(loaded_groups)} checkpoint groups != {len(current_groups)} current groups"
        )

    def key(group: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            group[name] if name in group else group[f"pre_{name}"]
            for name in identifier_keys
        )

    queues: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    runtime_params_by_position = [group["params"] for group in loaded_groups]
    for group in loaded_groups:
        queues.setdefault(key(group), []).append(group)

    reordered = []
    for current_group, runtime_params in zip(
        current_groups,
        runtime_params_by_position,
        strict=True,
    ):
        group_key = key(current_group)
        candidates = queues.get(group_key, [])
        if not candidates:
            available = "\n".join(str(value) for value, groups in queues.items() if groups)
            raise ValueError(
                f"could not find checkpoint optimizer group {group_key}; "
                f"available groups:\n{available}"
            )
        group = candidates.pop(0)
        group["params"] = runtime_params
        reordered.append(group)

    leftovers = {group_key: len(groups) for group_key, groups in queues.items() if groups}
    if leftovers:
        raise ValueError(f"unused checkpoint optimizer groups: {leftovers}")
    return reordered


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
        "optimizer_cuda_graph_compatibility": (
            "repository-local capture-safe chained grad-norm reduction; no host scalar conversion"
        ),
        "checkpoint_step_compatibility": (
            "repository-local value comparison for equal capturable CUDA step tensors"
        ),
    }


def install_qwen38_muon_adapter(config) -> None:
    """Install a process-local adapter without altering container packages."""
    from megatron.core.optimizer import emerging_optimizers
    from megatron.core.optimizer import optimizer as optimizer_module
    from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32
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

    # Process-local correction for duplicate Muon parameter-group identifiers;
    # the container checkout remains pristine and replaceable.
    optimizer_module.MegatronOptimizer._filter_and_reorder_param_groups = staticmethod(
        lambda current, loaded: _filter_and_reorder_optimizer_groups(
            current,
            loaded,
            optimizer_module.param_group_identifier_keys,
        )
    )

    chained_optimizer = optimizer_module.ChainedOptimizer
    if getattr(chained_optimizer, "_archlab_qwen38_capture_safe", False):
        return
    original_get_grad_norm = chained_optimizer.get_grad_norm

    @torch.no_grad()
    def capture_safe_get_grad_norm(self):
        if len(self.chained_optimizers) == 1 or self.grads_states_parallel_group_is_shared():
            return original_get_grad_norm(self)
        return _combine_grad_norms(
            [optimizer.get_grad_norm() for optimizer in self.chained_optimizers]
        )

    @torch.no_grad()
    def capture_safe_step(self):
        found_inf = self.prepare_grads()
        if found_inf:
            return False, None, None

        grad_norm = self.get_grad_norm()
        for optimizer in self.chained_optimizers:
            if getattr(optimizer, "is_stub_optimizer", False):
                continue
            parameters = optimizer.get_parameters()
            if not parameters:
                continue
            threshold = optimizer.config.grad_norm_skip_threshold
            if not math.isinf(threshold):
                raise ValueError(
                    "optimizer CUDA graph requires an infinite grad-norm skip threshold; "
                    "finite clipping remains enabled independently"
                )
            if optimizer.config.clip_grad > 0.0:
                clip_grad_by_total_norm_fp32(
                    parameters,
                    max_norm=optimizer.config.clip_grad,
                    total_norm=grad_norm,
                    use_decoupled_grad=(
                        optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
                        or (
                            optimizer.config.use_precision_aware_optimizer
                            and getattr(parameters[0], "__fsdp_param__", False)
                        )
                    ),
                )

        num_zeros = self.count_zeros() if self.config.log_num_zeros_in_grad else None
        return self.step_with_ready_grads(), grad_norm, num_zeros

    def checkpoint_safe_synchronize_steps(self):
        groups = [
            group
            for optimizer in self.chained_optimizers
            for group in optimizer.optimizer.param_groups
            if group["params"] and "step" in group
        ]
        step = _canonical_optimizer_step([group["step"] for group in groups])
        for group in groups:
            group["step"] = step
        return step

    chained_optimizer.get_grad_norm = capture_safe_get_grad_norm
    chained_optimizer.step = capture_safe_step
    chained_optimizer._synchronize_steps = checkpoint_safe_synchronize_steps
    chained_optimizer._archlab_qwen38_capture_safe = True
