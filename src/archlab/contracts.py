"""Research-grade comparison and training-budget contracts.

The historical campaigns remain readable exactly as published.  New campaigns
must state which scientific question they answer instead of overloading a
display label such as ``100m`` with incompatible budget semantics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ContractError(ValueError):
    """Raised when a comparison contract is incomplete or internally inconsistent."""


class ComparisonRegime(str, Enum):
    """The three supported experiment questions."""

    CONTROLLED = "controlled"
    FIXED_COMPUTE = "fixed_compute"
    SCALING = "scaling"


@dataclass(frozen=True)
class ComparisonContract:
    """Controls that make a baseline/variant comparison interpretable."""

    regime: ComparisonRegime
    baseline_variant: str = "baseline"
    paired_seed: bool = True
    paired_data_order: bool = True
    shared_initialization: str = "hash"
    primary_metric: str = "validation_bpb"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ComparisonContract:
        try:
            regime = ComparisonRegime(str(value["regime"]))
        except KeyError as error:
            raise ContractError("comparison.regime is required") from error
        except ValueError as error:
            choices = ", ".join(item.value for item in ComparisonRegime)
            raise ContractError(f"comparison.regime must be one of: {choices}") from error
        contract = cls(
            regime=regime,
            baseline_variant=str(value.get("baseline_variant", "baseline")),
            paired_seed=bool(value.get("paired_seed", True)),
            paired_data_order=bool(value.get("paired_data_order", True)),
            shared_initialization=str(value.get("shared_initialization", "hash")),
            primary_metric=str(value.get("primary_metric", "validation_bpb")),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if not self.baseline_variant:
            raise ContractError("comparison.baseline_variant must not be empty")
        if self.shared_initialization not in {"hash", "bit_identical", "not_applicable"}:
            raise ContractError(
                "comparison.shared_initialization must be hash, bit_identical, or "
                "not_applicable"
            )
        if not self.primary_metric:
            raise ContractError("comparison.primary_metric must not be empty")


@dataclass(frozen=True)
class ArtifactPolicy:
    """Durability requirements for a reportable run."""

    mode: str = "research"
    save_final_checkpoint: bool = True
    save_optimizer_state: bool = True
    save_dataloader_state: bool = True
    raw_metrics: bool = True
    require_data_manifest: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArtifactPolicy:
        policy = cls(
            mode=str(value.get("mode", "research")),
            save_final_checkpoint=bool(value.get("save_final_checkpoint", True)),
            save_optimizer_state=bool(value.get("save_optimizer_state", True)),
            save_dataloader_state=bool(value.get("save_dataloader_state", True)),
            raw_metrics=bool(value.get("raw_metrics", True)),
            require_data_manifest=bool(value.get("require_data_manifest", True)),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.mode not in {"research", "metrics_only", "frozen_historical"}:
            raise ContractError(
                "artifacts.mode must be research, metrics_only, or frozen_historical"
            )
        if self.mode == "research" and not all(
            (
                self.save_final_checkpoint,
                self.save_optimizer_state,
                self.save_dataloader_state,
                self.raw_metrics,
                self.require_data_manifest,
            )
        ):
            raise ContractError("research artifact policy may not disable required artifacts")


@dataclass(frozen=True)
class BudgetResolution:
    """A discrete optimizer-step budget derived from one declared regime."""

    regime: ComparisonRegime
    steps: int
    effective_training_tokens: int
    requested_training_tokens: int | None
    requested_model_flops: float | None
    effective_model_flops: float | None
    tokens_per_parameter: float


def _positive_int(value: int | None, label: str) -> int:
    if value is None or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return int(value)


def resolve_training_budget(
    regime: ComparisonRegime | str,
    *,
    batch_tokens: int,
    parameter_count: int,
    algorithmic_flops_per_token: float | None = None,
    target_train_tokens: int | None = None,
    target_model_flops: float | None = None,
    tokens_per_parameter: float | None = None,
) -> BudgetResolution:
    """Resolve a budget without silently changing the scientific question.

    Controlled and fixed-compute budgets are strict upper bounds, so they use
    floor division. Scaling runs preserve the conventional nearest-step
    tokens/parameter target because it is a ratio rather than a hard cap.
    """

    regime = ComparisonRegime(regime)
    batch_tokens = _positive_int(batch_tokens, "batch_tokens")
    parameter_count = _positive_int(parameter_count, "parameter_count")
    requested_tokens: int | None = None
    requested_flops: float | None = None

    if regime is ComparisonRegime.CONTROLLED:
        requested_tokens = _positive_int(target_train_tokens, "target_train_tokens")
        if target_model_flops is not None or tokens_per_parameter is not None:
            raise ContractError(
                "controlled comparisons accept only target_train_tokens; do not mix budgets"
            )
        steps = requested_tokens // batch_tokens
    elif regime is ComparisonRegime.FIXED_COMPUTE:
        if target_model_flops is None or not math.isfinite(target_model_flops):
            raise ContractError("target_model_flops must be finite for fixed_compute")
        if target_model_flops <= 0:
            raise ContractError("target_model_flops must be positive")
        if algorithmic_flops_per_token is None or not math.isfinite(
            algorithmic_flops_per_token
        ):
            raise ContractError(
                "algorithmic_flops_per_token must be finite for fixed_compute"
            )
        if algorithmic_flops_per_token <= 0:
            raise ContractError("algorithmic_flops_per_token must be positive")
        if target_train_tokens is not None or tokens_per_parameter is not None:
            raise ContractError(
                "fixed_compute comparisons accept only target_model_flops; do not mix budgets"
            )
        requested_flops = float(target_model_flops)
        steps = math.floor(
            requested_flops / (float(algorithmic_flops_per_token) * batch_tokens)
        )
    else:
        if tokens_per_parameter is None or not math.isfinite(tokens_per_parameter):
            raise ContractError("tokens_per_parameter must be finite for scaling")
        if tokens_per_parameter <= 0:
            raise ContractError("tokens_per_parameter must be positive")
        if target_train_tokens is not None or target_model_flops is not None:
            raise ContractError(
                "scaling comparisons accept only tokens_per_parameter; do not mix budgets"
            )
        requested_tokens = round(float(tokens_per_parameter) * parameter_count)
        steps = round(requested_tokens / batch_tokens)

    if steps < 1:
        raise ContractError("the resolved budget is smaller than one optimizer step")
    effective_tokens = steps * batch_tokens
    effective_flops = (
        None
        if algorithmic_flops_per_token is None
        else float(algorithmic_flops_per_token) * effective_tokens
    )
    return BudgetResolution(
        regime=regime,
        steps=steps,
        effective_training_tokens=effective_tokens,
        requested_training_tokens=requested_tokens,
        requested_model_flops=requested_flops,
        effective_model_flops=effective_flops,
        tokens_per_parameter=effective_tokens / parameter_count,
    )


def assert_paired_controls(
    baseline: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    fields: tuple[str, ...] = (
        "seed",
        "dataset_manifest_sha256",
        "tokenizer_sha256",
        "data_order_id",
        "training_tokens",
        "sequence_length",
        "global_batch_tokens",
        "optimizer_contract_sha256",
    ),
) -> None:
    """Reject a purported controlled comparison when a frozen axis differs."""

    mismatches = {
        field: (baseline.get(field), variant.get(field))
        for field in fields
        if baseline.get(field) != variant.get(field)
    }
    if mismatches:
        details = ", ".join(
            f"{field}={left!r}/{right!r}" for field, (left, right) in mismatches.items()
        )
        raise ContractError(f"controlled comparison mismatch: {details}")
