"""Frozen comparison contracts shared by execution backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

COMPARISON_SEEDS = (42, 43, 44)
COMPARISON_SEQUENCE_LENGTH = 2048
HISTORICAL_MICRO_BATCH_SIZE = 16
COMPARISON_GLOBAL_BATCH_SIZE = 192
COMPARISON_BATCH_TOKENS = COMPARISON_SEQUENCE_LENGTH * COMPARISON_GLOBAL_BATCH_SIZE
COMPARISON_EVAL_TOKENS = 3_932_160

# Public compatibility aliases for the original 10M campaign API.
TEN_M_SEEDS = COMPARISON_SEEDS
TEN_M_SEQUENCE_LENGTH = COMPARISON_SEQUENCE_LENGTH
TEN_M_MICRO_BATCH_SIZE = HISTORICAL_MICRO_BATCH_SIZE
TEN_M_GLOBAL_BATCH_SIZE = COMPARISON_GLOBAL_BATCH_SIZE
TEN_M_BATCH_TOKENS = COMPARISON_BATCH_TOKENS
TEN_M_EVAL_TOKENS = COMPARISON_EVAL_TOKENS


@dataclass(frozen=True)
class CampaignVariant:
    """One architecture arm from a frozen parameter-size campaign."""

    name: str
    parameter_count: int
    steps: int
    warmup_steps: int
    model_overrides: tuple[tuple[str, Any], ...]

    @property
    def training_tokens(self) -> int:
        return self.steps * COMPARISON_BATCH_TOKENS

    def model_kwargs(self) -> dict[str, Any]:
        return dict(self.model_overrides)


def _variant(
    name: str,
    parameter_count: int,
    steps: int,
    warmup_steps: int,
    **model_overrides: Any,
) -> CampaignVariant:
    return CampaignVariant(
        name=name,
        parameter_count=parameter_count,
        steps=steps,
        warmup_steps=warmup_steps,
        model_overrides=tuple(model_overrides.items()),
    )


TEN_M_VARIANTS = (
    _variant(
        "baseline",
        9_363_488,
        286,
        14,
        arch_family="sota_pool",
        sota_extra_lr=0.005,
        canon_kernel_size=4,
        bov_target_fraction=1.0 / 3.0,
        sota_variant="baseline",
    ),
    _variant(
        "engram",
        10_284_800,
        314,
        16,
        arch_family="engram",
        engram_ngram_orders=(2, 3),
        engram_kernel_size=4,
        engram_seed=0,
        engram_layers=(2,),
        engram_num_heads=2,
        engram_dim=28,
        engram_vocab_multiplier=1,
    ),
    _variant(
        "kda",
        9_372_437,
        286,
        14,
        arch_family="kimi_kda",
        kda_pattern="KKKG",
        kda_rope_policy="global_only",
        kda_variant="kimi_linear",
        kda_force_final_global=True,
    ),
    _variant(
        "dsa",
        9_391_248,
        287,
        14,
        arch_family="deepseek_dsa",
        dsa_top_k=32,
        dsa_query_chunk_size=128,
        dsa_backend="sdpa_masked",
        dsa_warmup_indexer_lr=0.001,
        dsa_sparse_indexer_lr=0.0000073,
        dsa_index_heads=2,
        dsa_index_head_dim=32,
        dsa_index_rope_dim=8,
    ),
    _variant(
        "attnres",
        9_364_709,
        286,
        14,
        arch_family="kimi_attnres",
        attn_res_block_size=2,
        attn_res_recompute=True,
        attn_res_variant="kimi_k3_block_attnres",
        attn_res_heads=1,
    ),
    _variant("mhc", 9_417_518, 287, 14, arch_family="mhc"),
    _variant(
        "gated-attention",
        9_365_448,
        286,
        14,
        arch_family="sota_pool",
        sota_extra_lr=0.005,
        canon_kernel_size=4,
        bov_target_fraction=1.0 / 3.0,
        sota_variant="gated_attention",
    ),
    _variant(
        "situ-glu",
        9_363_208,
        286,
        14,
        arch_family="frontier_pool",
        frontier_variant="kimi_situ_glu",
        relative_dim=8,
    ),
    _variant(
        "inkling-relative-attention",
        9_420_128,
        287,
        14,
        arch_family="frontier_pool",
        frontier_variant="inkling_relative_attention",
        relative_dim=8,
    ),
    _variant(
        "glm-mla",
        9_394_848,
        287,
        14,
        arch_family="frontier_pool",
        frontier_variant="glm_mla_muon_split",
        relative_dim=8,
    ),
    _variant(
        "xielu",
        9_363_498,
        286,
        14,
        arch_family="sota_pool",
        sota_extra_lr=0.005,
        canon_kernel_size=4,
        bov_target_fraction=1.0 / 3.0,
        sota_variant="xielu",
    ),
    _variant(
        "qwen-gdn",
        9_423_916,
        288,
        14,
        arch_family="frontier_pool",
        frontier_variant="qwen_gdn",
        relative_dim=8,
    ),
    _variant(
        "inkling-sconv-kv",
        9_365_728,
        286,
        14,
        arch_family="frontier_pool",
        frontier_variant="inkling_sconv_kv",
        relative_dim=8,
    ),
    _variant(
        "inkling-sconv-residual",
        9_365_728,
        286,
        14,
        arch_family="frontier_pool",
        frontier_variant="inkling_sconv_residual",
        relative_dim=8,
    ),
    _variant(
        "partial-rope-25",
        9_363_488,
        286,
        14,
        arch_family="frontier_pool",
        frontier_variant="partial_rope_25",
        relative_dim=8,
    ),
    _variant(
        "kimi-k3-kda-update",
        9_385_429,
        286,
        14,
        arch_family="kimi_kda",
        kda_variant="kimi_k3",
        kda_pattern="KKKG",
        kda_rope_policy="none",
        kda_force_final_global=True,
    ),
)

TEN_M_VARIANTS_BY_NAME = {variant.name: variant for variant in TEN_M_VARIANTS}

HUNDRED_M_VARIANTS = (
    _variant(
        "baseline",
        105_775_510,
        3_228,
        40,
        arch_family="sota_pool",
        sota_extra_lr=0.005,
        canon_kernel_size=4,
        bov_target_fraction=1.0 / 3.0,
        sota_variant="baseline",
    ),
)
HUNDRED_M_VARIANTS_BY_NAME = {variant.name: variant for variant in HUNDRED_M_VARIANTS}

CAMPAIGN_GEOMETRIES = {
    "10m": {"depth": 5, "aspect_ratio": 11, "head_dim": 8},
    "100m": {"depth": 10, "aspect_ratio": 38, "head_dim": 64},
}
CAMPAIGN_VARIANTS = {
    "10m": TEN_M_VARIANTS_BY_NAME,
    "100m": HUNDRED_M_VARIANTS_BY_NAME,
}

# Kept as an alias so downstream imports do not break while the generalized
# name is used by new scale-aware code.
TenMVariant = CampaignVariant


def get_ten_m_variant(name: str) -> TenMVariant:
    try:
        return TEN_M_VARIANTS_BY_NAME[name]
    except KeyError as error:
        choices = ", ".join(TEN_M_VARIANTS_BY_NAME)
        raise ValueError(f"unknown 10M variant {name!r}; choose one of: {choices}") from error


def get_campaign_variant(scale: str, name: str) -> CampaignVariant:
    try:
        variants = CAMPAIGN_VARIANTS[scale]
    except KeyError as error:
        choices = ", ".join(CAMPAIGN_VARIANTS)
        raise ValueError(f"unknown campaign scale {scale!r}; choose one of: {choices}") from error
    try:
        return variants[name]
    except KeyError as error:
        choices = ", ".join(variants)
        raise ValueError(
            f"unknown {scale} variant {name!r}; choose one of: {choices}"
        ) from error


def ten_m_model_config_kwargs(variant: TenMVariant) -> dict[str, Any]:
    """Return the exact shared geometry plus the arm-specific mechanism."""
    return {
        "depth": 5,
        "aspect_ratio": 11,
        "head_dim": 8,
        "max_seq_len": TEN_M_SEQUENCE_LENGTH,
        "vocab_size": 32_768,
        "window_pattern": "L",
        "fog_variant": "flash",
        "per_head_muon": True,
        **variant.model_kwargs(),
    }


def campaign_model_config_kwargs(scale: str, variant: CampaignVariant) -> dict[str, Any]:
    """Return exact shared geometry plus the arm-specific mechanism."""
    try:
        geometry = CAMPAIGN_GEOMETRIES[scale]
    except KeyError as error:
        choices = ", ".join(CAMPAIGN_GEOMETRIES)
        raise ValueError(f"unknown campaign scale {scale!r}; choose one of: {choices}") from error
    return {
        **geometry,
        "max_seq_len": COMPARISON_SEQUENCE_LENGTH,
        "vocab_size": 32_768,
        "window_pattern": "L",
        "fog_variant": "flash",
        "per_head_muon": True,
        **variant.model_kwargs(),
    }


def verify_ten_m_contract() -> dict[str, int]:
    names = [variant.name for variant in TEN_M_VARIANTS]
    if len(names) != 16 or len(set(names)) != 16:
        raise RuntimeError("the 10M campaign must contain 16 unique variants")
    if any(variant.training_tokens <= 0 for variant in TEN_M_VARIANTS):
        raise RuntimeError("the 10M campaign contains an invalid token budget")
    return {
        "variants": len(TEN_M_VARIANTS),
        "seeds": len(TEN_M_SEEDS),
        "runs": len(TEN_M_VARIANTS) * len(TEN_M_SEEDS),
    }
