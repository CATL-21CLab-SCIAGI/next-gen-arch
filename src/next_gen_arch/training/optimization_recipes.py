"""Auditable speedrun recipes layered over the frozen 10M comparison contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OptimizationRecipe:
    """One isolated optimization hypothesis and its trainer/model controls."""

    name: str
    summary: str
    sources: tuple[str, ...]
    model_overrides: dict[str, Any] = field(default_factory=dict)
    matrix_optimizer: str = "normuon"
    matrix_lr: float = 0.02
    unembedding_lr: float = 0.008
    embedding_lr: float = 0.3
    scalar_lr: float = 0.5
    gradient_clip: float = 0.0
    adam_update_every: int = 1
    cautious_adam_weight_decay: bool = False


MARIN_DIGEST = (
    "https://github.com/marin-community/marin/blob/main/docs/reports/agent-moe-experiments.md"
)
MODDED = "https://github.com/KellerJordan/modded-nanogpt"
MUONEQ = "https://github.com/marin-community/marin/issues/6066"


def _recipe(name: str, summary: str, *sources: str, **controls: Any) -> OptimizationRecipe:
    return OptimizationRecipe(name=name, summary=summary, sources=tuple(sources), **controls)


OPTIMIZATION_RECIPES = {
    recipe.name: recipe
    for recipe in (
        _recipe("baseline", "Frozen per-head NorMuon comparison contract.", MODDED),
        _recipe(
            "full-matrix-muon",
            "Ablate per-head optimizer partitioning with one full Q/K/V matrix per layer.",
            MODDED,
            model_overrides={"per_head_muon": False},
        ),
        _recipe(
            "full-matrix-clip01",
            "Full-matrix NorMuon combined with the promoted 0.1 gradient clip.",
            MODDED,
            MARIN_DIGEST,
            model_overrides={"per_head_muon": False},
            gradient_clip=0.1,
        ),
        _recipe(
            "partial-rope",
            "Rotate half of every attention head and leave half stationary.",
            MARIN_DIGEST,
            MODDED,
            model_overrides={"rope_fraction": 0.5},
        ),
        _recipe(
            "partial-rope-25",
            "Rotate one quarter of every attention head.",
            MARIN_DIGEST,
            model_overrides={"rope_fraction": 0.25},
        ),
        _recipe(
            "pko",
            "Partial key offset on every layer over the stationary half-head dimensions.",
            MARIN_DIGEST,
            MODDED,
            model_overrides={"rope_fraction": 0.5, "partial_key_offset": "all"},
        ),
        _recipe(
            "pko-last",
            "Partial key offset only in the final full-context layer.",
            MARIN_DIGEST,
            model_overrides={"rope_fraction": 0.5, "partial_key_offset": "last"},
        ),
        _recipe(
            "embed-std1",
            "Unit-standard-deviation token embedding initialization.",
            MARIN_DIGEST,
            model_overrides={"embedding_init_std": 1.0},
        ),
        _recipe(
            "qk-gain",
            "Learnable per-head Q/K gain initialized to the frozen 1.2 scale.",
            MARIN_DIGEST,
            model_overrides={"learnable_qk_gain": True},
        ),
        _recipe(
            "cached-attention",
            "Reuse one attention input across the final three layers.",
            MARIN_DIGEST,
            model_overrides={"cached_attention_layers": 3},
        ),
        _recipe(
            "midpoint-kv",
            "Reuse the midpoint residual as the K/V source in upper layers.",
            MARIN_DIGEST,
            model_overrides={"reuse_midpoint_kv": True},
        ),
        _recipe(
            "bf16-loss",
            "Keep logits and cross entropy in BF16 to test the lower-precision loss path.",
            MODDED,
            model_overrides={"loss_fp32": False},
        ),
        _recipe(
            "asymmetric-logits",
            "Use the current Modded asymmetric sigmoid logit transform.",
            MODDED,
            model_overrides={"logit_transform": "asymmetric"},
        ),
        _recipe(
            "z-loss-1e-4",
            "Marin's final-logit stabilization loss at weight 1e-4.",
            MARIN_DIGEST,
            model_overrides={"z_loss_weight": 1e-4},
        ),
        _recipe(
            "z-loss-5e-6",
            "The smaller DCLM/Marin final-logit z-loss at weight 5e-6.",
            MARIN_DIGEST,
            model_overrides={"z_loss_weight": 5e-6},
        ),
        _recipe(
            "z-loss-1e-6",
            "Lower-bound final-logit z-loss refinement at weight 1e-6.",
            MARIN_DIGEST,
            model_overrides={"z_loss_weight": 1e-6},
        ),
        _recipe(
            "z-loss-2e-6",
            "Final-logit z-loss refinement at weight 2e-6.",
            MARIN_DIGEST,
            model_overrides={"z_loss_weight": 2e-6},
        ),
        _recipe(
            "z-loss-1e-5",
            "Upper-neighborhood final-logit z-loss refinement at weight 1e-5.",
            MARIN_DIGEST,
            model_overrides={"z_loss_weight": 1e-5},
        ),
        _recipe(
            "z-loss-2e-5",
            "Upper-bound final-logit z-loss refinement at weight 2e-5.",
            MARIN_DIGEST,
            model_overrides={"z_loss_weight": 2e-5},
        ),
        _recipe(
            "z-loss-5e-6-clip005",
            "Final-logit z-loss 5e-6 combined with gradient clipping at 0.05.",
            MARIN_DIGEST,
            model_overrides={"z_loss_weight": 5e-6},
            gradient_clip=0.05,
        ),
        _recipe(
            "z-loss-5e-6-clip01",
            "Final-logit z-loss 5e-6 combined with gradient clipping at 0.1.",
            MARIN_DIGEST,
            model_overrides={"z_loss_weight": 5e-6},
            gradient_clip=0.1,
        ),
        _recipe(
            "full-matrix-z-loss-5e-6-clip01",
            "Full-matrix NorMuon with final-logit z-loss 5e-6 and clipping at 0.1.",
            MODDED,
            MARIN_DIGEST,
            model_overrides={"per_head_muon": False, "z_loss_weight": 5e-6},
            gradient_clip=0.1,
        ),
        _recipe(
            "muonh",
            "NorMuon direction with a Frobenius-hyperball norm-preserving step.",
            MARIN_DIGEST,
            MODDED,
            model_overrides={"matrix_init_recipe": "hyperball"},
            matrix_optimizer="muonh",
        ),
        _recipe(
            "muonh-lr01",
            "MuonH learning-rate retune at 0.01.",
            MARIN_DIGEST,
            MODDED,
            model_overrides={"matrix_init_recipe": "hyperball"},
            matrix_optimizer="muonh",
            matrix_lr=0.01,
        ),
        _recipe(
            "muonh-lr005",
            "MuonH learning-rate retune at 0.005.",
            MARIN_DIGEST,
            MODDED,
            model_overrides={"matrix_init_recipe": "hyperball"},
            matrix_optimizer="muonh",
            matrix_lr=0.005,
        ),
        _recipe(
            "muonh-lr03",
            "MuonH learning-rate retune at 0.03.",
            MARIN_DIGEST,
            MODDED,
            model_overrides={"matrix_init_recipe": "hyperball"},
            matrix_optimizer="muonh",
            matrix_lr=0.03,
        ),
        _recipe(
            "muonh-lr05",
            "MuonH learning-rate retune at 0.05.",
            MARIN_DIGEST,
            MODDED,
            model_overrides={"matrix_init_recipe": "hyperball"},
            matrix_optimizer="muonh",
            matrix_lr=0.05,
        ),
        _recipe(
            "muonh-tuned-aux",
            "MuonH with the Modded optimization-track auxiliary LR proportions.",
            MARIN_DIGEST,
            MODDED,
            model_overrides={"matrix_init_recipe": "hyperball"},
            matrix_optimizer="muonh",
            matrix_lr=0.018,
            unembedding_lr=0.00173,
            embedding_lr=0.246,
            scalar_lr=0.0195,
        ),
        _recipe(
            "muoneqh-half",
            "MuonH with two-sided row/column equilibration exponent -1/2.",
            MUONEQ,
            model_overrides={"matrix_init_recipe": "hyperball"},
            matrix_optimizer="muoneqh-half",
        ),
        _recipe(
            "muoneqh-quarter",
            "MuonH with gentler two-sided equilibration exponent -1/4.",
            MUONEQ,
            model_overrides={"matrix_init_recipe": "hyperball"},
            matrix_optimizer="muoneqh-quarter",
        ),
        _recipe(
            "adamh",
            "Adam-preconditioned Frobenius-hyperball updates for hidden matrices.",
            MARIN_DIGEST,
            MODDED,
            model_overrides={"matrix_init_recipe": "hyperball"},
            matrix_optimizer="adamh",
        ),
        _recipe(
            "grad-clip-01",
            "Global gradient clipping at 0.1.",
            MARIN_DIGEST,
            gradient_clip=0.1,
        ),
        _recipe(
            "grad-clip-005",
            "Refine the promoted clipping range at 0.05.",
            MARIN_DIGEST,
            gradient_clip=0.05,
        ),
        _recipe(
            "grad-clip-015",
            "Refine the promoted clipping range at 0.15.",
            MARIN_DIGEST,
            gradient_clip=0.15,
        ),
        _recipe(
            "grad-clip-02",
            "Refine the promoted clipping range at 0.2.",
            MARIN_DIGEST,
            gradient_clip=0.2,
        ),
        _recipe(
            "grad-clip-03",
            "Global gradient clipping at 0.3.",
            MARIN_DIGEST,
            gradient_clip=0.3,
        ),
        _recipe(
            "adam-every-2",
            "Accumulate auxiliary Adam gradients and update those parameters every second step.",
            MODDED,
            adam_update_every=2,
        ),
        _recipe(
            "cautious-adam-wd",
            "Apply auxiliary Adam weight decay only where update and parameter agree in sign.",
            MODDED,
            cautious_adam_weight_decay=True,
        ),
        _recipe(
            "marin-compound",
            "Portable portion of Marin's combined-best recipe.",
            MARIN_DIGEST,
            model_overrides={
                "rope_fraction": 0.5,
                "partial_key_offset": "all",
                "cached_attention_layers": 3,
                "embedding_init_std": 1.0,
            },
        ),
    )
}


def get_optimization_recipe(name: str) -> OptimizationRecipe:
    try:
        return OPTIMIZATION_RECIPES[name]
    except KeyError as error:
        choices = ", ".join(OPTIMIZATION_RECIPES)
        raise ValueError(
            f"unknown optimization recipe {name!r}; choose one of: {choices}"
        ) from error
