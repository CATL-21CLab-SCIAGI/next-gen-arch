"""Architecture capability metadata across execution backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ArchitectureCapability:
    variant: str
    speedrun: bool
    megatron: bool
    note: str = ""


_SPEEDRUN_VARIANTS = (
    "attnres",
    "baseline",
    "dsa",
    "engram",
    "gated-attention",
    "glm-mla",
    "inkling-relative-attention",
    "inkling-sconv-kv",
    "inkling-sconv-residual",
    "kda",
    "kimi-k3-kda-update",
    "mhc",
    "partial-rope-25",
    "qwen-gdn",
    "situ-glu",
    "xielu",
)

CAPABILITIES = {
    variant: ArchitectureCapability(
        variant=variant,
        speedrun=True,
        megatron=variant == "baseline",
        note=(
            "upstream MCore baseline"
            if variant == "baseline"
            else "requires an explicit MCore adapter before scaling"
        ),
    )
    for variant in _SPEEDRUN_VARIANTS
}


def require_backend_support(variant: str, backend: str) -> None:
    capability = CAPABILITIES.get(variant)
    if capability is None:
        raise ValueError(f"unknown architecture variant: {variant}")
    if not getattr(capability, backend, False):
        raise ValueError(
            f"variant {variant!r} has no validated {backend} adapter; {capability.note}"
        )


def capability_rows() -> list[dict[str, object]]:
    return [asdict(CAPABILITIES[key]) for key in sorted(CAPABILITIES)]
