"""Read and execute the frozen architecture-scaling experiment contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_MANIFEST = REPOSITORY_ROOT / "results" / "parameter-scale-100m-1b-v1-manifest.json"
_INSTALLED_MANIFEST = Path(__file__).parent / "data" / "parameter-scale-100m-1b-v1-manifest.json"
DEFAULT_MANIFEST = _SOURCE_MANIFEST if _SOURCE_MANIFEST.exists() else _INSTALLED_MANIFEST
EXPECTED_SIZES = {"100m", "300m", "1b"}
EXPECTED_SEEDS = {42, 43, 44}
EXPECTED_VARIANTS = {
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
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    size_id: str
    variant_id: str
    seed: int
    parameter_count: int
    steps: int
    tokens: int
    depth: int
    aspect_ratio: int
    head_dim: int
    warmup_steps: int
    save_every: int
    variant_args: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunSpec:
        fields = {
            "run_id",
            "size_id",
            "variant_id",
            "seed",
            "parameter_count",
            "steps",
            "tokens",
            "depth",
            "aspect_ratio",
            "head_dim",
            "warmup_steps",
            "save_every",
        }
        values = {field: payload[field] for field in fields}
        values["variant_args"] = tuple(payload["variant_args"])
        return cls(**values)

    def command(self, *, run_name: str = "dummy") -> list[str]:
        """Return the exact portable training command for this frozen run."""
        common = [
            "python",
            "-m",
            "archlab.speedrun.base_train",
            f"--run={run_name}",
            f"--model-tag={self.run_id}",
            f"--seed={self.seed}",
            f"--depth={self.depth}",
            f"--aspect-ratio={self.aspect_ratio}",
            f"--head-dim={self.head_dim}",
            "--max-seq-len=2048",
            "--window-pattern=L",
            "--per-head-muon",
            "--device-batch-size=16",
            "--total-batch-size=393216",
            f"--num-iterations={self.steps}",
            f"--warmup-steps={self.warmup_steps}",
            "--warmdown-ratio=0.65",
            "--final-lr-frac=0.05",
            "--embedding-lr=0.3",
            "--unembedding-lr=0.008",
            "--matrix-lr=0.02",
            "--scalar-lr=0.5",
            "--weight-decay=0.28",
            "--eval-every=250",
            "--eval-tokens=3932160",
            "--core-metric-every=-1",
            "--sample-every=-1",
            f"--save-every={self.save_every}",
            "--no-save-final-checkpoint",
            "--finite-check-every=1",
        ]
        return common + list(self.variant_args)


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("campaign") != "parameter-scale-100m-1b-v1":
        raise ValueError(f"Unexpected campaign in {path}")
    return manifest


def load_runs(path: Path = DEFAULT_MANIFEST) -> list[RunSpec]:
    return [RunSpec.from_dict(run) for run in load_manifest(path)["runs"]]


def find_run(size_id: str, variant_id: str, seed: int) -> RunSpec:
    matches = [
        run
        for run in load_runs()
        if (run.size_id, run.variant_id, run.seed) == (size_id, variant_id, seed)
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one run for size={size_id}, variant={variant_id}, seed={seed}; "
            f"found {len(matches)}"
        )
    return matches[0]


def verify_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, int]:
    manifest = load_manifest(path)
    runs = load_runs(path)
    unique_keys = {(run.size_id, run.variant_id, run.seed) for run in runs}
    sizes = {run.size_id for run in runs}
    variants = {run.variant_id for run in runs}
    seeds = {run.seed for run in runs}
    if sizes != EXPECTED_SIZES or variants != EXPECTED_VARIANTS or seeds != EXPECTED_SEEDS:
        raise ValueError("Manifest axes do not match the frozen campaign")
    expected = len(EXPECTED_SIZES) * len(EXPECTED_VARIANTS) * len(EXPECTED_SEEDS)
    if len(runs) != expected or len(unique_keys) != expected:
        raise ValueError(
            f"Manifest is not a complete Cartesian grid: {len(runs)} runs, "
            f"{len(unique_keys)} unique keys, {expected} expected"
        )
    if manifest["contract"]["sequence_length"] != 2048:
        raise ValueError("Unexpected sequence length")
    if manifest["contract"]["precision"] != "bfloat16":
        raise ValueError("Unexpected precision")
    if manifest["scope"]["total_runs"] != expected:
        raise ValueError("Manifest total_runs disagrees with the frozen grid")
    if len({run.run_id for run in runs}) != expected:
        raise ValueError("Run IDs are not unique")
    if any(
        run.parameter_count <= 0 or run.steps <= 0 or run.tokens <= 0 or not run.variant_args
        for run in runs
    ):
        raise ValueError("A run has invalid count or command metadata")
    return {
        "runs": len(runs),
        "sizes": len(sizes),
        "variants": len(variants),
        "seeds": len(seeds),
    }
