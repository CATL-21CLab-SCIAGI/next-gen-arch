"""Aggregate the frozen 10M campaign and compare Megatron with speedrun."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from next_gen_arch.training.campaigns import TEN_M_SEEDS, TEN_M_VARIANTS_BY_NAME

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_REFERENCE = REPOSITORY_ROOT / "results" / "speedrun-10m-reference.csv"
DEFAULT_REFERENCE = (
    _SOURCE_REFERENCE
    if _SOURCE_REFERENCE.is_file()
    else Path(str(files("next_gen_arch").joinpath("data/speedrun-10m-reference.csv")))
)


@dataclass(frozen=True)
class RunResult:
    backend: str
    variant: str
    seed: int
    parameter_count: int
    training_steps: int
    training_tokens: int
    final_bpb: float
    tokens_per_second: float
    wall_seconds: float

    @property
    def key(self) -> tuple[str, str, int]:
        return self.backend, self.variant, self.seed


@dataclass(frozen=True)
class VariantSummary:
    backend: str
    variant: str
    seeds: int
    mean_bpb: float
    paired_delta_bpb: float
    mean_tokens_per_second: float
    normalized_throughput: float
    mean_wall_seconds: float


def _parse_row(row: dict[str, str]) -> RunResult:
    return RunResult(
        backend=row["backend"],
        variant=row["variant"],
        seed=int(row["seed"]),
        parameter_count=int(row["parameter_count"]),
        training_steps=int(row["training_steps"]),
        training_tokens=int(row["training_tokens"]),
        final_bpb=float(row["final_bpb"]),
        tokens_per_second=float(row["tokens_per_second"]),
        wall_seconds=float(row["wall_seconds"]),
    )


def load_speedrun_reference(path: Path) -> list[RunResult]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [_parse_row(row) for row in csv.DictReader(handle)]


def load_megatron_results(root: Path) -> list[RunResult]:
    results: list[RunResult] = []
    for path in sorted(root.rglob("result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("backend") != "megatron" or payload.get("mode") != "full":
            continue
        variant_field = payload["variant"]
        variant = variant_field["name"] if isinstance(variant_field, dict) else variant_field
        results.append(
            RunResult(
                backend="megatron",
                variant=variant,
                seed=int(payload["seed"]),
                parameter_count=int(payload["parameter_count"]),
                training_steps=int(payload["training_steps"]),
                training_tokens=int(payload["training_tokens"]),
                final_bpb=float(payload["final_bpb"]),
                tokens_per_second=float(payload["tokens_per_second"]),
                wall_seconds=float(payload["wall_seconds"]),
            )
        )
    return results


def validate_results(rows: Iterable[RunResult], *, allow_partial: bool) -> list[RunResult]:
    rows = list(rows)
    keys = [row.key for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("campaign results contain duplicate backend/variant/seed rows")
    expected_variants = set(TEN_M_VARIANTS_BY_NAME)
    expected_seeds = set(TEN_M_SEEDS)
    for row in rows:
        if row.variant not in expected_variants or row.seed not in expected_seeds:
            raise ValueError(f"unexpected campaign row: {row.key}")
        contract = TEN_M_VARIANTS_BY_NAME[row.variant]
        expected = (contract.parameter_count, contract.steps, contract.training_tokens)
        actual = (row.parameter_count, row.training_steps, row.training_tokens)
        if actual != expected:
            raise ValueError(f"contract drift for {row.key}: {actual} != {expected}")
        if not math.isfinite(row.final_bpb) or row.final_bpb <= 0:
            raise ValueError(f"invalid BPB for {row.key}: {row.final_bpb}")
        if not math.isfinite(row.tokens_per_second) or row.tokens_per_second <= 0:
            raise ValueError(f"invalid throughput for {row.key}: {row.tokens_per_second}")
    if not allow_partial:
        for backend in {row.backend for row in rows}:
            backend_keys = {(row.variant, row.seed) for row in rows if row.backend == backend}
            expected_keys = {
                (variant, seed) for variant in expected_variants for seed in expected_seeds
            }
            if backend_keys != expected_keys:
                missing = sorted(expected_keys - backend_keys)
                raise ValueError(f"{backend} is missing {len(missing)} frozen runs: {missing[:5]}")
    return rows


def summarize(rows: Iterable[RunResult]) -> list[VariantSummary]:
    grouped: dict[tuple[str, str], list[RunResult]] = defaultdict(list)
    baselines: dict[tuple[str, int], RunResult] = {}
    for row in rows:
        grouped[(row.backend, row.variant)].append(row)
        if row.variant == "baseline":
            baselines[(row.backend, row.seed)] = row

    summaries = []
    for (backend, variant), variant_rows in sorted(grouped.items()):
        paired = [
            (row, baselines[(backend, row.seed)])
            for row in variant_rows
            if (backend, row.seed) in baselines
        ]
        if not paired:
            continue
        summaries.append(
            VariantSummary(
                backend=backend,
                variant=variant,
                seeds=len(paired),
                mean_bpb=statistics.fmean(row.final_bpb for row, _baseline in paired),
                paired_delta_bpb=statistics.fmean(
                    row.final_bpb - baseline.final_bpb for row, baseline in paired
                ),
                mean_tokens_per_second=statistics.fmean(
                    row.tokens_per_second for row, _baseline in paired
                ),
                normalized_throughput=statistics.fmean(
                    row.tokens_per_second / baseline.tokens_per_second for row, baseline in paired
                ),
                mean_wall_seconds=statistics.fmean(row.wall_seconds for row, _baseline in paired),
            )
        )
    return summaries


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else float("nan")


def cross_backend_metrics(summaries: Iterable[VariantSummary]) -> dict[str, Any]:
    indexed = {(row.backend, row.variant): row for row in summaries}
    variants = sorted(
        variant
        for variant in TEN_M_VARIANTS_BY_NAME
        if variant != "baseline"
        and ("speedrun", variant) in indexed
        and ("megatron", variant) in indexed
    )
    speedrun_deltas = [indexed[("speedrun", variant)].paired_delta_bpb for variant in variants]
    megatron_deltas = [indexed[("megatron", variant)].paired_delta_bpb for variant in variants]
    gaps = [
        megatron - speedrun
        for megatron, speedrun in zip(megatron_deltas, speedrun_deltas, strict=True)
    ]
    return {
        "variants": variants,
        "variant_count": len(variants),
        "delta_pearson": _pearson(speedrun_deltas, megatron_deltas),
        "mean_absolute_delta_gap": statistics.fmean(abs(gap) for gap in gaps) if gaps else None,
        "mean_signed_delta_gap": statistics.fmean(gaps) if gaps else None,
    }


def _write_csv(path: Path, summaries: list[VariantSummary]) -> None:
    fieldnames = list(VariantSummary.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in summaries)


def _markdown(summaries: list[VariantSummary], cross_backend: dict[str, Any]) -> str:
    correlation = cross_backend["delta_pearson"]
    correlation_text = f"{correlation:.6f}" if math.isfinite(correlation) else "n/a"
    mean_gap = cross_backend["mean_absolute_delta_gap"]
    mean_gap_text = f"{mean_gap:.6f} BPB" if mean_gap is not None else "n/a"
    lines = [
        "# 10M backend comparison",
        "",
        "All deltas and throughput ratios are paired to the same backend and seed baseline.",
        "",
        "| Backend | Variant | Seeds | Mean BPB | Paired Δ BPB | Throughput | tok/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda item: (item.backend, item.mean_bpb)):
        lines.append(
            f"| {row.backend} | {row.variant} | {row.seeds} | {row.mean_bpb:.6f} | "
            f"{row.paired_delta_bpb:+.6f} | {row.normalized_throughput:.2f}× | "
            f"{row.mean_tokens_per_second:,.0f} |"
        )
    lines.extend(
        (
            "",
            "## Cross-backend agreement",
            "",
            f"- Compared variants: {cross_backend['variant_count']}",
            f"- Variant-delta Pearson correlation: {correlation_text}",
            f"- Mean absolute paired-delta gap: {mean_gap_text}",
            "",
        )
    )
    return "\n".join(lines)


def write_comparison(
    output_dir: Path,
    rows: list[RunResult],
    summaries: list[VariantSummary],
    cross_backend: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "comparison.csv", summaries)
    (output_dir / "comparison.json").write_text(
        json.dumps(
            {
                "runs": [asdict(row) for row in rows],
                "summary": [asdict(row) for row in summaries],
                "cross_backend": cross_backend,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "comparison.md").write_text(_markdown(summaries, cross_backend), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    rows = validate_results(
        [
            *load_speedrun_reference(args.reference),
            *load_megatron_results(args.megatron_root),
        ],
        allow_partial=args.allow_partial,
    )
    summaries = summarize(rows)
    cross_backend = cross_backend_metrics(summaries)
    write_comparison(args.output_dir, rows, summaries, cross_backend)
    print(_markdown(summaries, cross_backend), end="")


if __name__ == "__main__":
    main()
