"""Command-line input validation for the Megatron comparison entrypoint."""

import argparse
from pathlib import Path

from archlab.campaigns import CAMPAIGN_VARIANTS, COMPARISON_SEEDS, FINEWEB_CAMPAIGN_VARIANTS
from archlab.contracts import ComparisonRegime
from archlab.megatron.profiles import MEGATRON_BACKEND_PROFILES
from archlab.optimizers.recipes import OPTIMIZATION_RECIPES

DATASETS = ("climbmix", "fineweb10b", "fineweb100b")


def parse_args(argv=None, *, description=None):
    parser = argparse.ArgumentParser(description=description)
    scales = tuple(dict.fromkeys((*CAMPAIGN_VARIANTS, *FINEWEB_CAMPAIGN_VARIANTS)))
    parser.add_argument("--dataset", choices=DATASETS, default="climbmix")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--scale", choices=scales, default="10m")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=COMPARISON_SEEDS)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument(
        "--comparison-regime",
        choices=tuple(item.value for item in ComparisonRegime),
        help="controlled=fixed tokens, fixed_compute=fixed model FLOPs, scaling=fixed tokens/parameter",
    )
    parser.add_argument("--target-train-tokens", type=int)
    parser.add_argument("--target-model-flops", type=float)
    parser.add_argument("--tokens-per-parameter", type=float)
    parser.add_argument(
        "--artifact-policy",
        choices=("metrics_only", "research"),
        default="metrics_only",
        help="research requires content provenance, initialization hash, and final checkpoint",
    )
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument(
        "--data-verification",
        choices=("metadata", "full"),
        default="metadata",
    )
    parser.add_argument(
        "--initialization-hash",
        choices=("none", "shared", "full"),
        default="none",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        help="FineWeb rank-local micro batch; Megatron accumulates to the fixed global batch",
    )
    parser.add_argument(
        "--exact-global-batch-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="preserve the 192-sequence canonical stream over non-divisible DP worlds",
    )
    parser.add_argument("--metrics-every", type=int, default=10)
    parser.add_argument(
        "--throughput-warmup-steps",
        type=int,
        default=10,
        help="optimizer-step intervals excluded from steady-state throughput",
    )
    parser.add_argument(
        "--throughput-measurement-steps",
        type=int,
        default=0,
        help="steady-state intervals to measure (0 uses every post-warmup interval)",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="load model/optimizer/RNG/scheduler state and derive the data cursor from iteration",
    )
    parser.add_argument(
        "--backend-profile",
        choices=tuple(MEGATRON_BACKEND_PROFILES),
        default="legacy",
        help="explicit Megatron wrapper optimization profile",
    )
    parser.add_argument(
        "--optimization-recipe",
        choices=tuple(OPTIMIZATION_RECIPES),
        default="baseline",
        help="isolated speedrun/Marin optimization recipe",
    )
    args = parser.parse_args(argv)
    if args.probe_steps < 0:
        parser.error("--probe-steps must be non-negative")
    if args.metrics_every < 1:
        parser.error("--metrics-every must be positive")
    if args.throughput_warmup_steps < 0 or args.throughput_measurement_steps < 0:
        parser.error("throughput step counts must be non-negative")
    if args.micro_batch_size is not None and args.micro_batch_size < 1:
        parser.error("--micro-batch-size must be positive")
    if args.checkpoint_dir is not None and (args.save_interval is None or args.save_interval < 1):
        parser.error("--checkpoint-dir requires a positive --save-interval")
    if args.checkpoint_dir is None and args.save_interval is not None:
        parser.error("--save-interval requires --checkpoint-dir")
    if args.resume and args.checkpoint_dir is None:
        parser.error("--resume requires --checkpoint-dir")
    if args.artifact_policy == "research":
        if args.checkpoint_dir is None:
            parser.error("research artifact policy requires --checkpoint-dir")
        if args.data_manifest is None:
            parser.error("research artifact policy requires --data-manifest")
        if args.initialization_hash == "none":
            parser.error("research artifact policy requires --initialization-hash=shared or full")
    return parser, args
