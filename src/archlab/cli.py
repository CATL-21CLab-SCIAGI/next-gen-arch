"""Console entry point for frozen results and portable backend launch plans."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from archlab.capabilities import capability_rows
from archlab.launch import get_backend
from archlab.prompts import load_prompts
from archlab.registry import find_run, load_runs, verify_manifest
from archlab.spec import load_experiment


def _path_overrides(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key or not path:
            raise argparse.ArgumentTypeError(f"expected KEY=PATH, got {value!r}")
        if key in result:
            raise argparse.ArgumentTypeError(f"duplicate path override: {key}")
        result[key] = path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="next-gen-arch")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("list", help="list frozen campaign axes")
    subparsers.add_parser("verify", help="validate frozen artifacts")
    subparsers.add_parser("backends", help="show architecture/backend support")

    for action in ("show", "command"):
        command = subparsers.add_parser(action)
        command.add_argument("--size", required=True, choices=("100m", "300m", "1b"))
        command.add_argument("--variant", required=True)
        command.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
        if action == "command":
            command.add_argument("--run-name", default="dummy")

    render = subparsers.add_parser("render", help="render a portable experiment launch plan")
    render.add_argument("--config", required=True, type=Path)
    render.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="override a portable runtime path without editing YAML",
    )
    render.add_argument("--json", action="store_true", help="emit the complete launch plan")

    doctor = subparsers.add_parser("doctor", help="validate one execution backend")
    doctor.add_argument("--backend", required=True, choices=("speedrun", "megatron"))

    prompts = subparsers.add_parser("prompts", help="inspect a portable prompt set")
    prompts.add_argument("--file", type=Path)

    manifest = subparsers.add_parser(
        "data-manifest", help="create or verify a content-addressed dataset manifest"
    )
    manifest_actions = manifest.add_subparsers(dest="manifest_action", required=True)
    create = manifest_actions.add_parser("create")
    create.add_argument("--root", required=True, type=Path)
    create.add_argument("--dataset", required=True)
    create.add_argument("--revision", required=True)
    create.add_argument("--pattern", action="append", default=[])
    create.add_argument("--output", required=True, type=Path)
    check = manifest_actions.add_parser("verify")
    check.add_argument("--root", required=True, type=Path)
    check.add_argument("--manifest", required=True, type=Path)
    check.add_argument("--mode", choices=("metadata", "full"), default="full")

    pair = subparsers.add_parser(
        "pair-check", help="verify the frozen axes of a controlled baseline/variant pair"
    )
    pair.add_argument("--baseline", required=True, type=Path)
    pair.add_argument("--variant", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "verify":
        from archlab.results import verify_metrics

        payload = {"manifest": verify_manifest(), "metrics": verify_metrics()}
        print(json.dumps(payload, indent=2))
        return 0
    if args.action == "backends":
        print(json.dumps(capability_rows(), indent=2))
        return 0
    if args.action == "doctor":
        backend = get_backend(args.backend)
        payload = backend.doctor() if hasattr(backend, "doctor") else {"status": "ready"}
        print(json.dumps(payload, indent=2))
        return 0
    if args.action == "prompts":
        print(json.dumps([prompt.__dict__ for prompt in load_prompts(args.file)], indent=2))
        return 0
    if args.action == "data-manifest":
        from archlab.provenance import (
            create_dataset_manifest,
            verify_dataset_manifest,
            write_dataset_manifest,
        )

        if args.manifest_action == "create":
            manifest = create_dataset_manifest(
                args.root,
                dataset=args.dataset,
                revision=args.revision,
                patterns=tuple(args.pattern or ("**/*",)),
            )
            write_dataset_manifest(args.output, manifest)
            print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
            return 0
        payload = verify_dataset_manifest(args.root, args.manifest, mode=args.mode)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.action == "pair-check":
        from archlab.contracts import assert_paired_controls

        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        variant = json.loads(args.variant.read_text(encoding="utf-8"))
        assert_paired_controls(baseline, variant)
        print(
            json.dumps(
                {"status": "matched", "baseline": str(args.baseline), "variant": str(args.variant)},
                indent=2,
            )
        )
        return 0
    if args.action == "render":
        try:
            overrides = _path_overrides(args.path)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        spec = load_experiment(args.config)
        plan = get_backend(spec.backend).render(spec, path_overrides=overrides)
        print(plan.json() if args.json else plan.shell(), end="" if args.json else "\n")
        return 0

    runs = load_runs()
    if args.action == "list":
        payload = {
            "sizes": sorted({run.size_id for run in runs}),
            "variants": sorted({run.variant_id for run in runs}),
            "seeds": sorted({run.seed for run in runs}),
        }
        print(json.dumps(payload, indent=2))
        return 0
    run = find_run(args.size, args.variant, args.seed)
    if args.action == "show":
        print(json.dumps(run.__dict__, indent=2))
    else:
        print(shlex.join(run.command(run_name=args.run_name)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
