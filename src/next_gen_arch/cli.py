"""Console entry point for frozen results and portable backend launch plans."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from next_gen_arch.architectures import capability_rows
from next_gen_arch.backends import get_backend
from next_gen_arch.prompts import load_prompts
from next_gen_arch.registry import find_run, load_runs, verify_manifest
from next_gen_arch.spec import load_experiment


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "verify":
        from next_gen_arch.results import verify_metrics

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
