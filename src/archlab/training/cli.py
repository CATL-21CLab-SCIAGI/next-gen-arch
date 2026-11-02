"""Resolve, qualify, or launch one package-backed experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock
from omegaconf import OmegaConf

from archlab.artifacts import atomic_write_json
from archlab.training.config import compose_verl, read_experiment


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("resolve", "qualify", "train"))
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--verl", type=Path, default=Path("third_party/verl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_intermixed_args(argv)
    experiment = read_experiment(args.recipe, args.overrides, overlays=args.overlay)
    config = compose_verl(experiment, args.verl)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "resolve":
        with FileLock(args.output / ".qualification.lock", timeout=0):
            if (args.output / "qualification.json").exists():
                raise FileExistsError("admission records are immutable; use a new output directory")
            OmegaConf.save(config, args.output / "resolved.yaml", resolve=True)
            atomic_write_json(args.output / "experiment.json", asdict(experiment))
        return 0
    from archlab.training.qualification import qualify

    if args.action == "qualify" or not (args.output / "qualification.json").exists():
        result = qualify(experiment, config, args.verl, args.output)
        if not result["passed"]:
            print("Qualification blocked; see", args.output / "qualification.json")
            return 1
    if args.action == "train":
        from archlab.training.runtime import run

        run(config, qualification=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
