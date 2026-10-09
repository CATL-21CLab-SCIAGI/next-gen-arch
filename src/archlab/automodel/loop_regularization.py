"""Preregister the Figure 5 repeated-data protocol transferred to V4.1 d128."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from archlab.artifacts import atomic_write_json

REFERENCE_COMMIT = "9139c396957a57b6de9a1e7efe7e35a4a863f1a6"
WEIGHT_DECAYS = (0.05, 0.2, 0.4, 0.8, 1.2, 1.6)
GRID = {
    4: (6,),
    6: (1, 2, 3, 4, 6),
    8: (1, 2, 3, 4, 6, 8, 12),
    10: (1, 2, 3, 4, 6, 8, 12),
    12: (1, 2, 3, 4, 6, 8, 12),
    14: (1, 2, 3, 4, 6, 8),
    16: (1, 2, 3),
    18: (1,),
}


def cells():
    result = [
        dict(
            reference_depth=depth,
            stored_layers=depth * 5 // 2,
            hidden_size=128 * ((depth + 7) // 8),
            recursions=k,
            weight_decay=wd,
            unique_targets=100_000_000,
            epochs=10,
            reference_commit=REFERENCE_COMMIT,
        )
        for depth, loops in GRID.items()
        for k in loops
        for wd in WEIGHT_DECAYS
    ]
    # Run the requested 20-layer d128 loop first. Then prioritize the main
    # fixed-WD curves; keep all weight-decay controls in the preregistered grid.
    return sorted(
        result,
        key=lambda c: (
            not (c["reference_depth"] == 8 and c["recursions"] == 2 and c["weight_decay"] == 0.8),
            c["weight_decay"] != 0.8,
            c["reference_depth"],
            c["recursions"],
            c["weight_decay"],
        ),
    )


def validate_cell(cell):
    if cell not in cells():
        raise ValueError("cell differs from the preregistered repeated-data grid")


def cell_name(cell):
    validate_cell(cell)
    wd = str(cell["weight_decay"]).replace(".", "p")
    return f"loop-d{cell['hidden_size']}-L{cell['stored_layers']}-K{cell['recursions']}-wd{wd}"


def write_plan(base_plan, *, source, commit, destination):
    base = json.loads(Path(base_plan).read_text())
    root = Path(base["root"])
    template = next(r for r in base["runs"] if r["variant"] == "normal" and r["width"] == 128)
    environment = copy.deepcopy(base["environment"])
    old_python = environment["PYTHONPATH"].split(":")
    environment["PYTHONPATH"] = ":".join([str(source / "src"), *old_python[1:]])
    runs = []
    for index, cell in enumerate(cells()):
        name = cell_name(cell)
        path = root / "sweep-cells" / f"{name}.json"
        atomic_write_json(path, cell)
        runs.append(
            {
                **template,
                "supplemental": True,
                "queue_index": index,
                "width": cell["hidden_size"],
                "source": str(source),
                "source_commit": commit,
                "environment": environment,
                "master_port": 29000 + index * 2,
                "qualification": str(root / f"qualification-{name}"),
                "output": str(root / name),
                "extra_args": [*template.get("extra_args", []), "--sweep-cell", str(path)],
                "training_token_budget": 1_000_000_000,
                "token_allowance": 10_000_000_000,
                "cell": cell,
            }
        )
    value = {
        "format": "archlab-v41-figure5-transfer-v2",
        "runs": runs,
        "reference_commit": REFERENCE_COMMIT,
        "cells_sha256": hashlib.sha256(json.dumps(cells(), sort_keys=True).encode()).hexdigest(),
        "planned_training_tokens": len(runs) * 1_000_000_000,
        "token_allowance_per_run": 10_000_000_000,
        "scope": "Figure 5 protocol transfer; V4.1 d128/20-layer anchor at reference d8; coupled width/depth controls rounded upward to multiples of 128 for exact 3.0 MoE ratio",
        "primary_compute_axis": "measured B300 training GPU-seconds, excluding evaluation and checkpoint IO",
    }
    atomic_write_json(destination, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = write_plan(
        args.base_plan, source=args.source, commit=args.commit, destination=args.output
    )
    print(json.dumps({k: v for k, v in value.items() if k != "runs"}))


if __name__ == "__main__":
    main()
