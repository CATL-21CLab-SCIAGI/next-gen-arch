"""Record completion of an exact-step scratch checkpoint catch-up."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from archlab.artifacts import atomic_write_json
from archlab.storage.checkpoint_retention import validate

atomic_json = partial(atomic_write_json, sort_keys=False, allow_nan=True, create_parents=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 12 * 3600
    while time.monotonic() < deadline:
        state = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "target_step": args.step,
            "target_supervised_tokens": args.tokens,
            "variants": {},
            "matched": False,
        }
        markers = {}
        for variant in ("normal", "simplicial"):
            run = args.root / f"production-{variant}-v5"
            if list(run.glob("*failure.json")):
                raise RuntimeError(f"{variant} training failed")
            rows = [
                json.loads(x)
                for x in (run / "train-metrics.jsonl").read_text().split("\n")[:-1]
                if x
            ]
            last = rows[-1]
            if last["step"] > args.step:
                raise RuntimeError(f"{variant} overshot target")
            checkpoint = run / "checkpoints" / f"step-{args.step:06d}"
            state["variants"][variant] = {
                "step": last["step"],
                "tokens": last["consumed_supervised_tokens"],
                "checkpoint": str(checkpoint),
                "complete": (checkpoint / "COMPLETE.json").exists(),
            }
            if (checkpoint / "COMPLETE.json").exists():
                marker = json.loads((checkpoint / "COMPLETE.json").read_text())
                if (
                    marker["cursor"]["step"] != args.step
                    or marker["cursor"]["supervised_tokens"] != args.tokens
                ):
                    raise RuntimeError("checkpoint target mismatch")
                markers[variant] = marker
        atomic_json(args.output / "STATUS.json", state)
        if len(markers) == 2:
            if markers["normal"]["cursor"] != markers["simplicial"]["cursor"]:
                raise RuntimeError("checkpoint cursors differ")
            state["verification"] = {
                v: validate(Path(state["variants"][v]["checkpoint"])) for v in markers
            }
            state["matched"] = True
            atomic_json(args.output / "MATCHED.json", state)
            atomic_json(args.output / "STATUS.json", state)
            print(json.dumps(state), flush=True)
            return
        time.sleep(30)
    raise TimeoutError("catch-up did not finish within 12 hours")


if __name__ == "__main__":
    main()
