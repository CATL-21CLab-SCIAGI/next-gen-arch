"""Continue verified result offloading as new completed payloads appear."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from archlab.storage.model_migration import save
from archlab.storage.results_plan import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    status = args.database.parent / "RESULTS_OFFLOAD_WATCH_STATUS.json"
    if args.wait_for_pid:
        process = Path("/proc") / str(args.wait_for_pid)
        while process.exists():
            try:
                command = (process / "cmdline").read_bytes().split(b"\0")
            except FileNotFoundError:
                break
            if command and command != [b""] and b"archlab.storage.bulk_offload" not in command:
                raise RuntimeError("Initial transfer PID changed identity")
            save(
                status,
                {
                    "state": "waiting_for_initial_transfer",
                    "initial_pid": args.wait_for_pid,
                    "utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            time.sleep(30)
        receipt = json.loads(args.database.with_suffix(".json").read_text())
        if not receipt["complete"] or receipt["errors"]:
            raise RuntimeError("Initial transfer needs attention; no follow-on changes started")
    while True:
        existing = {
            row["source"]: row
            for row in [
                json.loads(line) for line in args.plan.read_text().split("\n") if line.strip()
            ]
        }
        candidates, summary = build(args.source, args.destination)
        added = 0
        for row in candidates:
            if row["source"] not in existing:
                existing[row["source"]] = row
                added += 1
        if added:
            temporary = args.plan.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(row) + "\n" for row in existing.values()))
            temporary.replace(args.plan)
            save(
                status,
                {
                    "state": "transferring_new_payloads",
                    "new_files": added,
                    "utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "archlab.storage.bulk_offload",
                    "--plan",
                    str(args.plan),
                    "--database",
                    str(args.database),
                    "--workers",
                    str(args.workers),
                ],
                check=True,
            )
        save(
            status,
            {
                "state": "caught_up",
                "new_files": added,
                "scan": summary,
                "utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        time.sleep(120)


if __name__ == "__main__":
    main()
