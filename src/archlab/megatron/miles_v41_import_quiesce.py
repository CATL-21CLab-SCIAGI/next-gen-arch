"""Briefly quiesce selected checkpoint readers; always resume them in finally."""
import argparse
import json
import time
from pathlib import Path


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch", type=Path)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--pids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        raise ValueError("quiescence must be bounded by five minutes")
    started = json.loads(args.launch.read_text())["started_at_epoch"]
    processes = [psutil.Process(pid) for pid in args.pids]
    for process in processes:
        if (process.create_time() < started or not process.cmdline()
                or "MegatronTrainRayActor" not in process.cmdline()[0]):
            raise ValueError(f"process {process.pid} does not belong to this trainer launch")
    suspended = []
    try:
        for process in processes:
            process.suspend()
            suspended.append(process)
        print(json.dumps(dict(suspended=[p.pid for p in suspended], seconds=args.seconds)), flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            time.sleep(min(1., deadline-time.monotonic()))
    finally:
        for process in suspended:
            try:
                process.resume()
            except psutil.NoSuchProcess:
                pass
        print(json.dumps(dict(resumed=[p.pid for p in suspended])), flush=True)


if __name__ == "__main__":
    main()
