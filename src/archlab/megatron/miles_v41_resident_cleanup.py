"""Retire only serving process trees belonging to an explicitly selected run."""
import argparse
import json
from pathlib import Path


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--variant", choices=("normal", "simplicial"), required=True)
    args = parser.parse_args()
    model_path = str(args.run_root.absolute() / f"model-{args.variant}")
    roots = []
    for process in psutil.process_iter():
        try:
            argv = process.cmdline()
            if ("sglang.launch_server" in argv and "--model-path" in argv
                    and argv[argv.index("--model-path") + 1] == model_path):
                roots.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    selected = {p.pid: p for root in roots for p in [*root.children(recursive=True), root]}
    for process in reversed(list(selected.values())):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(selected.values()), timeout=10)
    for process in alive:
        process.kill()
    _, alive = psutil.wait_procs(alive, timeout=10)
    zombies = [p.pid for p in alive if p.status() == psutil.STATUS_ZOMBIE]
    alive = [p for p in alive if p.status() != psutil.STATUS_ZOMBIE]
    print(json.dumps(dict(model_path=model_path, retired_pids=sorted(selected), zombie_pids=zombies,
                          remaining_pids=[p.pid for p in alive])), flush=True)
    if alive:
        raise RuntimeError("serving processes remain")


if __name__ == "__main__":
    main()
