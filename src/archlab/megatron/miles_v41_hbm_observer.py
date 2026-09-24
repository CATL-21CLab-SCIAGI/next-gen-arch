"""Sample physical HBM for all colocated processes; never admit a run by itself."""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    import pynvml as nv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=int, default=14400)
    args = parser.parse_args()
    nv.nvmlInit()
    handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
    minima = [1.] * len(handles)
    start = time.time()
    last_write = 0
    samples = 0
    try:
        while time.time() - start < args.seconds:
            current = []
            for index, handle in enumerate(handles):
                memory = nv.nvmlDeviceGetMemoryInfo(handle)
                fraction = memory.free / memory.total
                minima[index] = min(minima[index], fraction)
                current.append(dict(free_bytes=memory.free, total_bytes=memory.total))
            samples += 1
            now = time.time()
            if now - last_write >= 5:
                temporary = args.output.with_suffix(".tmp")
                temporary.write_text(json.dumps(dict(pid=os.getpid(), started_epoch=start,
                    sampled_epoch=now, samples=samples, sample_interval_seconds=.1,
                    gpu_model="NVIDIA B300", minimum_free_hbm_fractions=minima,
                    current=current, establishes_admission=False)))
                temporary.replace(args.output)
                last_write = now
            time.sleep(.1)
    finally:
        nv.nvmlShutdown()


if __name__ == "__main__":
    main()
