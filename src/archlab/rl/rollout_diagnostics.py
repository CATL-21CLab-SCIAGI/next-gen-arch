"""Opt-in rank-local progress and bounded Python stacks for async sampling.

Observation uses only host metadata and Python's public faulthandler timer.
It neither calls CUDA nor changes sampling, policy snapshots or optimizer state.
The timer is refreshed by actor progress, so normal long responses do not count
as a stall simply because their total generation time exceeds the interval.
"""

from __future__ import annotations

import faulthandler
import math
import os
import sys
import threading
import time
from pathlib import Path

from archlab.artifacts import atomic_write_json


class RolloutDiagnostics:
    def __init__(self, output, *, rank, interval=120.0, max_dumps=4):
        if not math.isfinite(interval) or not 10 <= interval <= 3600 or not 1 <= max_dumps <= 16:
            raise ValueError("invalid bounded rollout diagnostic limits")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.rank, self.interval, self.max_dumps = rank, interval, max_dumps
        self.path = self.output / f"PROGRESS-rank{rank}.json"
        self.trace = (self.output / f"PYTHON_STACKS-rank{rank}.log").open("ab", buffering=0)
        self.lock = threading.Lock()
        self.state = dict(pid=os.getpid(), rank=rank, interval_seconds=interval, max_dumps=max_dumps,
                          sampling_changed=False, cuda_synchronization_added=False, components={})
        self.active = self.armed = False
        self.dumps = 0
        self.trace_bytes = self.trace.tell()
        self.failed = False

    @classmethod
    def from_environment(cls):
        output = os.environ.get("ARCHLAB_ROLLOUT_DIAGNOSTICS_DIR")
        if not output:
            return None
        try:
            return cls(output, rank=int(os.environ.get("RANK", "0")),
                       interval=float(os.environ.get("ARCHLAB_ROLLOUT_DIAGNOSTICS_INTERVAL", "120")),
                       max_dumps=int(os.environ.get("ARCHLAB_ROLLOUT_DIAGNOSTICS_MAX_DUMPS", "4")))
        except OSError as exc:
            print(f"rollout diagnostics unavailable: {exc!r}", file=sys.stderr)
            return None

    def _cancel(self):
        if self.armed:
            faulthandler.cancel_dump_traceback_later()
            self.armed = False
        size = os.fstat(self.trace.fileno()).st_size
        if size > self.trace_bytes:
            self.dumps += 1
        self.trace_bytes = size

    def _arm(self):
        self._cancel()
        if self.active and self.dumps < self.max_dumps:
            faulthandler.dump_traceback_later(self.interval, repeat=False, file=self.trace, exit=False)
            self.armed = True

    def mark(self, component, phase, **fields):
        if self.failed:
            return
        try:
            with self.lock:
                now = time.time()
                self.state["components"][component] = dict(phase=phase, time=now,
                                                           thread_id=threading.get_ident(), **fields)
                if component == "actor":
                    self._arm()
                self.state.update(time=now, stack_dumps=self.dumps)
                atomic_write_json(self.path, self.state, allow_nan=False)
        except (OSError, RuntimeError, ValueError) as exc:
            # A diagnostic filesystem failure cannot turn a healthy rollout
            # into a training failure or invalidate its sampled RNG trajectory.
            self.failed = True
            if self.armed:
                faulthandler.cancel_dump_traceback_later()
                self.armed = False
            print(f"rollout diagnostics disabled on rank {self.rank}: {exc!r}", file=sys.stderr)

    def begin(self, **fields):
        self.active = True
        self.mark("actor", "job_begin", **fields)

    def end(self):
        self.active = False
        self.mark("actor", "job_end")

    def close(self):
        self.active = False
        if not self.trace.closed:
            if self.armed:
                faulthandler.cancel_dump_traceback_later()
                self.armed = False
            self.trace.close()
