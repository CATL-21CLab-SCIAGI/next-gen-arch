"""Bounded, checksum-verified OSS publication of immutable NAS checkpoints."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


class CheckpointPublisher:
    """Overlap one immutable checkpoint upload with training, surface failures."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oss-checkpoint")
        self.pending = None

    def submit(self, publish, *args):
        # Bound disk pressure and never overwrite an in-flight checkpoint.
        self.wait()
        self.pending = self.executor.submit(publish, *args)

    def check(self):
        if self.pending is not None and self.pending.done():
            self.pending.result()

    def wait(self):
        if self.pending is not None:
            self.pending.result()
            self.pending = None

    def close(self):
        try:
            self.wait()
        finally:
            self.executor.shutdown(wait=True)
