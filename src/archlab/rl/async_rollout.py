"""One-batch asynchronous rollout prefetch with explicit policy/RNG provenance."""

from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass


class BatchLookahead:
    """Peek without changing Trainer's batch order, epoch length, or resume skip."""

    def __init__(self):
        self.iterator = None
        self.buffer = None

    def collect(self, collect, iterator, num_batches, device):
        if iterator is not self.iterator:
            self.iterator, self.buffer = iterator, None
        prefix = [] if self.buffer is None else [self.buffer]
        self.buffer = None
        result = collect(itertools.chain(prefix, iterator), num_batches, device)
        self.buffer = next(iterator, None)
        return result


@dataclass
class PolicySnapshot:
    version: int
    weights: object
    ready: object = None


class AsyncRolloutQueue:
    """Keep one prefetched batch and checkpoint it rather than dropping data."""

    def __init__(self, generate, generator, snapshot, stop_requested, max_policy_lag=1, *, diagnostics=None):
        self.generate = generate
        self.generator = generator
        self.snapshot = snapshot
        self.closing = threading.Event()
        self.stop_requested = lambda: self.closing.is_set() or stop_requested()
        self.max_policy_lag = max_policy_lag
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="math-rollout")
        self.pending = None
        self.pending_prompts = None
        self.diagnostics = diagnostics

    def publish(self, snapshot):
        if snapshot.version < self.snapshot.version:
            raise ValueError("actor policy version moved backwards")
        self.snapshot = snapshot

    def _job(self, prompts, snapshot):
        if self.diagnostics is not None:
            self.diagnostics.begin(policy_version=snapshot.version, prompt_count=len(prompts))
        try:
            rng_before = self.generator.get_state().clone()
            batch, timings = self.generate(prompts, snapshot, self.generator)
            return dict(prompts=list(prompts), batch=batch, timings=timings,
                        policy_version=snapshot.version, rng_before=rng_before)
        finally:
            if self.diagnostics is not None:
                self.diagnostics.end()

    def prefetch(self, prompts):
        if prompts is None or self.stop_requested():
            return
        if self.pending is not None:
            raise RuntimeError("rollout prefetch is already occupied")
        self.pending_prompts = tuple(prompts)
        self.pending = self.executor.submit(self._job, self.pending_prompts, self.snapshot)

    def consume(self, prompts, policy_version):
        if self.pending is None:
            self.prefetch(prompts)
            if self.pending is None:
                # A graceful stop still needs a final, explicitly censored
                # rollout for upstream Trainer to finish its update boundary.
                self.pending_prompts = tuple(prompts)
                self.pending = self.executor.submit(self._job, self.pending_prompts, self.snapshot)
        if tuple(prompts) != self.pending_prompts:
            raise ValueError("prefetched prompts differ from the resumed data cursor")
        started = time.perf_counter()
        if self.diagnostics is not None:
            self.diagnostics.mark("learner", "consume_wait", policy_version=policy_version)
        result = self.pending.result()
        if self.diagnostics is not None:
            self.diagnostics.mark("learner", "consume_ready", policy_version=policy_version)
        wait = time.perf_counter() - started
        self.pending = self.pending_prompts = None
        lag = policy_version - result["policy_version"]
        if not 0 <= lag <= self.max_policy_lag:
            raise ValueError("rollout policy lag exceeds the declared contract")
        result["timings"].update(rollout_wait_seconds=wait, source_policy_version=result["policy_version"],
                                 learner_policy_version=policy_version, policy_lag=lag)
        return result["batch"], result["timings"]

    def checkpoint_state(self):
        result = self.drain()
        if result is not None and (self.stop_requested() or "stop" in result["batch"].get("finish_reason", [])):
            # Resume regenerates this unconsumed batch with the same actor RNG.
            self.generator.set_state(result["rng_before"])
            self.pending = self.pending_prompts = None
            result = None
        return dict(generator=self.generator.get_state().clone(), pending=result)

    def drain(self):
        """Wait for actor work without consuming a batch or changing its RNG."""
        return self.pending.result() if self.pending is not None else None

    def restore(self, state=None, *, initial_rng=None):
        if self.pending is not None:
            raise RuntimeError("cannot restore a running rollout queue")
        self.generator.set_state(state["generator"] if state else initial_rng)
        if state and state["pending"] is not None:
            self.pending = Future()
            self.pending.set_result(state["pending"])
            self.pending_prompts = tuple(state["pending"]["prompts"])

    def close(self):
        self.closing.set()
        try:
            if self.pending is not None:
                self.pending.result()
        finally:
            self.executor.shutdown(wait=True)
            if self.diagnostics is not None:
                self.diagnostics.close()
