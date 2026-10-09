import itertools
import threading

import pytest
import torch

from archlab.rl.async_rollout import AsyncRolloutQueue, BatchLookahead, PolicySnapshot


def test_lookahead_preserves_batch_order_and_epoch_boundaries():
    lookahead = BatchLookahead()

    def collect(iterator, count, device):
        return list(itertools.islice(iterator, count)), None

    for size in (13, 16, 1):
        iterator = iter(range(size))
        batches = []
        for _ in range((size + 3) // 4):
            rows, _ = lookahead.collect(collect, iterator, 4, None)
            batches.extend(rows)
        assert batches == list(range(size))
        assert lookahead.buffer is None


def make_queue(stop=lambda: False, gate=None, entered=None):
    def generate(prompts, snapshot, generator):
        if entered:
            entered.set()
        if gate:
            assert gate.wait(5)
        return dict(completion_ids=torch.randint(100, (4,), generator=generator).tolist(),
                    finish_reason=["eos"]), dict(weight=snapshot.weights)

    return AsyncRolloutQueue(generate, torch.Generator().manual_seed(123),
                             PolicySnapshot(0, "old"), stop)


def test_prefetch_uses_immutable_old_policy_and_overlaps_learner():
    gate, entered = threading.Event(), threading.Event()
    queue = make_queue(gate=gate, entered=entered)
    try:
        queue.prefetch(["question"])
        assert entered.wait(5)
        queue.publish(PolicySnapshot(1, "new"))
        gate.set()
        _, timings = queue.consume(["question"], 1)
        assert timings["weight"] == "old" and timings["policy_lag"] == 1
    finally:
        gate.set()
        queue.close()


def test_checkpoint_replays_pending_batch_and_next_rng_exactly():
    queue = make_queue()
    restored = make_queue()
    try:
        queue.prefetch(["next"])
        state = queue.checkpoint_state()
        restored.restore(state)
        left, lt = queue.consume(["next"], 1)
        right, rt = restored.consume(["next"], 1)
        assert left == right
        assert {k: v for k, v in lt.items() if k != "rollout_wait_seconds"} == {k: v for k, v in rt.items() if k != "rollout_wait_seconds"}
        assert queue.consume(["following"], 1)[0] == restored.consume(["following"], 1)[0]
    except AssertionError:
        # Waiting durations are intentionally execution measurements.
        pytest.fail("pending rollout payload changed across restore")
    finally:
        queue.close()
        restored.close()


def test_eval_drain_retains_prefetched_batch_rng_and_policy():
    gate, entered = threading.Event(), threading.Event()
    queue = make_queue(gate=gate, entered=entered)
    drained = threading.Event()
    try:
        queue.prefetch(["next"])
        assert entered.wait(5)
        pending = queue.pending
        waiter = threading.Thread(target=lambda: (queue.drain(), drained.set()))
        waiter.start()
        assert not drained.wait(.05)
        gate.set()
        waiter.join(5)
        assert drained.is_set() and queue.pending is pending
        before = queue.generator.get_state().clone()
        saved = queue.checkpoint_state()
        assert saved["pending"] is pending.result()
        batch, timing = queue.consume(["next"], 1)
        assert batch is saved["pending"]["batch"] and timing["policy_lag"] == 1
        assert torch.equal(before, queue.generator.get_state())
    finally:
        gate.set()
        queue.close()


def test_stop_rolls_back_unconsumed_actor_rng():
    stopped = False
    queue = make_queue(stop=lambda: stopped)
    try:
        before = queue.generator.get_state().clone()
        queue.prefetch(["next"])
        stopped = True
        state = queue.checkpoint_state()
        assert state["pending"] is None
        assert torch.equal(state["generator"], before)
    finally:
        queue.close()


def test_mismatched_prompts_and_excessive_lag_fail_closed():
    queue = make_queue()
    try:
        queue.prefetch(["expected"])
        with pytest.raises(ValueError, match="cursor"):
            queue.consume(["wrong"], 1)
        with pytest.raises(ValueError, match="lag"):
            queue.consume(["expected"], 2)
    finally:
        queue.close()


def test_background_errors_are_not_silently_dropped():
    def broken(*args):
        raise RuntimeError("actor failed")

    queue = AsyncRolloutQueue(broken, torch.Generator(), PolicySnapshot(0, None), lambda: False)
    queue.prefetch(["question"])
    with pytest.raises(RuntimeError, match="actor failed"):
        queue.consume(["question"], 0)
    with pytest.raises(RuntimeError, match="actor failed"):
        queue.close()


def test_close_requests_bounded_background_termination():
    entered = threading.Event()

    def generate(prompts, snapshot, generator):
        entered.set()
        assert queue.closing.wait(5)
        return dict(finish_reason=["stop"]), {}

    queue = AsyncRolloutQueue(generate, torch.Generator(), PolicySnapshot(0, None), lambda: False)
    queue.prefetch(["question"])
    assert entered.wait(5)
    queue.close()
    assert queue.stop_requested()
