"""The no-overlap rollout boundary retains prefetch state before collectives."""

import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from archlab.rl.async_rollout import AsyncRolloutQueue, PolicySnapshot
from archlab.rl.limite_rollout import native_rollout


@pytest.fixture(autouse=True)
def unused_trl_import(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("async rollout must not unwrap the learner")

    monkeypatch.setitem(sys.modules, "trl.models.utils", SimpleNamespace(unwrap_model_for_generation=forbidden))


def trainer(queue, output, overlap=False, next_prompts=("next",)):
    result = SimpleNamespace(
        archlab_async_rollout=queue,
        archlab_policy_version=lambda: 5,
        archlab_next_rollout_prompts=next_prompts,
        model=SimpleNamespace(training=True),
        _metrics={"train": defaultdict(list)},
        accelerator=SimpleNamespace(process_index=0, num_processes=1),
        args=SimpleNamespace(output_dir=str(output / "trainer")),
        state=SimpleNamespace(global_step=10),
    )
    if overlap is not None:
        result.archlab_overlap_actor_learner = overlap
    return result


def queue(generate):
    return AsyncRolloutQueue(generate, torch.Generator().manual_seed(123),
                             PolicySnapshot(4, "old"), lambda: False)


def sampled(prompts, snapshot, generator):
    return dict(completion_ids=torch.randint(100, (4,), generator=generator).tolist(),
                finish_reason=["eos"]), dict(weight=snapshot.weights)


def test_no_actor_work_is_pending_when_learner_metadata_is_entered(tmp_path):
    entered, released, active = threading.Event(), threading.Event(), threading.Event()

    def generate(prompts, snapshot, generator):
        active.set()
        try:
            if prompts == ("next",):
                entered.set()
                assert released.wait(5)
            return sampled(prompts, snapshot, generator)
        finally:
            active.clear()

    actor = queue(generate)
    learner = trainer(actor, tmp_path)
    metadata = []

    def learner_rollout_then_metadata():
        result = native_rollout(["current"], learner)
        metadata.append(not active.is_set() and actor.pending.done())
        return result

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            work = executor.submit(learner_rollout_then_metadata)
            try:
                assert entered.wait(5)
                pending = actor.pending
                assert not work.done() and not metadata
            finally:
                released.set()
            assert work.result(timeout=5)["finish_reason"] == ["eos"]
        assert metadata == [True] and actor.pending is pending
        assert actor.pending_prompts == ("next",)
        assert learner._metrics["train"]["rollout/drain_seconds"][0] > 0
        row = json.loads((tmp_path / "behavior.jsonl").read_text())
        assert row["rollout_drain_seconds"] > 0 and row["source_policy_version"] == 4
    finally:
        released.set()
        actor.close()


def test_fast_rank_cannot_enter_metadata_while_peer_actor_runs(tmp_path):
    slow_entered, release_slow, fast_waiting = (threading.Event() for _ in range(3))
    rendezvous = threading.Barrier(2, timeout=5)
    actors = []
    learners = []
    for rank in range(2):
        def generate(prompts, snapshot, generator, rank=rank):
            if rank == 1 and prompts == ("next",):
                slow_entered.set()
                assert release_slow.wait(5)
            return sampled(prompts, snapshot, generator)

        def wait_for_peers(rank=rank):
            if rank == 0:
                fast_waiting.set()
            rendezvous.wait()

        output = tmp_path / str(rank)
        output.mkdir()
        actor = queue(generate)
        learner = trainer(actor, output)
        learner.accelerator.num_processes = 2
        learner.archlab_rollout_rendezvous = wait_for_peers
        actors.append(actor)
        learners.append(learner)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            work = [executor.submit(native_rollout, ["current"], learner) for learner in learners]
            try:
                assert slow_entered.wait(5) and fast_waiting.wait(5)
                assert actors[0].pending.done() and not actors[1].pending.done()
                assert not any(future.done() for future in work)
            finally:
                release_slow.set()
            assert work[0].result(timeout=5) == work[1].result(timeout=5)
        assert all(actor.pending.done() for actor in actors)
        assert learners[0]._metrics["train"]["rollout/peer_wait_seconds"][0] > 0
    finally:
        release_slow.set()
        for actor in actors:
            actor.close()


def test_distributed_no_overlap_fails_closed_without_host_group(tmp_path):
    actor = queue(sampled)
    try:
        learner = trainer(actor, tmp_path)
        learner.accelerator.num_processes = 2
        with pytest.raises(RuntimeError, match="host rendezvous"):
            native_rollout(["current"], learner)
    finally:
        actor.close()


@pytest.mark.parametrize("overlap", [True, None])
def test_explicit_and_legacy_overlap_remain_nonblocking(tmp_path, overlap):
    entered, released = threading.Event(), threading.Event()

    def generate(prompts, snapshot, generator):
        if prompts == ("next",):
            entered.set()
            assert released.wait(5)
        return sampled(prompts, snapshot, generator)

    actor = queue(generate)
    try:
        learner = trainer(actor, tmp_path, overlap)
        native_rollout(["current"], learner)
        assert entered.wait(5) and not actor.pending.done()
        assert not learner._metrics["train"]["rollout/drain_seconds"]
    finally:
        released.set()
        actor.close()


def test_drain_preserves_payload_rng_snapshot_and_checkpoint_replay(tmp_path):
    def run(overlap, output):
        output.mkdir()
        actor, restored = queue(sampled), queue(sampled)
        try:
            learner = trainer(actor, output, overlap)
            current = native_rollout(["current"], learner)
            pending = actor.pending
            saved = actor.checkpoint_state()
            assert actor.pending is pending and saved["pending"] is pending.result()
            assert saved["pending"]["policy_version"] == 4
            assert saved["pending"]["timings"]["weight"] == "old"
            assert saved["pending"]["prompts"] == ["next"]
            before = actor.generator.get_state().clone()
            actor.publish(PolicySnapshot(5, "new"))
            restored.publish(PolicySnapshot(5, "new"))
            restored.restore(saved)
            expected, timing = actor.consume(["next"], 5)
            actual, restored_timing = restored.consume(["next"], 5)
            assert expected == actual and timing["policy_lag"] == restored_timing["policy_lag"] == 1
            assert torch.equal(before, actor.generator.get_state())
            following, ft = actor.consume(["following"], 5)
            assert (following, ft["source_policy_version"]) == (
                restored.consume(["following"], 5)[0], 5,
            )
            assert torch.equal(actor.generator.get_state(), restored.generator.get_state())
            return current, saved["pending"]["batch"], saved["generator"], following, actor.generator.get_state()
        finally:
            actor.close()
            restored.close()

    overlap = run(True, tmp_path / "overlap")
    drained = run(False, tmp_path / "drained")
    for left, right in zip(overlap, drained, strict=True):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        else:
            assert left == right


@pytest.mark.parametrize("next_prompts,stopped", [(None, False), (("next",), True)])
def test_no_next_batch_or_graceful_stop_does_not_create_prefetch(tmp_path, next_prompts, stopped):
    actor = queue(sampled)
    try:
        actor.prefetch(["current"])
        actor.drain()
        actor.stop_requested = lambda: stopped
        learner = trainer(actor, tmp_path, next_prompts=next_prompts)
        result = native_rollout(["current"], learner)
        assert result["finish_reason"] == ["eos"] and actor.pending is None
        assert learner._metrics["train"]["rollout/drain_seconds"][0] >= 0
    finally:
        actor.close()


def test_portable_async_recovery_recipe_changes_only_execution_overlap():
    root = Path(__file__).parents[1]
    recipe = yaml.safe_load((root / "recipes/limite/full_math_rl_async.yaml").read_text())
    assert recipe["name"] == "limite-native-full-math-rl-async-v3"
    execution = recipe["execution"]
    assert execution["overlap_actor_learner"] is False
    assert execution["rollout_rendezvous"] == "gloo_after_drain"
    assert execution["async_rollouts"] is True and execution["max_policy_lag"] == 1
    assert execution["actor_rng"] == "restored_rank_cuda_then_independent_generator"
    assert execution["checkpoint_prefetch"] == "completed_batch_and_generator_state"
    assert recipe["rollout"]["max_tokens"] == 16384
