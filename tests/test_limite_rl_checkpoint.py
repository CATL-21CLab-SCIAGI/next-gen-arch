import io
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from archlab.rl.limite_checkpoint import (
    capture_rng,
    checkpoint_payloads,
    legacy_resume_state,
    resize_rank_state,
    restore_rng,
)


def test_smaller_world_resume_preserves_clocks_and_rewinds_discarded_prefetch():
    state = dict(world_size=4, scheduler={"last_epoch": 190}, evidence={"applied_updates": 157},
                 rank_rng=[dict(cpu=torch.tensor([rank]), rollout=dict(
                     generator=torch.tensor([rank + 10]),
                     pending=dict(rng_before=torch.tensor([rank + 20]), batch={"old_partition": rank})))
                     for rank in range(4)])
    resized = resize_rank_state(state, 2)
    assert resized["world_size"] == len(resized["rank_rng"]) == 2
    assert resized["scheduler"] == {"last_epoch": 190}
    assert resized["evidence"]["applied_updates"] == 157
    assert resized["evidence"]["topology_migration"]["discarded_prefetched_batches"] == 4
    assert not resized["evidence"]["topology_migration"]["exact_distributed_rng_resume"]
    for rank, row in enumerate(resized["rank_rng"]):
        assert row["rollout"]["pending"] is None
        assert torch.equal(row["rollout"]["generator"], torch.tensor([rank + 20]))
    assert state["world_size"] == 4 and "topology_migration" not in state["evidence"]
    assert all(row["rollout"]["pending"] is not None for row in state["rank_rng"])
    assert resize_rank_state(state, 4) is state
    with pytest.raises(ValueError, match="divisible smaller"):
        resize_rank_state(state, 3)


def test_rng_roundtrip_preserves_every_cpu_generator_without_cuda_contexts(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    random.seed(31)
    np.random.seed(37)
    torch.manual_seed(41)
    state = capture_rng()
    stream = io.BytesIO()
    torch.save(state, stream)
    expected = (random.random(), np.random.rand(4), torch.rand(4))
    stream.seek(0)
    restore_rng(torch.load(stream, weights_only=True))
    assert random.random() == expected[0]
    assert np.array_equal(np.random.rand(4), expected[1])
    assert torch.equal(torch.rand(4), expected[2])


def test_checkpoint_includes_optimizer_update_evidence_trainer_clock_and_rank_rng(monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class State:
        global_step: int = 17
        epoch: float = 0.2

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    trainer = SimpleNamespace(state=State(), lr_scheduler=SimpleNamespace(state_dict=lambda: {"last_epoch": 17}))
    evidence = dict(applied_updates=11, flat_batches=2, pending=False)
    payloads = checkpoint_payloads(trainer, evidence)
    assert payloads["trainer_state.json"]["global_step"] == 17
    state = payloads["rl_state.pt"]
    assert state["world_size"] == len(state["rank_rng"]) == 1
    assert state["scheduler"]["last_epoch"] == 17
    assert state["evidence"] == evidence
    evidence["applied_updates"] += 1
    assert state["evidence"]["applied_updates"] == 11


def test_legacy_migration_recovers_clock_without_modifying_saved_payloads(tmp_path):
    pytest.importorskip("transformers")
    checkpoint = tmp_path / "checkpoint" / "step-0000073"
    checkpoint.mkdir(parents=True)
    receipt = dict(step=73, files=dict())
    (checkpoint / "COMPLETE.json").write_text(json.dumps(receipt))
    (checkpoint / "optimizer.pt").write_bytes(b"immutable-moments")
    output = tmp_path / "rl"
    output.mkdir()
    (output / "updates.jsonl").write_text(json.dumps(dict(step=73, applied_updates=45, flat_batches=0)) + "\n")
    (output / "metrics.jsonl").write_text(json.dumps(dict(step=73, epoch=0.19, loss=0.0, num_tokens=1234)) + "\n")
    view, migration = legacy_resume_state(checkpoint, output, max_steps=400)
    state = json.loads((view / "trainer_state.json").read_text())
    assert state["global_step"] == 73 and state["epoch"] == 0.19
    assert state["train_batch_size"] == 1
    assert state["num_input_tokens_seen"] == 1234
    from transformers import TrainerState
    from transformers.trainer_utils import compare_trainer_and_checkpoint_args

    compare_trainer_and_checkpoint_args(
        SimpleNamespace(per_device_train_batch_size=1, n_gpu=1, logging_steps=1, eval_steps=100, save_steps=500),
        TrainerState.load_from_json(str(view / "trainer_state.json")),
    )
    assert migration["evidence"]["applied_updates"] == 45
    assert migration["exact_distributed_rng_resume"] is False
    assert (view / "optimizer.pt").is_symlink()
    assert (checkpoint / "optimizer.pt").read_bytes() == b"immutable-moments"
    assert not (checkpoint / "trainer_state.json").exists()
    assert legacy_resume_state(checkpoint, output, max_steps=400) == (view, migration)
    with pytest.raises(ValueError, match="existing legacy resume view differs"):
        legacy_resume_state(checkpoint, output, max_steps=401)


def test_async_prefetch_is_drained_before_all_rank_rng_capture(monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class State:
        global_step: int = 17

    order = []
    rollout = dict(generator=torch.Generator().get_state(), pending=dict(prompts=["next"], policy_version=10))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch, "get_rng_state", lambda: order.append("rng") or torch.zeros(4, dtype=torch.uint8))
    trainer = SimpleNamespace(state=State(), lr_scheduler=SimpleNamespace(state_dict=lambda: {}),
                              archlab_async_rollout=SimpleNamespace(checkpoint_state=lambda: order.append("drain") or rollout))
    payload = checkpoint_payloads(trainer, {})
    assert order == ["drain", "rng"]
    assert payload["rl_state.pt"]["rank_rng"][0]["rollout"] is rollout


def test_checkpoint_host_group_preserves_intact_rank_rng_and_pending_batches(monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class State:
        global_step: int = 190

    order = []
    local_rng = dict(cpu=torch.tensor([1, 2], dtype=torch.uint8), python=("local",))
    local_rollout = dict(generator=torch.tensor([3], dtype=torch.uint8),
                         pending=dict(prompts=["next"], completion_ids=[[11, 12]]))
    peer_rng = dict(cpu=torch.tensor([4, 5], dtype=torch.uint8), python=("peer",),
                    rollout=dict(generator=torch.tensor([6], dtype=torch.uint8),
                                 pending=dict(prompts=["peer next"], completion_ids=[[13]])))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def unexpected_default_collective(*args, **kwargs):
        raise AssertionError("checkpoint object must use the existing host group")

    monkeypatch.setattr(torch.distributed, "all_gather_object", unexpected_default_collective)
    monkeypatch.setattr("archlab.rl.limite_checkpoint.capture_rng", lambda: order.append("rng") or local_rng)

    def gather_rank_objects(value):
        order.append("host gather")
        assert value["rollout"] is local_rollout
        return [value, peer_rng]

    trainer = SimpleNamespace(
        state=State(), lr_scheduler=SimpleNamespace(state_dict=lambda: dict(last_epoch=190)),
        archlab_async_rollout=SimpleNamespace(checkpoint_state=lambda: order.append("drain") or local_rollout),
        archlab_rollout_rendezvous=SimpleNamespace(gather_rank_objects=gather_rank_objects),
    )
    payload = checkpoint_payloads(trainer, dict(applied_updates=157))
    saved = payload["rl_state.pt"]
    assert order == ["drain", "rng", "host gather"]
    assert saved["world_size"] == 2
    assert saved["rank_rng"][0] is local_rng and saved["rank_rng"][1] is peer_rng
    assert saved["evidence"] == dict(applied_updates=157)
    assert saved["scheduler"] == dict(last_epoch=190)
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)["rl_state.pt"]
    for original, loaded in zip(saved["rank_rng"], restored["rank_rng"], strict=True):
        assert torch.equal(loaded["cpu"], original["cpu"])
        assert loaded["python"] == original["python"]
        assert torch.equal(loaded["rollout"]["generator"], original["rollout"]["generator"])
        assert loaded["rollout"]["pending"] == original["rollout"]["pending"]
