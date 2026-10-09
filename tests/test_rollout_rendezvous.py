from unittest.mock import Mock

import pytest

from archlab.rl import rollout_rendezvous


def test_rendezvous_is_host_only_and_uses_all_training_ranks(monkeypatch):
    group = object()
    distributed = Mock()
    distributed.is_initialized.return_value = True
    distributed.get_world_size.return_value = 16
    distributed.new_group.return_value = group
    monkeypatch.setattr(rollout_rendezvous, "dist", distributed)
    rendezvous = rollout_rendezvous.RolloutRendezvous(16)
    assert distributed.new_group.call_args.kwargs["backend"] == "gloo"
    rendezvous()
    assert distributed.monitored_barrier.call_args.kwargs["group"] is group
    assert distributed.monitored_barrier.call_args.kwargs["wait_all_ranks"] is True
    distributed.barrier.assert_not_called()


def test_single_rank_needs_no_distributed_runtime(monkeypatch):
    distributed = Mock()
    monkeypatch.setattr(rollout_rendezvous, "dist", distributed)
    rollout_rendezvous.RolloutRendezvous(1)()
    assert not distributed.mock_calls


def test_single_rank_object_logs_match_accelerate_identity(monkeypatch):
    distributed = Mock()
    monkeypatch.setattr(rollout_rendezvous, "dist", distributed)
    values = ["completion", ["nested"]]
    assert rollout_rendezvous.RolloutRendezvous(1).gather_object(values) is values
    assert not distributed.mock_calls


def test_object_logs_use_existing_host_group_and_flatten_one_level(monkeypatch):
    group = object()
    distributed = Mock()
    distributed.is_initialized.return_value = True
    distributed.get_world_size.return_value = 3
    distributed.new_group.return_value = group

    def gather(packets, value, *, group):
        packets[:] = [value, [], ["peer", ["nested"]]]

    distributed.all_gather_object.side_effect = gather
    monkeypatch.setattr(rollout_rendezvous, "dist", distributed)
    rendezvous = rollout_rendezvous.RolloutRendezvous(3)
    assert rendezvous.gather_object(["local"]) == ["local", "peer", ["nested"]]
    assert distributed.all_gather_object.call_args.kwargs["group"] is group
    assert distributed.new_group.call_count == 1


def test_world_mismatch_is_rejected(monkeypatch):
    distributed = Mock()
    distributed.is_initialized.return_value = True
    distributed.get_world_size.return_value = 8
    monkeypatch.setattr(rollout_rendezvous, "dist", distributed)
    with pytest.raises(RuntimeError, match="training world"):
        rollout_rendezvous.RolloutRendezvous(16)
    distributed.new_group.assert_not_called()
