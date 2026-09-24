from types import SimpleNamespace

import pytest
import torch

from archlab.megatron.miles_v41_resident_policy import install


def actor_type():
    class Actor:
        @property
        def _enable_weight_backup(self):
            return True

        def _get_actor_weights(self):
            return "upstream-backup"

    install(Actor)
    return Actor


def make_actor(cls, **overrides):
    actor = cls()
    actor.args = SimpleNamespace(colocate=True, offload_train=False, keep_old_actor=False)
    actor.with_ref = actor.with_opd_teacher = False
    actor._active_model_tag = "actor"
    for name, value in overrides.items():
        if hasattr(actor.args, name):
            setattr(actor.args, name, value)
        else:
            setattr(actor, name, value)
    return actor


def test_sync_reads_updated_live_weights_without_a_backup():
    actor = make_actor(actor_type())
    weight = torch.zeros(2)
    actor._named_actor_weights = lambda: [("weight", weight)]
    assert not actor._enable_weight_backup
    weight.add_(1)
    result = actor._get_actor_weights()["weight"]
    assert result.data_ptr() == weight.data_ptr()
    torch.testing.assert_close(result, torch.ones(2))
    actor._active_model_tag = "ref"
    with pytest.raises(RuntimeError, match="live actor"):
        actor._get_actor_weights()


@pytest.mark.parametrize("option", ["offload_train", "keep_old_actor", "with_ref", "with_opd_teacher"])
def test_other_lifecycles_keep_upstream_backups(option):
    actor = make_actor(actor_type(), **{option: True})
    assert actor._enable_weight_backup
    assert actor._get_actor_weights() == "upstream-backup"
