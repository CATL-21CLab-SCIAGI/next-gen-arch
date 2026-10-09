import json
import threading

import pytest
import torch
from torch import nn

from archlab.automodel.checkpoint_publication import CheckpointPublisher
from archlab.automodel.limite_adapter_common import file_hash, save_adapter


def test_training_after_snapshot_cannot_mutate_async_optimizer_or_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: pytest.fail("CPU snapshot used CUDA"))
    model = nn.Module()
    model.model = nn.Module()
    model.model.adapters = nn.Linear(2, 2)
    model.model.base = nn.Linear(2, 2)
    model.model.trainable_mode = "full"
    model.model.adapter_config = {"variant": "normal"}
    model.archlab_base_snapshot_sha256 = "fixture"
    optimizer = torch.optim.Adam(model.parameters(), lr=.1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    before_moments = optimizer.state_dict()["state"][0]["exp_avg"].clone()
    release = threading.Event()

    class PausedPublisher(CheckpointPublisher):
        def submit(self, publish, *args):
            def delayed():
                assert release.wait(10)
                return publish(*args)
            return super().submit(delayed)

    publisher = PausedPublisher()
    dest = save_adapter(model, optimizer, tmp_path / "nas", 1, 0, tmp_path / "oss", publisher=publisher)
    assert not (tmp_path / "oss" / dest.name / "COMPLETE.json").exists()
    optimizer.step()
    assert not torch.equal(before_moments, optimizer.state_dict()["state"][0]["exp_avg"])
    release.set()
    publisher.close()
    receipt = json.loads((dest / "COMPLETE.json").read_text())
    for name, checksum in receipt["files"].items():
        assert (dest / name).is_symlink() and file_hash(dest / name) == checksum
    saved = torch.load(dest / "backbone.pt", weights_only=True)
    for name, tensor in saved.items():
        assert torch.equal(tensor, before[name])
    saved_optimizer = torch.load(dest / "optimizer.pt", weights_only=True)
    assert torch.equal(saved_optimizer["state"][0]["exp_avg"], before_moments)
    assert torch.load(dest / "rng.pt", weights_only=True)["cuda"] is None
