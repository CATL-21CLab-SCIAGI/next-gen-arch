"""A native RL evaluation must read the trained full weights, not its publisher parent."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from archlab.artifacts import sha256_file


def test_native_evaluation_loads_trained_weights_and_checks_receipt(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from archlab.automodel import limite_native_checkpoint as native
    from archlab.automodel import limite_pipeline_benchmark as evaluation

    identity = dict(repo="paradigma-inc/limite-1b-violetto", revision="verified", receipt_sha256="publisher")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.projection = nn.Linear(2, 2, bias=False)
            nn.init.eye_(self.model.projection.weight)

        def forward(self, x):
            return self.model.projection(x)

    monkeypatch.setattr(native, "publisher_identity", lambda snapshot: identity)
    monkeypatch.setattr(native, "load_model", lambda *args, **kwargs: Model())
    monkeypatch.setattr(evaluation, "build_model", lambda *args, **kwargs: pytest.fail("inserted an adapter"))
    checkpoint = tmp_path / "step-0000400"
    checkpoint.mkdir()
    state = Model().state_dict()
    state["model.projection.weight"] *= 3
    torch.save(state, checkpoint / "model.pt")
    receipt = dict(model_kind="native", trainable_mode="full", step=400,
                   publisher_identity=identity, files={"model.pt": sha256_file(checkpoint / "model.pt")})
    (checkpoint / "COMPLETE.json").write_text(json.dumps(receipt))
    phase = dict(variant="native", checkpoint=str(checkpoint),
                 checkpoint_receipt_sha256=sha256_file(checkpoint / "COMPLETE.json"))
    model = evaluation.load_phase_model(dict(model="publisher"), phase,
                                        checkpoint_cache=None, device="cpu")
    torch.testing.assert_close(model(torch.tensor([[1., 2.]])), torch.tensor([[3., 6.]]))
    assert model.archlab_native_checkpoint and not hasattr(model.model, "adapters")
    (checkpoint / "COMPLETE.json").write_text(json.dumps(dict(receipt, step=399)))
    with pytest.raises(ValueError, match="checkpoint changed"):
        evaluation.load_phase_model(dict(model="publisher"), phase, checkpoint_cache=None, device="cpu")


def test_publisher_phase_cannot_silently_discard_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from archlab.automodel import limite_pipeline_benchmark as evaluation

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "COMPLETE.json").write_text("{}")
    monkeypatch.setattr(evaluation, "load_model", lambda *args, **kwargs: SimpleNamespace())
    phase = dict(variant="base", checkpoint=str(checkpoint),
                 checkpoint_receipt_sha256=sha256_file(checkpoint / "COMPLETE.json"))
    with pytest.raises(ValueError, match="cannot ignore"):
        evaluation.load_phase_model(dict(model="publisher"), phase, checkpoint_cache=None, device="cpu")
    with pytest.raises(ValueError, match="registered checkpoint"):
        evaluation.load_phase_model(dict(model="publisher"), dict(variant="native"), checkpoint_cache=None)
