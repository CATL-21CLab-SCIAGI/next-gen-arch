"""The early signal check must retain upstream eval-to-train semantics."""

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.mark.parametrize("signal", [False, True])
def test_post_eval_training_prepares_one_training_microbatch(monkeypatch, signal):
    modes, forwards = [], []

    class UpstreamTrainer:
        def _prepare_inputs(self, inputs):
            # Upstream GRPO dispatches using model.training. Eval keeps all four
            # generations; train splits them into one-sample microbatches.
            modes.append(self.model.training)
            if self.model.training:
                return {key: value[:1] for key, value in inputs.items()}
            return inputs

        def training_step(self, model, inputs, num_items_in_batch):
            model.train()
            prepared = self._prepare_inputs(inputs)
            forwards.append(prepared["completion_mask"].shape[0])
            self._step += 1
            self._metrics["train"]["step_time"].append(0.0)
            return torch.tensor(1.0)

    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(GRPOTrainer=UpstreamTrainer))
    path = Path(__file__).parents[1] / "src/archlab/rl/limite_trainer.py"
    spec = importlib.util.spec_from_file_location("_limite_transition_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trainer = module.BehaviorGRPO()
    trainer.model = nn.Linear(1, 1).eval()
    trainer.accelerator = SimpleNamespace(device="cpu")
    trainer.archlab_skip_flat_backward = True
    trainer.beta = 0
    trainer._step = 0
    trainer.current_gradient_accumulation_steps = 1
    trainer._current_train_step_time = 0.0
    trainer._metrics = {"train": defaultdict(list)}
    inputs = dict(
        advantages=torch.full((4,), float(signal)),
        completion_mask=torch.ones(4, 3),
    )
    trainer.training_step(trainer.model, inputs, 12)
    assert modes == [True]
    assert forwards == ([1] if signal else [])
    assert trainer.model.training and trainer._step == 1
    assert getattr(trainer, "_archlab_prepared_once", None) is None
