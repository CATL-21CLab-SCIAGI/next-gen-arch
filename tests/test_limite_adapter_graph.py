"""CPU coverage for opt-in synchronization and the final partial-step fallback."""

from types import SimpleNamespace

import pytest
import torch

from archlab.automodel.limite_adapter_graph import ManualTrainingGraph, graph_contract


def test_static_synchronization_contract_requires_manual_graphs():
    assert graph_contract() == {"mode": "none"}
    native = graph_contract("manual")
    assert graph_contract("manual", static_gradient_sync=False) == native
    static = graph_contract("manual", static_gradient_sync=True)
    assert static.pop("gradient_presence") == "qualified_all_used_manual_graph"
    assert static == native
    with pytest.raises(ValueError, match="requires manual graphs"):
        graph_contract(static_gradient_sync=True)


def test_unqualified_cli_rejects_static_synchronization_before_cuda(monkeypatch, tmp_path):
    import sys

    from archlab.automodel import limite_adapter_warmup

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "warmup",
            "--model",
            "unused-model",
            "--variant",
            "normal",
            "--data",
            str(tmp_path),
            "--output",
            str(tmp_path),
            "--static-gradient-sync",
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid execution contracts must fail before CUDA initialization")

    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    with pytest.raises(ValueError, match="requires qualified manual graphs"):
        limite_adapter_warmup.main()


def test_resume_with_only_partial_step_qualifies_gradients_without_capture():
    class PartialObjective(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))
            self.policy = SimpleNamespace(
                model=SimpleNamespace(prepare_static_training_context=lambda ids: None)
            )

        def forward(self, ids, labels):
            return (self.weight * ids.float())[labels != -100].sum()

    # Isolate the ordinary CPU fallback after constructor qualification; the
    # real 16-rank geometry and capture are covered by distributed qualification.
    objective = PartialObjective()
    graph = ManualTrainingGraph.__new__(ManualTrainingGraph)
    graph.objective = objective
    graph.optimizer = torch.optim.AdamW(objective.parameters())
    graph.parameters = tuple(objective.parameters())
    graph.world_size = 16
    graph.full_tokens = 128 * 2048
    graph._static_gradient_sync = True
    graph.gradient_groups = None
    graph.partial_fallbacks = 0
    graph.graph = None
    graph._validate = lambda ids, labels: None
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[0, 0, -100, -100]])
    loss = graph.backward(ids, labels, remaining_tokens=2)
    assert loss.item() == 6.0
    assert objective.weight.grad.item() == 24.0
    assert graph.partial_fallbacks == 1
    assert graph.graph is None
    assert graph.gradient_groups is not None
    gradient = objective.weight.grad
    graph.gradient_groups.synchronize(world=1)
    assert objective.weight.grad is gradient
    assert not graph.optimizer.state
    with pytest.raises(AttributeError):
        graph.static_gradient_sync = False


def test_static_synchronization_rejects_stream_change_before_reduction(monkeypatch):
    from archlab.automodel.limite_adapter_communication import StaticGradientGroups

    parameter = torch.nn.Parameter(torch.tensor(0.25))
    parameter.grad = torch.tensor(0.5)
    graph = ManualTrainingGraph.__new__(ManualTrainingGraph)
    graph._static_gradient_sync = True
    graph.gradient_groups = StaticGradientGroups([parameter])
    graph.world_size = 1
    graph.stream = SimpleNamespace(cuda_stream=3)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=4))
    with pytest.raises(RuntimeError, match="synchronization changed CUDA stream"):
        graph.synchronize_gradients()
    assert parameter.grad.item() == 0.5
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: graph.stream)
    graph.synchronize_gradients()
    assert parameter.grad.item() == 0.5
