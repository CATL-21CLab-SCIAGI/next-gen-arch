import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from archlab.megatron.miles_v41_policy_parity import InitialPolicyParity


def setup_observer(tmp_path, monkeypatch):
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    observer = InitialPolicyParity.__new__(InitialPolicyParity)
    observer.output = tmp_path
    observer.totals = torch.zeros(2, dtype=torch.float64)
    observer.maximum = torch.zeros((), dtype=torch.float64)
    observer.finished = False
    return observer


def test_parity_checks_only_response_tokens_and_only_initial_policy(tmp_path, monkeypatch):
    observer = setup_observer(tmp_path, monkeypatch)
    observer.record(torch.tensor([.02, 10.]), torch.zeros(2), torch.tensor([1, 0]))
    observer.finish()
    receipt = json.loads((tmp_path / "policy-parity-rank-00.json").read_text())
    assert receipt["policy_parity"] and receipt["response_tokens"] == 1
    observer.record(torch.tensor([9.]), torch.zeros(1), torch.ones(1))
    observer.finish()
    assert json.loads((tmp_path / "policy-parity-rank-00.json").read_text()) == receipt


@pytest.mark.parametrize("delta", [float("nan"), .1, .6])
def test_bad_initial_policy_cannot_update(tmp_path, monkeypatch, delta):
    observer = setup_observer(tmp_path, monkeypatch)
    observer.record(torch.tensor([delta]), torch.zeros(1), torch.ones(1))
    with pytest.raises(ValueError, match="policy parity failed"):
        observer.finish()
    assert not observer.finished


def test_missing_policy_evidence_cannot_update(tmp_path, monkeypatch):
    observer = setup_observer(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="policy parity failed"):
        observer.finish()
