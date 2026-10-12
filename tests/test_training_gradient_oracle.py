"""Fixed numerical admission has meaningful negative controls."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from integration.test_megatron_agentic_training import (
    _assert_restored_state,
    _gradient_failures,
    _native_gradient_oracle,
    _pair_metrics,
    _rng_fingerprint,
    _snapshot,
    _state_fingerprint,
    _update_failures,
    _update_metrics,
)


@pytest.fixture
def native_pair(monkeypatch):
    actual = torch.nn.Linear(2, 1, bias=False)
    reference = deepcopy(actual)
    actual.weight.main_grad = torch.ones_like(actual.weight)
    reference.weight.grad = torch.ones_like(reference.weight)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    return SimpleNamespace(policy=actual), reference


def run_oracle(pair, tmp_path, *, actual=1., control=1., atomic=True):
    model, reference = pair
    model.policy.weight.main_grad.fill_(actual)
    def backward(item):
        item.weight.grad = torch.full_like(item.weight, control)
    return _native_gradient_oracle(
        model, reference, backward, atomic=atomic, root=tmp_path, name="test",
        native_factory=lambda: deepcopy(reference),
    )


def test_gradient_allows_bounded_native_variation(native_pair, tmp_path):
    report = run_oracle(native_pair, tmp_path, actual=1.001, control=1.002)
    assert len(report["unchanged_native_controls"]) == 2
    assert not _gradient_failures(report["bridge"])


def test_gradient_tiny_fixture_remains_exact(native_pair, tmp_path):
    with pytest.raises(AssertionError, match="bit-exact"):
        run_oracle(native_pair, tmp_path, actual=1.000001, atomic=False)


def test_scaled_gradient_fails(native_pair, tmp_path):
    with pytest.raises(AssertionError, match="global"):
        run_oracle(native_pair, tmp_path, actual=1.1)


def test_unstable_native_controls_block_admission(native_pair, tmp_path):
    with pytest.raises(AssertionError, match="native control exceeds"):
        run_oracle(native_pair, tmp_path, control=1.1)


def test_independent_control_rejects_changed_weights(native_pair, tmp_path):
    model, reference = native_pair
    def factory():
        control = deepcopy(reference)
        with torch.no_grad():
            control.weight.add_(1)
        return control
    with pytest.raises(AssertionError):
        _native_gradient_oracle(model, reference,
            lambda item: setattr(item.weight, "grad", torch.ones_like(item.weight)),
            atomic=True, root=tmp_path, name="changed", native_factory=factory)


def test_small_block_error_cannot_hide_in_global_norm():
    report = _pair_metrics([
        ("base.layers.0.weight", torch.ones(1000), torch.ones(1000)),
        ("adapters.7.weight", torch.tensor([1.1]), torch.ones(1)),
    ])
    assert "global" not in _gradient_failures(report)
    assert _gradient_failures(report) == ["adapters.7"]


def test_nonfinite_gradient_is_rejected():
    with pytest.raises(AssertionError, match="weight"):
        _pair_metrics([("weight", torch.tensor([float("nan")]), torch.ones(1))])


def test_unused_gradient_is_checked(native_pair, tmp_path):
    model, reference = native_pair
    reference.weight.grad = None
    model.policy.weight.main_grad.zero_()
    _native_gradient_oracle(model, reference, lambda _: None,
        atomic=False, root=tmp_path, name="zero")
    model.policy.weight.main_grad.fill_(1)
    with pytest.raises(AssertionError, match="bit-exact"):
        _native_gradient_oracle(model, reference, lambda _: None,
            atomic=False, root=tmp_path, name="nonzero")


def update_fixture(scale=1.):
    baseline = {"base.layers.0.weight": torch.ones(128),
                "adapters.7.weight": torch.ones(8)}
    expected = {key: value + .01 for key, value in baseline.items()}
    actual = {key: value + .01 * scale for key, value in baseline.items()}
    return actual, expected, baseline


def test_update_small_native_difference_passes():
    assert not _update_failures(_update_metrics(*update_fixture(1.00001)))


@pytest.mark.parametrize("scale", [0., .5, -1., 2.])
def test_update_noop_scaled_or_reversed_rejected(scale):
    failures = _update_failures(_update_metrics(*update_fixture(scale)))
    assert "step/magnitude" in failures or "step/direction" in failures


def test_update_parameter_block_perturbation_is_rejected():
    actual, expected, baseline = update_fixture()
    actual["adapters.7.weight"] += .001
    assert "weights/adapters.7" in _update_failures(_update_metrics(actual, expected, baseline))


@pytest.mark.parametrize("corruption", ["weight", "master", "moment", "group"])
def test_exact_state_detects_one_bit_corruption(corruption, tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1, bias=False)
    master = torch.nn.Parameter(model.weight.detach().clone())
    optimizer = torch.optim.AdamW([master], lr=.001)
    master.grad = torch.ones_like(master)
    optimizer.step()
    chained = SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=optimizer)])
    state = _state_fingerprint(model, chained)
    monkeypatch.setenv("ARCHLAB_MEGATRON_WORK", str(tmp_path))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    _assert_restored_state(model, chained, state, "clean")
    if corruption == "group":
        optimizer.param_groups[0]["lr"] *= 2
    else:
        target = {"weight": model.weight, "master": master,
                  "moment": optimizer.state[master]["exp_avg"]}[corruption]
        with torch.no_grad():
            target.view(torch.int32).reshape(-1)[0].bitwise_xor_(1)
    with pytest.raises(AssertionError):
        _assert_restored_state(model, chained, state, "corrupted")


def test_exact_state_empty_group_counter_is_inert_but_populated_counter_is_checked():
    model = torch.nn.Linear(2, 1, bias=False)
    master = torch.nn.Parameter(model.weight.detach().clone())
    optimizer = torch.optim.AdamW([master], lr=.001)
    optimizer.add_param_group({"params": []})
    chained = SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=optimizer)])
    before = _state_fingerprint(model, chained)
    optimizer.param_groups[1]["step"] = 1
    assert _state_fingerprint(model, chained) == before
    optimizer.param_groups[0]["step"] = 1
    assert _state_fingerprint(model, chained) != before


def test_rng_fingerprint_detects_changed_random_state():
    state = {"torch": torch.tensor([1, 2], dtype=torch.uint8),
             "numpy": np.array([3, 4], dtype=np.uint32), "python": (3, (1, 2), None)}
    engine = SimpleNamespace(checkpoint_mananager=SimpleNamespace(
        get_rng_state=lambda **kwargs: state))
    before = _rng_fingerprint(engine)
    assert _rng_fingerprint(engine) == before
    state["torch"][0] ^= 1
    assert _rng_fingerprint(engine) != before


def test_snapshot_keeps_device_and_owns_independent_storage(monkeypatch):
    model = torch.nn.Linear(2, 1, bias=False)
    model.tied_alias = model.weight
    def forbid_cpu_transfer(*args, **kwargs):
        raise AssertionError("oracle copies must stay on their current device")
    monkeypatch.setattr(torch.Tensor, "cpu", forbid_cpu_transfer)
    saved = _snapshot(model)
    assert list(saved) == ["weight"]
    assert saved["weight"].device == model.weight.device
    assert saved["weight"].dtype == model.weight.dtype
    assert saved["weight"].data_ptr() != model.weight.data_ptr()
    assert torch.equal(saved["weight"], model.weight)
    with torch.no_grad():
        model.weight.add_(1)
    assert not torch.equal(saved["weight"], model.weight)
