"""CPU-only oracles for exact restores and update-normalized native replay."""

from copy import deepcopy

import pytest
import torch

from archlab.automodel.deepseek_v41_official_training import (
    assert_exact_training_state,
    assess_replay_protocol,
)


_ADAPTER_PATH = ("adapters", 5, "weight")
_MOMENT_PATHS = (
    ("optimizers", 0, "state", 0, "momentum_buffer"),
    ("optimizers", 1, "state", 0, "exp_avg"),
    ("optimizers", 1, "state", 0, "exp_avg_sq"),
)


def _value(state, path):
    for key in path:
        state = state[key]
    return state


@pytest.fixture
def replay_case():
    saved = {
        "adapters": {5: {"weight": torch.tensor([100., -100.], dtype=torch.float64)}},
        "optimizers": [
            {
                "state": {0: {"momentum_buffer": torch.tensor([.8, -.4], dtype=torch.float64)}},
                "param_groups": [{"params": [0], "lr": 1e-5, "momentum": .95}],
            },
            {
                "state": {0: {
                    "step": torch.tensor(7.),
                    "exp_avg": torch.tensor([.3, -.2], dtype=torch.float64),
                    "exp_avg_sq": torch.tensor([.1, .2], dtype=torch.float64),
                }},
                "param_groups": [{"params": [0], "lr": 1e-5, "betas": (.9, .95)}],
            },
        ],
        "rng": {
            "cpu": torch.tensor([1, 2, 3], dtype=torch.uint8),
            "cuda": torch.tensor([4, 5, 6], dtype=torch.uint8),
        },
    }
    expected = deepcopy(saved)
    _value(expected, _ADAPTER_PATH).add_(torch.tensor([.125, -.25], dtype=torch.float64))
    for path in _MOMENT_PATHS:
        _value(expected, path).mul_(1.1)
    expected["optimizers"][1]["state"][0]["step"].add_(1)
    metric = {"loss": 2., "supervised_tokens": 128, "gradient_norm_before_clip": .75}
    return saved, expected, metric


def _perturbed(saved, expected, fraction, component=None):
    """Inject a known fraction of the actual update, not of stored weights."""
    actual = deepcopy(expected)
    paths = [_ADAPTER_PATH, *_MOMENT_PATHS]
    for path in paths:
        if component is None or path[0] == component:
            _value(actual, path).add_(fraction * (_value(expected, path) - _value(saved, path)))
    return actual


def _assess(case, checkpoint=None, memories=None, checkpoint_metric=None):
    saved, expected, metric = case
    if checkpoint is None:
        checkpoint = deepcopy(expected)
    if memories is None:
        memories = [(deepcopy(metric), deepcopy(expected)) for _ in range(2)]
    return assess_replay_protocol(
        saved, metric, expected, checkpoint_metric or deepcopy(metric), checkpoint, memories
    )


@pytest.mark.parametrize("path", [
    _ADAPTER_PATH,
    *_MOMENT_PATHS,
    ("rng", "cpu"),
    ("rng", "cuda"),
])
def test_exact_restore_rejects_each_corrupted_component(replay_case, path):
    saved, _, _ = replay_case
    corrupted = deepcopy(saved)
    _value(corrupted, path)[0] += 1
    with pytest.raises(AssertionError):
        assert_exact_training_state(saved, corrupted)


def test_exact_restore_accepts_independent_identical_snapshot(replay_case):
    saved, _, _ = replay_case
    report = assert_exact_training_state(saved, deepcopy(saved))
    assert report["exact"] is True
    assert set(report["components"]) == {"adapters", "optimizers", "cpu_rng", "cuda_rng"}


@pytest.mark.parametrize("component", ["adapters", "optimizers"])
@pytest.mark.parametrize("replay", ["checkpoint", "in_memory"])
def test_fixed_ceiling_rejects_one_point_one_percent_update_error(replay_case, component, replay):
    saved, expected, metric = replay_case
    bad = _perturbed(saved, expected, .011, component)
    checkpoint = bad if replay == "checkpoint" else deepcopy(expected)
    memories = [(deepcopy(metric), deepcopy(expected)), (deepcopy(metric), deepcopy(expected))]
    if replay == "in_memory":
        memories[1] = (deepcopy(metric), bad)
    with pytest.raises(AssertionError):
        _assess(replay_case, checkpoint=checkpoint, memories=memories)


@pytest.mark.parametrize("component", ["adapters", "optimizers"])
def test_checkpoint_error_below_ceiling_must_still_match_memory_baseline(replay_case, component):
    saved, expected, metric = replay_case
    checkpoint = _perturbed(saved, expected, .006, component)
    memories = [(deepcopy(metric), _perturbed(saved, expected, fraction, component))
                for fraction in (.001, .002)]
    with pytest.raises(AssertionError):
        _assess(replay_case, checkpoint=checkpoint, memories=memories)


def test_bounded_native_variation_passes_and_keeps_inputs_unchanged(replay_case):
    saved, expected, metric = replay_case
    checkpoint = _perturbed(saved, expected, .003)
    checkpoint_metric = {**metric, "loss": metric["loss"] + 1e-6}
    memories = [(deepcopy(metric), _perturbed(saved, expected, fraction))
                for fraction in (.001, -.002)]
    snapshots = deepcopy((saved, expected, checkpoint, memories))
    report = _assess(replay_case, checkpoint, memories, checkpoint_metric)
    assert report["passed"] is True
    assert report["protocol"] == "native-update-baseline-v1"
    assert set(report["components"]) == {"adapters", "optimizer_moments"}
    torch.testing.assert_close((saved, expected, checkpoint, memories), snapshots, rtol=0, atol=0)


@pytest.mark.parametrize("fraction,passes", [(5e-6, True), (2e-5, False)])
def test_exact_native_repeats_use_the_fixed_baseline_floor(replay_case, fraction, passes):
    saved, expected, _ = replay_case
    checkpoint = _perturbed(saved, expected, fraction)
    if passes:
        assert _assess(replay_case, checkpoint=checkpoint)["passed"] is True
    else:
        with pytest.raises(AssertionError):
            _assess(replay_case, checkpoint=checkpoint)


@pytest.mark.parametrize("path,replacement", [
    (("optimizers", 1, "state", 0, "step"), torch.tensor(9.)),
    (("optimizers", 0, "param_groups", 0, "lr"), 1.000001e-5),
    (("optimizers", 0, "param_groups", 0, "momentum"), .95000001),
    (("optimizers", 1, "param_groups", 0, "betas"), (.90000001, .95)),
    (("optimizers", 1, "param_groups", 0, "params"), [1]),
])
@pytest.mark.parametrize("replay", ["checkpoint", "in_memory"])
def test_optimizer_counters_and_options_are_exact(replay_case, path, replacement, replay):
    _, expected, metric = replay_case
    corrupted = deepcopy(expected)
    _value(corrupted, path[:-1])[path[-1]] = replacement
    checkpoint = corrupted if replay == "checkpoint" else deepcopy(expected)
    memories = [(deepcopy(metric), deepcopy(expected)), (deepcopy(metric), deepcopy(expected))]
    if replay == "in_memory":
        memories[1] = (deepcopy(metric), corrupted)
    with pytest.raises(AssertionError):
        _assess(replay_case, checkpoint, memories)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("path", [_ADAPTER_PATH, *_MOMENT_PATHS])
def test_nonfinite_replayed_tensor_fails(replay_case, nonfinite, path):
    _, expected, _ = replay_case
    checkpoint = deepcopy(expected)
    _value(checkpoint, path)[0] = nonfinite
    with pytest.raises(AssertionError):
        _assess(replay_case, checkpoint=checkpoint)


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), 2.001])
@pytest.mark.parametrize("replay", ["checkpoint", "in_memory"])
def test_each_replay_requires_finite_loss_within_fixed_tolerance(replay_case, loss, replay):
    _, expected, metric = replay_case
    checkpoint_metric = {**metric, "loss": loss} if replay == "checkpoint" else deepcopy(metric)
    memories = [(deepcopy(metric), deepcopy(expected)), (deepcopy(metric), deepcopy(expected))]
    if replay == "in_memory":
        memories[1] = ({**metric, "loss": loss}, deepcopy(expected))
    with pytest.raises(AssertionError):
        _assess(replay_case, memories=memories, checkpoint_metric=checkpoint_metric)


@pytest.mark.parametrize("repeat_count", [0, 1])
def test_two_in_memory_repeats_are_required(replay_case, repeat_count):
    _, expected, metric = replay_case
    memories = [(deepcopy(metric), deepcopy(expected)) for _ in range(repeat_count)]
    with pytest.raises(AssertionError):
        _assess(replay_case, memories=memories)
