"""A throughput claim needs a matched, measured reference."""
import copy
import json

import pytest

from archlab.training.performance import ThroughputContract, check_throughput, compare_throughput


def record(seconds=(1., 1.1, .9)):
    return dict(contract=dict(scope="learner update", settings=dict(samples=3, sequence_length=2048, warmup=2),
                              model="weights-digest", world=8, runtime="pinned"), seconds=list(seconds))


def test_median_gate_rejects_real_regression_but_not_one_timing_outlier():
    assert compare_throughput(record([1, 1.01, 20]), record(), 1.2)["status"] == "passed"
    assert compare_throughput(record([1.3, 1.4, 1.5]), record(), 1.2)["status"] == "failed"


@pytest.mark.parametrize("key,value", [("model", "other-weights"), ("world", 4), ("runtime", "other-runtime")])
def test_unmatched_reference_is_not_a_speedup(key, value):
    candidate = record()
    candidate["contract"][key] = value
    with pytest.raises(ValueError, match="different"):
        compare_throughput(candidate, record(), 1.2)


@pytest.mark.parametrize("seconds", [[1, 2], [1, float("nan"), 2], [0, 1, 2], [1, True, 2], [1, 1, 1, 1]])
def test_invalid_or_incomplete_measurement_fails(seconds):
    with pytest.raises(ValueError, match="samples"):
        compare_throughput(record(seconds), record(), 1.2)


def test_missing_baseline_remains_pending_and_self_reference_fails(tmp_path):
    output = tmp_path / "candidate"
    output.mkdir()
    receipt = output / "throughput.json"
    receipt.write_text(json.dumps(record()))
    assert check_throughput(output, ThroughputContract(samples=3))["status"] == "pending"
    with pytest.raises(ValueError, match="own throughput"):
        check_throughput(output, ThroughputContract(samples=3, baseline=str(receipt)))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(record([1.1, 1.1, 1.1])))
    result = check_throughput(output, ThroughputContract(samples=3, baseline=str(baseline)))
    assert result["status"] == "passed"
    assert result["baseline_sha256"] != result["candidate_sha256"]
    with pytest.raises(ValueError, match="declared throughput workload"):
        check_throughput(output, ThroughputContract(samples=3, sequence_length=4096, baseline=str(baseline)))


def test_baseline_cannot_silently_change_allowed_slowdown():
    with pytest.raises(ValueError, match="allowance"):
        compare_throughput(record(), copy.deepcopy(record()), float("inf"))
