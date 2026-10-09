from __future__ import annotations

from archlab.registry import find_run, load_runs, verify_manifest
from archlab.results import verify_metrics


def test_frozen_manifest_is_complete_cartesian_grid():
    assert verify_manifest() == {"runs": 144, "sizes": 3, "variants": 16, "seeds": 3}


def test_registry_has_unique_run_ids():
    runs = load_runs()
    assert len({run.run_id for run in runs}) == len(runs)


def test_registry_rejects_unknown_lookup():
    try:
        find_run("100m", "not-a-variant", 42)
    except KeyError as exc:
        assert "found 0" in str(exc)
    else:
        raise AssertionError("unknown variant lookup should fail")


def test_engram_1b_command_preserves_run_contract_and_enables_fail_fast():
    run = find_run("1b", "engram", 42)
    command = run.command()
    assert run.parameter_count == 1_060_350_426
    assert "--engram-layers=7,15,23" in command
    assert "--num-iterations=32359" in command
    assert "--finite-check-every=1" in command
    assert "--no-save-final-checkpoint" in command


def test_published_metric_table_is_well_formed():
    summary = verify_metrics()
    assert summary["rows"] >= 50
    assert summary["campaigns"] == 4
