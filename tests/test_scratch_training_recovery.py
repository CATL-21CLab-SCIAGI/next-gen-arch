"""Full-state recovery survives validation failure without extra eval milestones."""

import json

import pytest

from archlab.automodel.deepseek_v41_scratch_training import (
    checkpoint_cursor,
    consume_pending_validation,
    record_recovery_checkpoint,
    recovery_checkpoint_due,
)


@pytest.mark.parametrize("step,validation_due,expected", [
    (9, False, False), (10, False, True), (99, False, False), (100, False, True),
    (1829, True, True), (1829, False, False),
])
def test_recovery_between_milestones_and_before_validation(step, validation_due, expected):
    assert recovery_checkpoint_due(step, start_step=0, validation_due=validation_due) is expected


def test_resumed_training_also_saves_after_ten_new_updates():
    assert recovery_checkpoint_due(1839, start_step=1829, validation_due=False)
    assert not recovery_checkpoint_due(1838, start_step=1829, validation_due=False)


def completed_checkpoint(output, step, contract):
    cursor = dict(step=step, phase_step=step, supervised_tokens=step * 50_000,
                  window_cursor=step * 64)
    path = output / "recovery" / f"step-{step:06d}"
    path.mkdir(parents=True)
    (path / "COMPLETE.json").write_text(json.dumps(dict(
        format="archlab-v41-full-sharded-v1", cursor=cursor, contract=contract
    )))
    (path / "full-state.pt").write_bytes(b"weights, optimizer, and RNG")
    return path, cursor


def test_recovery_retains_two_full_states_and_preserves_oss_eval_checkpoints(tmp_path):
    contract = dict(target_supervised_tokens=1_000_000_000)
    eval_payload = tmp_path / "oss-eval-state"
    eval_payload.mkdir()
    (eval_payload / "full-state.pt").write_bytes(b"protected evaluation checkpoint")
    eval_link = tmp_path / "checkpoints" / "step-003658"
    eval_link.parent.mkdir()
    eval_link.symlink_to(eval_payload, target_is_directory=True)
    saves = []
    for step in (10, 100, 1829):
        path, cursor = completed_checkpoint(tmp_path, step, contract)
        record_recovery_checkpoint(tmp_path, path, cursor, contract)
        saves.append(path)
    assert not saves[0].exists()
    assert all((path / "full-state.pt").exists() for path in saves[1:])
    catalog = json.loads((tmp_path / "RECOVERY_CHECKPOINTS.json").read_text())
    assert [row["cursor"]["step"] for row in catalog["checkpoints"]] == [100, 1829]
    assert all(len(row["complete_marker_sha256"]) == 64 for row in catalog["checkpoints"])
    assert eval_link.is_symlink() and (eval_payload / "full-state.pt").exists()


@pytest.mark.parametrize("corruption", ["missing_marker", "wrong_contract", "wrong_cursor"])
def test_incomplete_recovery_never_replaces_last_completed_state(tmp_path, corruption):
    contract = dict(target_supervised_tokens=1_000_000_000)
    first, cursor = completed_checkpoint(tmp_path, 10, contract)
    record_recovery_checkpoint(tmp_path, first, cursor, contract)
    before = (tmp_path / "RECOVERY_CHECKPOINTS.json").read_bytes()
    new, cursor = completed_checkpoint(tmp_path, 100, contract)
    marker_path = new / "COMPLETE.json"
    if corruption == "missing_marker":
        marker_path.unlink()
    else:
        marker = json.loads(marker_path.read_text())
        if corruption == "wrong_contract":
            marker["contract"]["target_supervised_tokens"] = 10_000_000_000
        else:
            marker["cursor"]["step"] -= 1
        marker_path.write_text(json.dumps(marker))
    with pytest.raises((ValueError, FileNotFoundError)):
        record_recovery_checkpoint(tmp_path, new, cursor, contract)
    assert (tmp_path / "RECOVERY_CHECKPOINTS.json").read_bytes() == before
    assert (first / "full-state.pt").exists()


def test_recovery_catalog_cannot_prune_a_symlink_target(tmp_path):
    contract = dict(target_supervised_tokens=1_000_000_000)
    outside = tmp_path / "protected"
    outside.mkdir()
    (outside / "payload").write_text("preserve")
    root = tmp_path / "run"
    prior = root / "recovery" / "step-000010"
    prior.parent.mkdir(parents=True)
    prior.symlink_to(outside, target_is_directory=True)
    (root / "RECOVERY_CHECKPOINTS.json").write_text(json.dumps(dict(
        format="archlab-v41-scratch-recovery-v1",
        checkpoints=[dict(path=str(prior), cursor=dict(step=10))]
    )))
    new, cursor = completed_checkpoint(root, 100, contract)
    with pytest.raises(ValueError, match="owned checkpoints"):
        record_recovery_checkpoint(root, new, cursor, contract)
    assert (outside / "payload").read_text() == "preserve"


@pytest.mark.parametrize("corruption", ["catalog_format", "completion_marker"])
def test_recovery_rejects_changed_prior_evidence_before_pruning(tmp_path, corruption):
    contract = dict(target_supervised_tokens=1_000_000_000)
    first, cursor = completed_checkpoint(tmp_path, 10, contract)
    record_recovery_checkpoint(tmp_path, first, cursor, contract)
    if corruption == "catalog_format":
        path = tmp_path / "RECOVERY_CHECKPOINTS.json"
        value = json.loads(path.read_text())
        value["format"] = "unrecognized"
    else:
        path = first / "COMPLETE.json"
        value = json.loads(path.read_text())
        value["changed_without_catalog_update"] = True
    path.write_text(json.dumps(value))
    new, cursor = completed_checkpoint(tmp_path, 100, contract)
    with pytest.raises(ValueError, match="catalog format|marker changed"):
        record_recovery_checkpoint(tmp_path, new, cursor, contract, keep=1)
    assert (first / "full-state.pt").exists()


def test_prevalidation_checkpoint_round_trip_retries_the_failed_boundary_once(tmp_path):
    live = dict(step=1829, phase_step=1829, supervised_tokens=100_027_437,
                window_cursor=117_056)
    saved = checkpoint_cursor(live, next_validation_tokens=100_000_000)
    assert "pending_validation_tokens" not in live
    path = tmp_path / "recovery" / "step-001829"
    path.mkdir(parents=True)
    contract = dict(target_supervised_tokens=1_000_000_000)
    (path / "COMPLETE.json").write_text(json.dumps(dict(
        format="archlab-v41-full-sharded-v1", cursor=saved, contract=contract
    )))
    record_recovery_checkpoint(tmp_path, path, saved, contract)
    restored = json.loads((path / "COMPLETE.json").read_text())["cursor"]
    assert consume_pending_validation(restored) == 100_000_000
    assert restored == live
    assert consume_pending_validation(restored) is None
    assert checkpoint_cursor(live, next_validation_tokens=200_000_000)["pending_validation_tokens"] is None


@pytest.mark.parametrize("boundary", [True, -100_000_000, 50_000_000, 200_000_000, 100_000_000.0])
def test_corrupt_pending_validation_cannot_advance_the_restored_cursor(boundary):
    cursor = dict(supervised_tokens=100_027_437, pending_validation_tokens=boundary)
    before = dict(cursor)
    with pytest.raises(ValueError, match="pending validation"):
        consume_pending_validation(cursor)
    assert cursor == before
