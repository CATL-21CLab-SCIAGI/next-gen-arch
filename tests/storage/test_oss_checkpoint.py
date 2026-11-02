import json

import pytest

from archlab.storage import oss_checkpoint


def test_links_are_immutable_and_share_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(oss_checkpoint, "require_oss_mount", lambda path: {})
    local = tmp_path / "nas" / "checkpoint"
    remote = tmp_path / "oss" / "checkpoint"
    oss_checkpoint.prepare_link(local, remote)
    (local / "payload").write_bytes(b"checkpoint")
    assert (remote / "payload").read_bytes() == b"checkpoint"
    with pytest.raises(FileExistsError):
        oss_checkpoint.prepare_link(local, remote)
    assert (remote / "payload").read_bytes() == b"checkpoint"


def test_missing_oss_mount_never_silently_writes_local_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(
        oss_checkpoint.subprocess,
        "check_output",
        lambda *a, **k: json.dumps(
            {
                "filesystems": [{"target": "/", "fstype": "ext4"}],
            }
        ),
    )
    local, remote = tmp_path / "nas", tmp_path / "unmounted" / "checkpoint"
    with pytest.raises(ValueError, match="not mounted OSS"):
        oss_checkpoint.prepare_link(local, remote)
    assert not local.exists() and not remote.exists()


def test_evaluation_catalog_rejects_duplicate_milestones(tmp_path, monkeypatch):
    monkeypatch.setattr(oss_checkpoint, "require_oss_mount", lambda path: {})
    monkeypatch.setattr(oss_checkpoint, "on_rank_zero", lambda function: function())
    monkeypatch.setattr(oss_checkpoint, "validate", lambda path: {"ranks": 16})
    checkpoint, remote = tmp_path / "checkpoint", tmp_path / "oss"
    oss_checkpoint.prepare_link(checkpoint, remote)
    (remote / "COMPLETE.json").write_text(
        json.dumps(
            {
                "cursor": {"supervised_tokens": 2_000_000_037, "step": 123, "window_cursor": 7872},
            }
        )
    )
    oss_checkpoint.record_evaluation_checkpoint(tmp_path, checkpoint, milestone=2_000_000_000)
    with pytest.raises(ValueError, match="duplicate"):
        oss_checkpoint.record_evaluation_checkpoint(tmp_path, checkpoint, milestone=2_000_000_000)
    catalog = json.loads((tmp_path / "EVAL_CHECKPOINTS.json").read_text())
    assert len(catalog["checkpoints"]) == 1
    assert catalog["checkpoints"][0]["oss_path"] == str(remote)


def test_repeated_data_checkpoints_require_the_matching_one_billion_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(oss_checkpoint, "require_oss_mount", lambda path: {})
    monkeypatch.setattr(oss_checkpoint, "on_rank_zero", lambda function: function())
    monkeypatch.setattr(oss_checkpoint, "validate", lambda path: {"ranks": 16})
    checkpoint, remote = tmp_path / "checkpoint", tmp_path / "oss"
    oss_checkpoint.prepare_link(checkpoint, remote)
    (remote / "COMPLETE.json").write_text(
        json.dumps(
            {
                "contract": {"target_supervised_tokens": 1_000_000_000},
                "cursor": {"supervised_tokens": 200_000_001, "step": 1},
            }
        )
    )
    with pytest.raises(ValueError, match="milestones"):
        oss_checkpoint.record_evaluation_checkpoint(tmp_path, checkpoint, milestone=200_000_000)
    oss_checkpoint.record_evaluation_checkpoint(
        tmp_path, checkpoint, milestone=200_000_000, budget=1_000_000_000
    )
