import json

import pytest

from archlab.automodel.stage_checkpoint import stage_checkpoint


def test_verified_copy_preserves_source_and_rejects_overwrite(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.safetensors").write_bytes(b"test artifact, not a real tensor file")
    (source / "config.json").write_text('{"example": true}')
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "weights.safetensors"}}))
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    target = tmp_path / "cache"
    record = stage_checkpoint(source, target, workers=2)
    assert record["weight_shard_files"] == 1
    assert not record["weight_transformation"]
    assert record["total_bytes"] == sum(map(len, original.values()))
    assert (target / "ARCHLAB_VERIFIED_COPY.json").exists()
    for name, content in original.items():
        assert (source / name).read_bytes() == content == (target / name).read_bytes()
    with pytest.raises(FileExistsError):
        stage_checkpoint(source, target)


def test_incomplete_or_escaping_index_fails_before_copy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    index = source / "model.safetensors.index.json"
    for name in ("missing.safetensors", "../escape.safetensors"):
        index.write_text(json.dumps({"weight_map": {"w": name}}))
        with pytest.raises(ValueError):
            stage_checkpoint(source, tmp_path / "cache")
        assert not (tmp_path / "cache").exists()


def test_resume_repairs_only_known_unfinished_copy(tmp_path):
    source, target = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    target.mkdir()
    (source / "weights.safetensors").write_bytes(b"complete test bytes")
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "weights.safetensors"}}))
    (target / "weights.safetensors").write_bytes(b"partial")
    (target / "unrelated").write_bytes(b"preserve me")
    with pytest.raises(ValueError, match="unexpected"):
        stage_checkpoint(source, target, resume=True)
    (target / "unrelated").unlink()  # test-owned fixture
    stage_checkpoint(source, target, resume=True)
    assert (target / "weights.safetensors").read_bytes() == b"complete test bytes"
    with pytest.raises(ValueError, match="unfinished"):
        stage_checkpoint(source, target, resume=True)
