import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from archlab.artifacts import sha256_file
from archlab.automodel.checkpoint_cache import stage_checkpoint


def checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.pt").write_bytes(b"complete weights")
    (source / "optimizer.pt").write_bytes(b"complete optimizer")
    receipt = dict(files={name: sha256_file(source / name) for name in ["weights.pt", "optimizer.pt"]})
    (source / "COMPLETE.json").write_text(json.dumps(receipt))
    return source


def test_concurrent_ranks_share_one_verified_immutable_copy(tmp_path):
    source = checkpoint(tmp_path)
    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(lambda _: stage_checkpoint(source, tmp_path / "cache"), range(8)))
    assert len(set(paths)) == 1
    assert (paths[0] / "COMPLETE.json").read_bytes() == (source / "COMPLETE.json").read_bytes()
    assert (paths[0] / "optimizer.pt").read_bytes() == b"complete optimizer"
    (paths[0] / "optimizer.pt").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="payload changed"):
        stage_checkpoint(source, tmp_path / "cache")


def test_bad_source_never_publishes_partial_checkpoint(tmp_path):
    source = checkpoint(tmp_path)
    (source / "weights.pt").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        stage_checkpoint(source, tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*/COMPLETE.json"))


def test_receipt_cannot_escape_cache_directory(tmp_path):
    source = checkpoint(tmp_path)
    (source / "COMPLETE.json").write_text(json.dumps(dict(files={"../payload": "invalid"})))
    with pytest.raises(ValueError, match="unsafe"):
        stage_checkpoint(source, tmp_path / "cache")
