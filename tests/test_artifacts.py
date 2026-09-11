import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from archlab.artifacts import atomic_write_json, sha256_file


@pytest.mark.parametrize("chunk", [1, 7, 8192])
def test_streaming_file_hash_preserves_bytes(tmp_path, chunk):
    path = tmp_path / "bytes"
    value = bytes(range(256)) * 10
    path.write_bytes(value)
    assert sha256_file(path, chunk_bytes=chunk) == hashlib.sha256(value).hexdigest()
    with pytest.raises(ValueError):
        sha256_file(path, chunk_bytes=0)


@pytest.mark.parametrize("ascii_only", [False, True])
def test_atomic_json_preserves_legacy_encoding(tmp_path, ascii_only):
    path = tmp_path / "nested" / "metadata.json"
    payload = {"中文": [1, .5, "é"], "a": None}
    atomic_write_json(path, payload, ensure_ascii=ascii_only)
    expected = json.dumps(payload, ensure_ascii=ascii_only, indent=2, sort_keys=True) + "\n"
    assert path.read_bytes() == expected.encode("utf-8")
    assert list(path.parent.iterdir()) == [path]


def test_failed_write_leaves_previous_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text("old")
    with pytest.raises(ValueError):
        atomic_write_json(path, {"loss": float("nan")}, allow_nan=False)
    assert path.read_text() == "old"
    with pytest.raises(TypeError):
        atomic_write_json(path, {"unsupported": object()})
    assert path.read_text() == "old"

    def fail(*args):
        raise OSError("injected replace error")

    monkeypatch.setattr("archlab.artifacts.os.replace", fail)
    with pytest.raises(OSError, match="replace"):
        atomic_write_json(path, {"new": True})
    assert path.read_text() == "old"
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_writers_publish_complete_json(tmp_path):
    path = tmp_path / "state.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: atomic_write_json(path, {"index": i, "content": str(i) * 1000}), range(32)))
    value = json.loads(path.read_text())
    assert value["content"] == str(value["index"]) * 1000
    assert list(tmp_path.iterdir()) == [path]


def test_unicode_identity_dialects_remain_distinct_and_unchanged():
    from archlab.evaluation.qwen38_piqa import _canonical_sha256 as piqa_hash
    from archlab.megatron.qwen38_27b_train import _canonical_sha256 as trainer_hash
    from archlab.provenance import stable_json_sha256

    payload = {"文本": "数学"}
    escaped = hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    utf8 = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert piqa_hash(payload) == trainer_hash(payload) == escaped
    assert stable_json_sha256(payload) == utf8 != escaped
