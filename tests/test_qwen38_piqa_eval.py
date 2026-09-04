import copy

import pytest

from scripts.eval_qwen38_piqa_curve import (
    _canonical_sha256,
    _checkpoint_identity,
    _validate_cached_result,
)


def test_piqa_cached_result_requires_exact_evaluation_identity():
    identity = {
        "checkpoint": {"metadata_sha256": "checkpoint-a"},
        "tokenizer": {"sha256": "tokenizer-a"},
        "piqa": {"sha256": "piqa-a"},
        "evaluator": {"sha256": "evaluator-a"},
    }
    result = {
        "evaluation_identity": identity,
        "evaluation_identity_sha256": _canonical_sha256(identity),
    }

    _validate_cached_result(result, identity)

    changed = copy.deepcopy(identity)
    changed["tokenizer"]["sha256"] = "tokenizer-b"
    with pytest.raises(RuntimeError, match="--overwrite"):
        _validate_cached_result(result, changed)


def test_checkpoint_identity_changes_with_distributed_metadata(tmp_path):
    checkpoint = tmp_path / "iter_0000010"
    checkpoint.mkdir()
    (checkpoint / ".metadata").write_bytes(b"first")
    (checkpoint / "__0_0.distcp").write_bytes(b"weights")
    first = _checkpoint_identity(checkpoint)

    (checkpoint / ".metadata").write_bytes(b"second")
    second = _checkpoint_identity(checkpoint)

    assert first["metadata_sha256"] != second["metadata_sha256"]
