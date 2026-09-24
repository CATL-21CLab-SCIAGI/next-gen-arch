import json
import tempfile
from pathlib import Path

import pytest

from archlab.megatron.miles_v41_resident_admission import verify


def test_failed_numeric_receipt_cannot_admit_production():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "fp32-streaming-restore-admission").mkdir()
        (root / "fp32-streaming-restore-admission/rank-0.json").write_text(json.dumps({
            "rank": 0, "momentum_dtype": "float32", "passed": True, "max_update_relative_error": .36,
            "min_update_cosine": .94, "sinkhorn_distributed": True,
            "resident_resume_exact": True, "distributed_checkpoint_roundtrip": True, "streaming_checkpoint": True, "destructive_restore_verified": True}))
        with pytest.raises(ValueError, match="numerical admission rejected"):
            verify(root)


def test_component_success_without_full_model_is_not_admitted():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "fp32-streaming-restore-admission").mkdir()
        for rank in range(2):
            (root / f"fp32-streaming-restore-admission/rank-{rank}.json").write_text(json.dumps({
                "rank": rank, "momentum_dtype": "float32", "passed": True, "max_update_relative_error": .001,
                "min_update_cosine": .9999, "sinkhorn_distributed": True,
                "resident_resume_exact": True, "distributed_checkpoint_roundtrip": True, "streaming_checkpoint": True, "destructive_restore_verified": True}))
        with pytest.raises(FileNotFoundError):
            verify(root)


@pytest.mark.parametrize("error,cosine,dtype", [
    (float("nan"), 1., "float32"),
    (0., float("nan"), "float32"),
    (0., 1., "bfloat16"),
])
def test_nonfinite_or_wrong_precision_receipts_are_rejected(error, cosine, dtype):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "fp32-streaming-restore-admission").mkdir()
        (root / "fp32-streaming-restore-admission/rank-0.json").write_text(json.dumps({
            "rank": 0, "momentum_dtype": dtype, "passed": True,
            "max_update_relative_error": error, "min_update_cosine": cosine,
            "sinkhorn_distributed": True, "resident_resume_exact": True, "distributed_checkpoint_roundtrip": True, "streaming_checkpoint": True, "destructive_restore_verified": True}))
        with pytest.raises(ValueError, match="numerical admission rejected"):
            verify(root)
