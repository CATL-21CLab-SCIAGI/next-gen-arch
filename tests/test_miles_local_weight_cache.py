import hashlib
import json

import pytest
import torch

from archlab.megatron.miles_v41_storage import LocalWeightCache


def make_cache(root):
    value = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    raw = value.view(torch.uint8).numpy().tobytes()
    (root / "weights.bin").write_bytes(raw)
    (root / "COMPLETE.json").write_text(json.dumps({
        "bytes": len(raw), "parent": "matched-sft-parent",
        "entries": [{"name": "matrix", "offset": 0, "nbytes": len(raw),
                     "dtype": "torch.float32", "shape": [4, 6],
                     "sha256": hashlib.sha256(raw).hexdigest()}]}))
    return value


def test_local_cache_roundtrip_and_exact_contract(tmp_path):
    expected = make_cache(tmp_path)
    cache = LocalWeightCache(tmp_path)
    target = torch.empty_like(expected)
    cache.validate([("matrix", target)], "matched-sft-parent")
    cache.restore([("matrix", target)])
    torch.testing.assert_close(target, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="different SFT"):
        cache.validate([("matrix", target)], "other-parent")
    with pytest.raises(ValueError, match="inventory"):
        cache.validate([("wrong-name", target)], "matched-sft-parent")
    with pytest.raises(ValueError, match="contract"):
        cache.validate([("matrix", target.bfloat16())], "matched-sft-parent")


def test_local_cache_corruption_is_rejected(tmp_path):
    target = make_cache(tmp_path)
    with (tmp_path / "weights.bin").open("r+b") as stream:
        stream.write(b"\xff")
    cache = LocalWeightCache(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        cache.validate([("matrix", target)], "matched-sft-parent")
