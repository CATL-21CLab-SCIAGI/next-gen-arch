"""CPU regressions for frozen-runtime configuration and checkpoint safety."""

import hashlib
import json
from types import SimpleNamespace

import pytest


def test_checkpoint_cache_provenance_and_index_drift(tmp_path):
    from archlab.automodel.qwen.loading import audit_checkpoint_keys

    model = SimpleNamespace(
        state_dict=lambda: {},
        state_dict_adapter=SimpleNamespace(get_hf_state_dict_keys=lambda state: ["w"]),
    )
    index = tmp_path / "model.safetensors.index.json"
    raw = json.dumps({"weight_map": {"w": "weights.safetensors"}}).encode()
    index.write_bytes(raw)
    cache = tmp_path / "ARCHLAB_VERIFIED_COPY.json"
    cache.write_text(
        json.dumps(
            {
                "source": "/example/read-only-original",
                "source_index_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )
    evidence = audit_checkpoint_keys(model, tmp_path)
    assert evidence["source_checkpoint"] == "/example/read-only-original"
    assert evidence["cache_manifest_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    index.write_text(json.dumps({"weight_map": {"w": "changed.safetensors"}}))
    with pytest.raises(ValueError, match="no longer matches"):
        audit_checkpoint_keys(model, tmp_path)
