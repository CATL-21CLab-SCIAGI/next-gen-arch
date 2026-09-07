"""CPU regressions for frozen-runtime configuration and checkpoint safety."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from archlab.automodel.runtime import restrict_autotuner


def test_restrict_existing_configs_and_clear_unsafe_cache():
    configs = [SimpleNamespace(num_warps=w, num_stages=s, kwargs={"BV": 32})
               for w in (2, 4) for s in (2, 3, 4)]
    tuner = SimpleNamespace(configs=configs, cache={"old": configs[-1]}, cache_results=True)
    wrapped = SimpleNamespace(fn=SimpleNamespace(fn=tuner))
    evidence = restrict_autotuner(wrapped, num_warps=2, num_stages=4)
    assert evidence["before_count"] == 6 and evidence["after_count"] == 1
    assert tuner.configs == [configs[2]]  # retain the existing Config object
    assert tuner.cache == {} and not tuner.cache_results
    assert restrict_autotuner(wrapped, num_warps=2, num_stages=4)["after_count"] == 1


def test_reject_missing_safe_configs_and_unrecognized_wrapper():
    with pytest.raises(RuntimeError, match="wrapper"):
        restrict_autotuner(object(), num_warps=2)
    tuner = SimpleNamespace(configs=[SimpleNamespace(num_warps=4, num_stages=2)])
    with pytest.raises(RuntimeError, match="absent"):
        restrict_autotuner(tuner, num_warps=2)


def test_checkpoint_cache_provenance_and_index_drift(tmp_path):
    from archlab.automodel.loading import audit_checkpoint_keys

    model = SimpleNamespace(state_dict=lambda: {},
                            state_dict_adapter=SimpleNamespace(get_hf_state_dict_keys=lambda state: ["w"]))
    index = tmp_path / "model.safetensors.index.json"
    raw = json.dumps({"weight_map": {"w": "weights.safetensors"}}).encode()
    index.write_bytes(raw)
    cache = tmp_path / "ARCHLAB_VERIFIED_COPY.json"
    cache.write_text(json.dumps({"source": "/example/read-only-original",
                                 "source_index_sha256": hashlib.sha256(raw).hexdigest()}))
    evidence = audit_checkpoint_keys(model, tmp_path)
    assert evidence["source_checkpoint"] == "/example/read-only-original"
    assert evidence["cache_manifest_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    index.write_text(json.dumps({"weight_map": {"w": "changed.safetensors"}}))
    with pytest.raises(ValueError, match="no longer matches"):
        audit_checkpoint_keys(model, tmp_path)


def test_fresh_optimizer_checkpoint_load_uses_upstream_materialization(tmp_path, monkeypatch):
    import torch
    import torch.distributed.checkpoint as dcp
    from archlab.automodel.probe import checkpoint_payload, assert_state_equal

    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    module = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4, foreach=False)
    module(torch.ones(2, 4)).square().mean().backward()
    optimizer.step()
    expected = copy.deepcopy(optimizer.state_dict())
    path = tmp_path / "checkpoint"
    dcp.save(checkpoint_payload({"0": module}, optimizer), checkpoint_id=path)
    fresh = torch.optim.AdamW(module.parameters(), lr=1e-4, foreach=False)
    assert fresh.state_dict()["state"] == {}
    payload = checkpoint_payload({"0": module}, fresh)
    assert payload["optimizer"]["state"]
    dcp.load(payload, checkpoint_id=path)
    fresh.load_state_dict(payload["optimizer"])
    assert_state_equal(fresh.state_dict(), expected)
