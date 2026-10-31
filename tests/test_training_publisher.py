import hashlib
import json

import pytest

from archlab.training.publisher import verify_publisher


@pytest.fixture
def native_publisher(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    payloads = {"config.json": {}, "generation_config.json": {}, "tokenizer.json": {},
                "tokenizer_config.json": {}, "chat_template.jinja": "native template",
                "model.safetensors.index.json": {"weight_map": {"weight": "weights.safetensors"}},
                "weights.safetensors": "weight payload"}
    files = {}
    for name, value in payloads.items():
        data = (json.dumps(value) if isinstance(value, dict) else value).encode()
        (root / name).write_bytes(data)
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    record = {"schema_version": 1, "model_id": "publisher/model",
              "publisher_revision": "a" * 40, "assets": files}
    manifest = tmp_path / "publisher.json"
    manifest.write_text(json.dumps(record))
    return root, manifest, record


def test_native_publisher_binds_all_weights_and_assets(native_publisher):
    root, manifest, _ = native_publisher
    result = verify_publisher(root, manifest, model_id="publisher/model")
    assert result["shards"] == 1 and len(result["files"]) == 7
    assert result["revision"] == "a" * 40


@pytest.mark.parametrize("name", ["config.json", "chat_template.jinja", "weights.safetensors"])
def test_changed_publisher_payload_is_rejected(native_publisher, name):
    root, manifest, _ = native_publisher
    data = (root / name).read_bytes()
    (root / name).write_bytes(b"x" + data[1:])
    with pytest.raises(ValueError, match="bytes changed"):
        verify_publisher(root, manifest, model_id="publisher/model")


@pytest.mark.parametrize("defect", ["revision", "repo", "missing", "path", "uncovered"])
def test_incomplete_or_unpinned_manifest_is_rejected(native_publisher, defect):
    root, manifest, record = native_publisher
    if defect == "revision":
        record["publisher_revision"] = "main"
    elif defect == "repo":
        record["model_id"] = "another/model"
    elif defect == "missing":
        del record["assets"]["tokenizer_config.json"]
    elif defect == "path":
        record["assets"]["../outside"] = record["assets"]["config.json"]
    else:
        del record["assets"]["weights.safetensors"]
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        verify_publisher(root, manifest, model_id="publisher/model")


def test_shared_weights_require_the_recorded_symlink_target(native_publisher, tmp_path):
    root, manifest, record = native_publisher
    weight = root / "weights.safetensors"
    shared = tmp_path / "shared.safetensors"
    weight.rename(shared)
    weight.symlink_to(shared)
    with pytest.raises(ValueError, match="symlink target"):
        verify_publisher(root, manifest, model_id="publisher/model")
    record["assets"][weight.name]["target"] = str(shared)
    manifest.write_text(json.dumps(record))
    assert verify_publisher(root, manifest, model_id="publisher/model")["shards"] == 1


def test_post_training_state_cannot_masquerade_as_publisher(native_publisher):
    root, manifest, _ = native_publisher
    (root / "rl_state.pt").touch()
    with pytest.raises(ValueError, match="experiment state"):
        verify_publisher(root, manifest, model_id="publisher/model")
