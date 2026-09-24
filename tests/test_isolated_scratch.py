from types import SimpleNamespace

import pytest

from archlab.serving.isolated_sglang_runtime import scratch_binding


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr("archlab.serving.isolated_sglang_runtime.subprocess.check_output", lambda *a, **k: "ext4\n")
    monkeypatch.setattr("archlab.serving.isolated_sglang_runtime.shutil.disk_usage", lambda p: SimpleNamespace(free=100))
    return {"working_directory": str(tmp_path / "project"), "scratch_bind": {
        "source": str(tmp_path / "evergreen-test"), "destination": str(tmp_path / "project/offload"),
        "minimum_free_bytes": 90}}


def test_local_disk_binding(config):
    assert scratch_binding(config) == (config["scratch_bind"]["source"], config["scratch_bind"]["destination"], False)


@pytest.mark.parametrize("filesystem", ["nfs", "nfs4", "tmpfs"])
def test_rejects_network_and_memory_storage(config, monkeypatch, filesystem):
    monkeypatch.setattr("archlab.serving.isolated_sglang_runtime.subprocess.check_output", lambda *a, **k: filesystem)
    with pytest.raises(ValueError, match="local disk"):
        scratch_binding(config)


def test_capacity_gate(config):
    config["scratch_bind"]["minimum_free_bytes"] = 101
    with pytest.raises(ValueError, match="capacity"):
        scratch_binding(config)


def test_cannot_hide_existing_files(config):
    from pathlib import Path
    target = Path(config["scratch_bind"]["destination"])
    target.mkdir(parents=True)
    (target / "checkpoint").touch()
    with pytest.raises(ValueError, match="empty"):
        scratch_binding(config)


def test_cannot_escape_project(config):
    config["scratch_bind"]["destination"] = config["working_directory"] + "/../other"
    with pytest.raises(ValueError, match="project-local"):
        scratch_binding(config)
