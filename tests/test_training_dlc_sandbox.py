"""The maintained DLC client preserves the existing pool protocol and transport semantics."""

import json
import sys
import urllib.parse
from types import SimpleNamespace

import pytest

from archlab.training import dlc_sandbox as client


@pytest.fixture
def transport(monkeypatch):
    class TransportError(Exception):
        pass

    monkeypatch.setitem(sys.modules, "mimoagent.environments", SimpleNamespace(TransportError=TransportError))
    return TransportError


def test_affinity_order_is_stable_and_membership_changes_only_remove_one_server():
    servers = ["http://pod1:8765", "http://pod2:8765", "http://pod3:8765"]
    order = client._affinity_order("task@sha256:abc", servers)
    assert order == client._affinity_order("task@sha256:abc", list(reversed(servers)))
    assert [item for item in order if item != servers[1]] == client._affinity_order(
        "task@sha256:abc", [servers[0], servers[2]])


def test_stale_and_full_pool_endpoints_do_not_fail_task_creation(tmp_path, monkeypatch):
    servers = ["http://stale", "http://full", "http://ready"]
    for index, server in enumerate(servers):
        (tmp_path / f"pod-{index}.json").write_text(json.dumps({"url": server}))
    monkeypatch.setattr(client, "_affinity_order", lambda image, endpoints: servers)
    calls = []

    def call(method, url, body=b"", **kwargs):
        calls.append((method, url, body))
        if url == "http://stale":
            raise OSError("pod gone")
        if method == "GET":
            return 200, {"slots": 8, "used": 8 if url.endswith("full") else 0}
        if url.startswith("http://full"):
            return 503, {"error": "insufficient resources"}
        return 200, {"id": "new-task"}

    monkeypatch.setattr(client, "_call", call)
    env = client.DLCEnvironment(image="task@sha256:abc", pool_dir=str(tmp_path))
    env.start()
    assert env.url == "http://ready/sandboxes/new-task"
    assert json.loads(calls[-1][2]) == {"image": "task@sha256:abc", "lifetime": 7200}


def test_command_timeout_is_observation_and_transport_error_is_not_reward(monkeypatch, transport):
    env = client.DLCEnvironment(image="task")
    env.url = "http://pod/sandboxes/id"
    monkeypatch.setattr(client, "_call", lambda *args, **kwargs: (200, {"returncode": 124, "output": "partial"}))
    assert env.execute("sleep 99", timeout=1) == {"returncode": 124, "output": "partial"}
    monkeypatch.setattr(client, "_call", lambda *args, **kwargs: (500, {"error": "namespace unavailable"}))
    with pytest.raises(transport, match="infrastructure failure"):
        env.execute("test")


def test_directory_transfer_preserves_exact_protocol_and_rejects_links(tmp_path, monkeypatch):
    root = tmp_path / "input"
    (root / "empty").mkdir(parents=True)
    (root / "file 雪").write_bytes(b"exact\0bytes\r\n")
    calls = []

    def call(method, url, body=b"", **kwargs):
        calls.append((method, url, body, kwargs))
        return 200, {}

    monkeypatch.setattr(client, "_call", call)
    env = client.DLCEnvironment(image="task")
    env.url = "http://pod/sandboxes/id"
    env.copy_to(root, "/work/editor", timeout=37)
    assert len(calls) == 3
    parsed = [urllib.parse.urlparse(call[1]) for call in calls]
    assert [url.path for url in parsed] == ["/sandboxes/id/dirs", "/sandboxes/id/dirs", "/sandboxes/id/files"]
    assert urllib.parse.parse_qs(parsed[-1].query) == {"path": ["/work/editor/file 雪"]}
    assert calls[-1][2] == b"exact\0bytes\r\n"
    assert calls[-1][3] == {"timeout": 37, "content_type": "application/octet-stream"}
    (root / "link").symlink_to(root / "file 雪")
    calls.clear()
    with pytest.raises(ValueError, match="regular files and directories"):
        env.copy_to(root, "/work/editor")
    assert calls == []
