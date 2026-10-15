import json
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from archlab.training import dlc_sandbox

ROOT = Path(os.environ.get("ARCHLAB_DLC_SANDBOX_IMAGES", "/mnt/nas/evergreen/sandbox-images/mimo-v2.6-rl-oss"))
IMAGE = "xiaomimimo/mimo-v2.6-rl-oss@sha256:4fb3e32d0e57e2c2f5b8819202ed4c5bf622e8b40adb5c787aead7f9885efa61"


@pytest.fixture
def fake_server(tmp_path):
    """Scripted pool server: replies to POST /sandboxes from a queue; records every request."""
    state = {"create": [], "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, {})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            state["requests"].append((self.path, body))
            if self.path == "/sandboxes":
                status = state["create"].pop(0)
                return self.reply(status, {"id": "box"} if status == 200 else {"error": "insufficient resources"})
            if self.path.endswith("/exec"):
                return self.reply(200, {"output": "ok\n", "returncode": 3})
            self.reply(200, {})

        def do_DELETE(self):
            state["requests"].append((self.path, b""))
            self.reply(200, {})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "a.json").write_text(json.dumps({"url": f"http://127.0.0.1:{server.server_port}"}))
    (pool / "gone.json").write_text(json.dumps({"url": "http://127.0.0.1:9"}))  # stale announcement
    yield state, pool
    server.shutdown()


def test_capacity_refusals_wait_then_run(fake_server, monkeypatch):
    state, pool = fake_server
    state["create"] = [503, 503, 200]
    monkeypatch.setattr(dlc_sandbox.time, "sleep", lambda seconds: None)
    env = dlc_sandbox.DLCEnvironment(image=IMAGE, pool_dir=str(pool))
    env.start()
    assert env.execute("false", timeout=5) == {"output": "ok\n", "returncode": 3}
    exec_request = json.loads(state["requests"][-1][1])
    assert exec_request == {"command": "false", "cwd": "/testbed", "timeout": 5}
    env.cleanup()
    assert state["requests"][-1][0] == "/sandboxes/box" and env.url is None


def test_capacity_wait_is_bounded_and_other_errors_fail(fake_server, monkeypatch):
    from mimoagent.environments import TransportError

    state, pool = fake_server
    monkeypatch.setattr(dlc_sandbox.time, "sleep", lambda seconds: None)
    state["create"] = [503]
    with pytest.raises(TransportError, match="no capacity"):
        dlc_sandbox.DLCEnvironment(image=IMAGE, pool_dir=str(pool), capacity_wait_seconds=0).start()
    state["create"] = [500, 200]
    with pytest.raises(TransportError):
        dlc_sandbox.DLCEnvironment(image=IMAGE, pool_dir=str(pool)).start()
    assert state["create"] == [200]


def test_directory_transfer_creates_exact_destination(fake_server, tmp_path):
    state, pool = fake_server
    state["create"] = [200]
    source = tmp_path / "edit"
    (source / "sub").mkdir(parents=True)
    (source / "old").write_text("a")
    (source / "sub" / "new").write_text("b")
    env = dlc_sandbox.DLCEnvironment(image=IMAGE, pool_dir=str(pool))
    env.start()
    env.copy_to(str(source), "/tmp/edit")
    paths = [path for path, _ in state["requests"][1:]]
    assert paths == ["/sandboxes/box/dirs?path=/tmp/edit", "/sandboxes/box/dirs?path=/tmp/edit/sub",
                     "/sandboxes/box/files?path=/tmp/edit/old", "/sandboxes/box/files?path=/tmp/edit/sub/new"]
    (source / "link").symlink_to(source / "old")
    with pytest.raises(ValueError):
        env.copy_to(str(source), "/tmp/edit")


def _namespaces_available():
    probe = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "true"], capture_output=True)
    return probe.returncode == 0 and (ROOT / "exports" / f"{IMAGE.split('@')[1]}.tar.gz").exists()


@pytest.mark.skipif(not shutil.which("nsenter") or not _namespaces_available(),
                    reason="needs user namespaces and the exported task image")
def test_live_sandbox_isolation(tmp_path):
    pool = dlc_sandbox.Pool(ROOT, tmp_path / "scratch", slots=2)
    first, second = pool.create(IMAGE), pool.create(IMAGE)
    assert pool.create(IMAGE) is None  # full pod refuses like an E2B pool (HTTP 503)
    try:
        result = pool.execute(first, 'echo $PIP_NO_INDEX; id -u; pwd; git status --porcelain | wc -l', "/testbed", 60)
        assert result == {"output": "1\n0\n/testbed\n0\n", "returncode": 0}
        offline = pool.execute(first, "python -c \"import socket; socket.create_connection(('1.1.1.1', 53), 3)\"",
                               "/", 60)
        assert offline["returncode"] != 0 and "unreachable" in offline["output"]
        pool.write(first, "/testbed/new/file.txt", b"payload", False)
        assert pool.execute(first, "cat new/file.txt", "/testbed", 30)["output"] == "payload"
        assert pool.execute(second, "test -e new/file.txt", "/testbed", 30)["returncode"] == 1
        recreate = pool.execute(first, "rm -rf .git/hooks && mkdir .git/hooks && touch .git/hooks/x", "/testbed", 30)
        assert recreate["returncode"] == 0, recreate  # opaque directory over a deleted lower one
        expired = pool.execute(first, "echo started; sleep 30 & sleep 30", "/", 1)
        assert expired == {"output": "started\n\nCommand timed out (pod) after 1s", "returncode": 124}
        for _ in range(50):  # killed background processes are reaped by the anchor within its 1 s poll
            if not pool.members(first):
                break
            time.sleep(0.1)
        assert pool.members(first) == []
        anchor = pool.sandboxes[first]["init"]
    finally:
        pool.kill(first)
        pool.kill(second)
    assert not Path(f"/proc/{anchor}").exists() and not list((tmp_path / "scratch" / "sandboxes").iterdir())


@pytest.mark.skipif(not shutil.which("nsenter") or not _namespaces_available(),
                    reason="needs user namespaces and the exported task image")
def test_live_client_through_server(tmp_path):
    pool = dlc_sandbox.Pool(ROOT, tmp_path / "scratch", slots=1)
    server = dlc_sandbox.make_server(pool, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    announce = tmp_path / "pool"
    announce.mkdir()
    (announce / "pod.json").write_text(json.dumps({"url": f"http://127.0.0.1:{server.server_port}"}))
    source = tmp_path / "edit"
    (source / "sub").mkdir(parents=True)
    (source / "sub" / "new.py").write_text("x = 1\n")
    env = dlc_sandbox.DLCEnvironment(image=IMAGE, pool_dir=str(announce))
    try:
        env.start()
        env.copy_to(str(source / "sub" / "new.py"), "/tmp/patch.diff")
        env.copy_to(str(source), "/testbed/edit/")
        result = env.execute("cat /tmp/patch.diff edit/sub/new.py; exit 7")
        assert result == {"output": "x = 1\nx = 1\n", "returncode": 7}
    finally:
        env.cleanup()
        server.shutdown()
    assert pool.sandboxes == {}


def test_rootfs_eviction_is_lru_and_skips_pinned(tmp_path, monkeypatch):
    pool = dlc_sandbox.Pool(tmp_path / "images", tmp_path / "scratch", slots=1, reserve_fraction=0.5)
    rootfs = tmp_path / "scratch" / "rootfs"
    for age, name in enumerate(("sha256:aa", "sha256:bb", "sha256:cc")):
        (rootfs / name.split(":")[1]).mkdir(parents=True)
        (rootfs / name.split(":")[1] / "blob").write_bytes(b"x" * 100)
        marker = rootfs / f"{name}.complete"
        marker.touch()
        os.utime(marker, (age, age))  # aa least recently used
    pool.pinned["sha256:aa"] = 1
    usage = shutil._ntuple_diskusage(total=1000, used=850, free=150)
    monkeypatch.setattr(dlc_sandbox.shutil, "disk_usage", lambda path: usage)
    pool.evict()  # needs 500 free: evicts bb (+100) then cc (+100), never the pinned aa
    assert sorted(p.name for p in rootfs.glob("*.complete")) == ["sha256:aa.complete"]
    assert (rootfs / "aa").exists() and not (rootfs / "bb").exists()


def test_image_spec_filters_injected_and_secret_names(tmp_path):
    (tmp_path / "exports").mkdir()
    (tmp_path / "exports" / "sha256:aa.env").write_bytes(b"\0".join([
        b"PATH=/usr/local/cargo/bin:/usr/bin", b"CARGO_HOME=/usr/local/cargo", b"KUBERNETES_PORT=tcp://x",
        b"PAI_WORKSPACE_ID=270", b"RANK=0", b"HOSTNAME=pod", b"OSS_ACCESS_KEY_ID=x", b"API_TOKEN=y",
        b"LANG=C.UTF-8", b"MULTI=a=b"]) + b"\0")
    assert dlc_sandbox.image_spec(tmp_path, "sha256:aa")["env"] == [
        "PATH=/usr/local/cargo/bin:/usr/bin", "CARGO_HOME=/usr/local/cargo", "LANG=C.UTF-8", "MULTI=a=b"]
    with pytest.raises(FileNotFoundError):
        dlc_sandbox.image_spec(tmp_path, "sha256:bb")  # no record: refused, never guessed


def test_client_ignores_environment_proxy_and_skips_bad_servers(fake_server, monkeypatch):
    state, pool = fake_server
    state["create"] = [200]
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # a proxy that would refuse every request
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    (pool / "proxy-like.json").write_text(json.dumps({"url": "http://127.0.0.1:9/not-a-pool"}))
    env = dlc_sandbox.DLCEnvironment(image=IMAGE, pool_dir=str(pool))
    env.start()
    assert env.url.endswith("/sandboxes/box")
