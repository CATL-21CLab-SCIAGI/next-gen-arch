"""MiMo's Environment contract on CPU DLC pods: rootless namespace sandboxes over exported task images.

Contract
- images: ``<root>/exports/<digest>.tar.gz`` is the root filesystem of a digest-pinned task image
  (exported on DLC from the in-VPC registry mirror and checked against the public layer);
  ``<root>/images/<digest>.json`` holds its Env/WorkingDir, fetched by ``configs`` from the public
  registry with manifest and config digests verified. Nothing is built or modified.
- serve: one HTTP server per CPU pod. Each sandbox is ``unshare --user --map-root-user --mount
  --pid --net`` with an overlay (image rootfs lower, private tmpfs upper) as its root; it has no
  network beyond loopback, matching E2B ``allow_internet_access=False``. Commands run via ``nsenter``
  with the image environment plus the offline pip/uv variables. A full pod answers 503 like E2B's
  pool. Servers announce ``http://ip:port`` and capacity in ``<pool>/<host>.json``.
- DLCEnvironment: the client. Transport failures raise TransportError (never a test verdict); an
  expired command returns code 124 with its partial output, as upstream's Kubernetes environment does;
  capacity refusals wait within ``capacity_wait_seconds`` exactly like E2BEnvironment.
usage: python -m archlab.training.dlc_sandbox configs <template-map.json> <root>
       python -m archlab.training.dlc_sandbox serve <root> <pool-dir> [--slots N] [--port P]
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from archlab.artifacts import transfer_plan

# Namespace anchor and pid-1 reaper: commands entered with nsenter are re-parented to it when their
# nsenter dies (e.g. a timeout kill); without reaping, the namespace could never finish exiting.
ANCHOR = ("import os, sys, time\nos.chroot(sys.argv[1]); os.chdir('/')\nwhile True:\n"
          "    try:\n        os.waitpid(-1, 0)\n    except ChildProcessError:\n        time.sleep(1)\n")
OFFLINE_ENV = {"PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "UV_OFFLINE": "1"}
DEVICES = ("null", "zero", "full", "random", "urandom", "tty")  # bound into each sandbox's private /dev
ACCEPT = "application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json"


# ---------------------------------------------------------------------------------------- configs
def registry_get(image_name, path, accept=None):
    token_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{image_name}:pull"
    headers = {"Authorization": "Bearer " + json.load(urllib.request.urlopen(token_url, timeout=60))["token"]}
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(f"https://registry-1.docker.io/v2/{image_name}/{path}", headers=headers)
    return urllib.request.urlopen(request, timeout=120).read()


def configs(map_path, root):
    (Path(root) / "images").mkdir(parents=True, exist_ok=True)
    for image in sorted(json.loads(Path(map_path).read_text())["images"]):
        name, digest = image.split("@")
        record = Path(root) / "images" / f"{digest}.json"
        if record.exists():
            continue
        raw = registry_get(name, f"manifests/{digest}", accept=ACCEPT)
        manifest = json.loads(raw)
        config_raw = registry_get(name, f"blobs/{manifest['config']['digest']}")
        if ("sha256:" + hashlib.sha256(raw).hexdigest() != digest
                or "sha256:" + hashlib.sha256(config_raw).hexdigest() != manifest["config"]["digest"]):
            raise RuntimeError(f"registry digest mismatch: {image}")
        config = json.loads(config_raw)["config"]
        record.write_text(json.dumps({"image": image, "layers": manifest["layers"], "env": config.get("Env") or [],
                                      "workdir": config.get("WorkingDir") or "/"}, indent=1) + "\n")


# Names the DLC/Kubernetes runtime adds to an export pod's environment; never part of an image's Env.
INJECTED = re.compile(r"^(KUBERNETES_|PAI_|DLC_|PET_|POD_|NCCL_|NVSHMEM_|GLOO_|MASTER_ADDR$|MASTER_PORT$|RANK$|"
                      r"WORLD_SIZE|NPROC_PER_NODE$|REPLICA_TYPE$|REGION$|CLUSTER_NAME$|SCRAPE_PROMETHEUS_METRICS$|"
                      r"KATA_CONTAINER$|NVIDIA_VISIBLE_DEVICES$|VF_Alloc$|USE_ALIBABACLOUD_GPU$|HOSTNAME$|HOME$|"
                      r"SHLVL$|PWD$|OLDPWD$|_$)")
SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH", re.IGNORECASE)


def image_spec(root, digest):
    """Env/WorkingDir of an image: its registry config if fetched, else the Env captured on DLC.

    The captured Env is the export pod's process environment minus injected and secret-like names.
    An image with neither record is refused rather than run with a guessed environment.
    """
    record = Path(root) / "images" / f"{digest}.json"
    if record.exists():
        return json.loads(record.read_text())
    captured = Path(root) / "exports" / f"{digest}.env"
    if not captured.exists():
        raise FileNotFoundError(f"no Env record for {digest}")
    entries = [item for item in captured.read_bytes().decode(errors="replace").split("\0") if "=" in item]
    env = [item for item in entries if not INJECTED.match(item.split("=", 1)[0])
           and not SECRET.search(item.split("=", 1)[0])]
    return {"env": env, "workdir": "/"}


# ------------------------------------------------------------------------------------------ serve
class UnpackError(RuntimeError):
    """This pod could not unpack an image (e.g. scratch disk full); the client should try another pod."""


class Pool:
    def __init__(self, root, scratch, slots, upper_size="16g", reserve_fraction=0.25):
        self.root, self.scratch, self.slots, self.upper_size = Path(root), Path(scratch), slots, upper_size
        self.reserve_fraction = reserve_fraction
        self.sandboxes, self.lock, self.unpack_locks, self.pinned = {}, threading.Lock(), {}, {}

    def rootfs(self, digest):
        """Unpack an exported image on first use; later sandboxes share it as their read-only lower layer.

        Unpacked images are evicted least-recently-used (never one pinned by a live or starting sandbox)
        when free scratch space falls below ``reserve_fraction``.
        """
        spec = image_spec(self.root, digest)
        target = self.scratch / "rootfs" / digest.split(":")[1]  # ":" separates overlay lowerdirs
        marker = self.scratch / "rootfs" / f"{digest}.complete"
        with self.lock:
            lock = self.unpack_locks.setdefault(digest, threading.Lock())
        with lock:
            if marker.exists():
                os.utime(marker)
            else:
                self.evict()
                shutil.rmtree(target, ignore_errors=True)
                target.mkdir(parents=True)
                result = subprocess.run(["tar", "--numeric-owner", "-xpzf",
                                         str(self.root / "exports" / f"{digest}.tar.gz"), "-C", str(target)],
                                        capture_output=True)
                if result.returncode:
                    # A partial tree would keep the scratch disk full for every later unpack on this pod.
                    shutil.rmtree(target, ignore_errors=True)
                    raise UnpackError(f"{digest}: tar exit {result.returncode}: "
                                      f"{result.stderr.decode(errors='replace')[-300:]}")
                for mountpoint in ("proc", "sys", "dev", "mnt", "tmp", "run"):
                    (target / mountpoint).mkdir(exist_ok=True)
                marker.touch()
        return target, spec

    def evict(self):
        (self.scratch / "rootfs").mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self.scratch)
        with self.lock:
            unpinned = sorted((m for m in (self.scratch / "rootfs").glob("*.complete")
                               if not self.pinned.get(m.name.removesuffix(".complete"))),
                              key=lambda m: m.stat().st_mtime)
            free = usage.free
            for marker in unpinned:
                if free >= self.reserve_fraction * usage.total:
                    break
                digest = marker.name.removesuffix(".complete")
                target = self.scratch / "rootfs" / digest.split(":")[1]
                size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file() and not f.is_symlink())
                marker.unlink()
                shutil.rmtree(target, ignore_errors=True)
                free += size

    def create(self, image, lifetime=7200.0):
        digest = image.split("@")[1]
        with self.lock:
            if len(self.sandboxes) >= self.slots:
                return None
            sandbox_id = uuid.uuid4().hex[:16]
            self.sandboxes[sandbox_id] = None  # reserve the slot while unpacking
            self.pinned[digest] = self.pinned.get(digest, 0) + 1
        try:
            lower, spec = self.rootfs(digest)
            work = self.scratch / "sandboxes" / sandbox_id
            for part in ("rw", "merged"):
                (work / part).mkdir(parents=True)
            # overlayfs rejects an upper layer on the pod's own overlay root: the private layer is a tmpfs.
            # userxattr: in a user namespace, whiteouts/opaque directories (delete then recreate a lower
            # directory) need user.overlay.* xattrs; trusted.* ones fail with EIO.
            # A fresh proc mount is refused under the pod's masked /proc, so it is bound; the pid namespace
            # still makes killing the anchor end every process the sandbox started.
            # /dev is private: the sandbox's root is the pod's root (--map-root-user), so a shared /dev let one
            # task replace the pod's /dev/null for every later sandbox (2026-10-16). Bound device nodes cannot be
            # removed (EBUSY); anything else written to this /dev stays in the sandbox.
            dev = f"{work}/merged/dev"
            nodes = "; ".join(f"touch {dev}/{node}; mount --bind /dev/{node} {dev}/{node}" for node in DEVICES)
            setup = (f"set -e; mount -t tmpfs -o size={self.upper_size} tmpfs {work}/rw; "
                     f"mkdir {work}/rw/upper {work}/rw/work; "
                     f"mount -t overlay overlay -o userxattr,lowerdir={lower},upperdir={work}/rw/upper,workdir={work}/rw/work "
                     f"{work}/merged; mount -t tmpfs -o size=1m,mode=755 tmpfs {dev}; {nodes}; "
                     f"mkdir {dev}/pts {dev}/shm; mount -t devpts -o newinstance,ptmxmode=0666,mode=620 devpts "
                     f"{dev}/pts || mount --rbind /dev/pts {dev}/pts; mount -t tmpfs tmpfs {dev}/shm; "
                     f"ln -s pts/ptmx {dev}/ptmx; ln -s /proc/self/fd {dev}/fd; ln -s /proc/self/fd/0 {dev}/stdin; "
                     f"ln -s /proc/self/fd/1 {dev}/stdout; ln -s /proc/self/fd/2 {dev}/stderr; "
                     f"mount --rbind /proc {work}/merged/proc; "
                     f"mount -t tmpfs tmpfs {work}/merged/tmp; ip link set lo up || true; "
                     f"exec {shlex.quote(sys.executable)} -c {shlex.quote(ANCHOR)} {work}/merged")
            with (work / "setup.log").open("wb") as log:  # never inherit the server's stdout/stderr
                process = subprocess.Popen(["unshare", "--user", "--map-root-user", "--mount", "--pid", "--net",
                                            "--fork", "--kill-child", "sh", "-c", setup], stdin=subprocess.DEVNULL,
                                           stdout=subprocess.DEVNULL, stderr=log, start_new_session=True)
            init = None
            for _ in range(200):  # the chrooted reaper is the namespace anchor for nsenter
                children = subprocess.run(["pgrep", "-P", str(process.pid)], capture_output=True, text=True).stdout
                if children.split():
                    candidate = children.split()[0]
                    if Path(f"/proc/{candidate}/root").resolve() == (work / "merged").resolve():
                        init = int(candidate)
                        break
                if process.poll() is not None:
                    break
                time.sleep(0.05)
            if init is None:
                process.kill()
                process.wait()
                detail = (work / "setup.log").read_text(errors="replace")[-300:]
                shutil.rmtree(work, ignore_errors=True)
                raise RuntimeError(f"sandbox namespace failed to start: {detail}")
            env = dict(item.split("=", 1) for item in spec["env"])
            env.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
            env.update(OFFLINE_ENV, HOME="/root")
            with self.lock:
                self.sandboxes[sandbox_id] = {"process": process, "init": init, "work": work, "env": env,
                                              "digest": digest,
                                              "deadline": time.monotonic() + lifetime}
            return sandbox_id
        except Exception as error:
            with self.lock:
                self.sandboxes.pop(sandbox_id, None)
                self.pinned[digest] -= 1
            if isinstance(error, UnpackError):
                print(f"unpack failed, refusing like a full pod: {error}", file=sys.stderr, flush=True)
                return None
            raise

    def enter(self, sandbox_id):
        box = self.sandboxes[sandbox_id]
        return ["nsenter", "-t", str(box["init"]), "--user", "--mount", "--pid", "--net", "--root", "--wd",
                "/usr/bin/env", "-i", *(f"{k}={v}" for k, v in box["env"].items())]

    def execute(self, sandbox_id, command, cwd, timeout):
        # A login shell, as E2B's envd runs commands: the squashed task images carry no Env of their own.
        # nsenter resolves --wd outside the sandbox root, so the shell changes directory itself.
        script = f"cd -- {shlex.quote(cwd or '/')} || exit 1\n{command}"
        process = subprocess.Popen([*self.enter(sandbox_id), "/bin/bash", "-l", "-c", script], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill inside the sandbox only. nsenter must live to reap its child: an orphan is re-parented
            # through nsenter's (the pod's) pid namespace to the pod's non-reaping pid 1, and that zombie
            # would keep the sandbox namespace from ever finishing its exit.
            for pid in self.members(sandbox_id):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            output, _ = process.communicate()  # includes what the command printed before the deadline
            # Upstream's reference environment (kubernetes.py, pod_timeout): an expired command is an ordinary
            # observation with returncode 124, not a transport failure; the agent continues.
            return {"output": output.decode(errors="replace") + f"\nCommand timed out (pod) after {timeout}s",
                    "returncode": 124}
        return {"output": output.decode(errors="replace"), "returncode": process.returncode}

    def members(self, sandbox_id):
        """Processes in the sandbox's pid namespace other than its anchor."""
        init = self.sandboxes[sandbox_id]["init"]
        namespace, pids = os.readlink(f"/proc/{init}/ns/pid"), []
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit() and int(entry.name) != init:
                try:
                    if os.readlink(entry / "ns" / "pid") == namespace:
                        pids.append(int(entry.name))
                except OSError:
                    pass
        return pids

    def write(self, sandbox_id, path, data, directory):
        prefix = self.enter(sandbox_id)
        script = 'mkdir -p "$1"' if directory else 'mkdir -p "$(dirname "$1")" && cat > "$1"'
        result = subprocess.run([*prefix, "/bin/sh", "-c", script, "sh", path], input=data, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace")[-500:])

    def kill(self, sandbox_id):
        with self.lock:
            box = self.sandboxes.pop(sandbox_id, None)
            if box:
                self.pinned[box["digest"]] -= 1
        if box:
            # Killing the pid-namespace init ends every process in the sandbox (including ones entered with
            # nsenter); unshare must stay alive to reap it, or it lingers as a zombie under the pod's pid 1.
            try:
                os.kill(box["init"], signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                box["process"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                box["process"].kill()
                box["process"].wait()
            shutil.rmtree(box["work"], ignore_errors=True)


def make_server(pool, port):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body=None):
            data = json.dumps(body or {}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def do_POST(self):
            url = urllib.parse.urlparse(self.path)
            parts = url.path.strip("/").split("/")
            try:
                if parts == ["sandboxes"]:
                    request = json.loads(self.body())
                    sandbox_id = pool.create(request["image"], float(request.get("lifetime", 7200)))
                    if sandbox_id is None:
                        return self.reply(503, {"error": "insufficient resources"})
                    return self.reply(200, {"id": sandbox_id})
                if len(parts) == 3 and parts[2] == "exec":
                    request = json.loads(self.body())
                    return self.reply(200, pool.execute(parts[1], request["command"], request["cwd"],
                                                        request["timeout"]))
                if len(parts) == 3 and parts[2] in ("files", "dirs"):
                    path = urllib.parse.parse_qs(url.query)["path"][0]
                    pool.write(parts[1], path, self.body(), parts[2] == "dirs")
                    return self.reply(200)
            except KeyError:
                return self.reply(404, {"error": "unknown sandbox"})
            except Exception as error:
                return self.reply(500, {"error": f"{type(error).__name__}: {error}"[:500]})
            return self.reply(404)

        def do_DELETE(self):
            pool.kill(self.path.strip("/").split("/")[1])
            self.reply(200)

        def do_GET(self):
            self.reply(200, {"slots": pool.slots, "used": len(pool.sandboxes)})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    return server


def serve(root, pool_dir, slots, port, scratch):
    pool = Pool(root, scratch, slots)

    def expire():  # sandboxes a crashed client never cleaned up give their slots back, like E2B's timeout
        while True:
            time.sleep(30)
            for sandbox_id, box in list(pool.sandboxes.items()):
                if box is not None and time.monotonic() > box["deadline"]:
                    pool.kill(sandbox_id)

    threading.Thread(target=expire, daemon=True).start()
    server = make_server(pool, port)
    host = socket.gethostname()
    address = socket.gethostbyname(host)
    announce = Path(pool_dir) / f"{host}.json"
    announce.parent.mkdir(parents=True, exist_ok=True)
    announce.write_text(json.dumps({"url": f"http://{address}:{port}", "slots": slots}) + "\n")
    try:
        server.serve_forever()
    finally:
        announce.unlink(missing_ok=True)


# ----------------------------------------------------------------------------------------- client
def _transport(operation, detail):
    from mimoagent.environments import TransportError

    return TransportError(f"DLC sandbox {operation} infrastructure failure: {detail}")


# Pool endpoints are private pod addresses: never route them through an environment HTTP proxy.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _call(method, url, body=b"", timeout=60, content_type="application/json"):
    request = urllib.request.Request(url, data=body if method in ("POST", "PUT") else None, method=method,
                                     headers={"Content-Type": content_type})
    try:
        with _DIRECT.open(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw or b"{}")
        except ValueError:  # an intermediary (not the pool server) answered
            return error.code, {"error": raw[:200].decode(errors="replace")}


def _affinity_order(image, servers):
    """Rendezvous order: each image prefers the same few pool pods, falling back through all of them.

    A pod unpacks an image on first use and keeps only what fits its scratch disk, so random placement makes
    every pod unpack every task image (minutes each, and concurrent unpacks on a cold pod can fail). The order
    depends only on (image, server), so pods joining or leaving move few images.
    """
    return sorted(servers, key=lambda server: hashlib.sha256(f"{image}\0{server}".encode()).digest(), reverse=True)


@dataclass
class DLCEnvironmentConfig:
    image: str
    cwd: str = "/testbed"
    timeout: int = 300
    sandbox_timeout: int = 7200
    pool_dir: str = field(default_factory=lambda: os.environ.get("ARCHLAB_DLC_SANDBOX_POOL", ""))
    capacity_wait_seconds: float = 1800.0


class DLCEnvironment:
    def __init__(self, *, config_class=DLCEnvironmentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.url = None

    def start(self):
        if self.url is not None:
            raise RuntimeError("sandbox already started")
        deadline, delay = time.monotonic() + self.config.capacity_wait_seconds, 5.0
        body = json.dumps({"image": self.config.image, "lifetime": self.config.sandbox_timeout}).encode()
        while True:
            servers = [json.loads(p.read_text())["url"] for p in Path(self.config.pool_dir).glob("*.json")]
            for server in _affinity_order(self.config.image, servers):
                try:
                    # A stale or unreachable announcement must neither stall nor fail creation.
                    if _call("GET", server, timeout=5)[0] != 200:
                        continue
                    status, reply = _call("POST", f"{server}/sandboxes", body, timeout=900)
                except OSError:
                    continue  # a pod that is gone or restarting is skipped like a full one
                if status == 200:
                    self.url = f"{server}/sandboxes/{reply['id']}"
                    return
                if status != 503:
                    raise _transport("create", reply.get("error", status)) from None
            if time.monotonic() + delay >= deadline:
                raise _transport("create", "no capacity") from None
            time.sleep(delay * (0.5 + random.random()))
            delay = min(delay * 2, 60.0)

    def execute(self, command, cwd="", timeout=None):
        if self.url is None:
            raise RuntimeError("sandbox is not started")
        timeout = timeout or self.config.timeout
        request = json.dumps({"command": command, "cwd": cwd or self.config.cwd, "timeout": timeout}).encode()
        try:
            status, reply = _call("POST", f"{self.url}/exec", request, timeout=timeout + 60)
        except OSError as error:
            raise _transport("command", type(error).__name__) from None
        if status != 200:
            raise _transport("command", reply.get("error", status)) from None
        return reply

    def execute_detached(self, command, cwd="", timeout=None, **_):
        return self.execute(command, cwd=cwd, timeout=timeout)

    def copy_to(self, src_path, dest_path, *, timeout=300, max_retries=10):
        """Same transfer contract as E2BEnvironment.copy_to: regular files and directories only."""
        if self.url is None:
            raise RuntimeError("sandbox is not started")
        directories, files = transfer_plan(src_path, dest_path)
        uploads = [("dirs", str(d), b"") for d in directories] + [("files", d, p.read_bytes()) for p, d in files]
        for kind, path, data in uploads:
            try:
                status, reply = _call("POST", f"{self.url}/{kind}?path={urllib.parse.quote(path)}", data,
                                      timeout=timeout, content_type="application/octet-stream")
            except OSError as error:
                raise _transport("file transfer", type(error).__name__) from None
            if status != 200:
                raise _transport("file transfer", reply.get("error", status)) from None

    def get_template_vars(self):
        return {"cwd": self.config.cwd, "image": self.config.image}

    def cleanup(self):
        if self.url is not None:
            try:
                _call("DELETE", self.url, timeout=120)
            except OSError as error:
                raise _transport("cleanup", type(error).__name__) from None
            self.url = None


def main():
    action = sys.argv[1]
    if action == "configs":
        configs(sys.argv[2], sys.argv[3])
    elif action == "serve":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("root")
        parser.add_argument("pool_dir")
        parser.add_argument("--slots", type=int, default=max(1, os.cpu_count() // 4))
        parser.add_argument("--port", type=int, default=8765)
        parser.add_argument("--scratch", default="/tmp/archlab-dlc-sandbox")
        args = parser.parse_args(sys.argv[2:])
        serve(args.root, args.pool_dir, args.slots, args.port, args.scratch)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
