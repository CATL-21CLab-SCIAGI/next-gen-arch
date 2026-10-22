"""MiMo Environment client for the private CPU DLC sandbox pool.

The pool owns isolated execution and server lifecycle. This module only discovers
qualified servers, transfers task files, and reports transport errors separately
from process results. The immutable pool deployment owns server code.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from archlab.artifacts import transfer_plan


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

