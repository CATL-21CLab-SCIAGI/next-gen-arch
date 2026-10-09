"""Authenticated OpenAI-compatible chat front end for scheduled GPU windows.

HTTP threads never access model objects. The distributed owner consumes tickets
at a collective boundary and returns text through per-request queues.
"""

from __future__ import annotations

import hmac
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Full, Queue

MODELS = {"deepseek-v41-normal-latest": "normal", "deepseek-v41-simplicial-latest": "simplicial"}


def validate_request(value, *, max_tokens=256):
    if not isinstance(value, dict) or value.get("model") not in MODELS:
        raise ValueError("choose a served DeepSeek checkpoint model")
    messages = value.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) > 64:
        raise ValueError("messages must contain 1–64 text messages")
    clean = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in ("system", "user", "assistant"):
            raise ValueError("this checkpoint endpoint supports text chat without tools")
        if isinstance(content, list):
            if not all(
                isinstance(x, dict) and x.get("type") == "text" and isinstance(x.get("text"), str)
                for x in content
            ):
                raise ValueError("only text content is supported")
            content = "".join(x["text"] for x in content)
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        clean.append({"role": role, "content": content})
    if clean[-1]["role"] != "user":
        raise ValueError("the final message must be from the user")
    limit = value.get("max_completion_tokens", value.get("max_tokens", 128))
    if type(limit) is not int or not 1 <= limit <= max_tokens:
        raise ValueError(f"max_tokens must be between 1 and {max_tokens}")
    temperature = value.get("temperature", 0.0)
    top_p = value.get("top_p", 1.0)
    if (
        not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("temperature must be finite and in 0..2")
    if not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0,1]")
    if value.get("n", 1) != 1 or value.get("tools") or value.get("stop"):
        raise ValueError("only one text completion without tools or custom stops is supported")
    seed = value.get("seed", 42)
    if type(seed) is not int or not -(2**63) <= seed < 2**64:
        raise ValueError("seed must fit the PyTorch generator range")
    return {
        "model": value["model"],
        "variant": MODELS[value["model"]],
        "messages": clean,
        "max_tokens": limit,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": seed,
        "stream": bool(value.get("stream", False)),
    }


@dataclass
class Ticket:
    payload: dict
    request_id: str = field(default_factory=lambda: "chatcmpl-" + uuid.uuid4().hex)
    created: int = field(default_factory=lambda: int(time.time()))
    events: Queue = field(default_factory=Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)


class ChatFront:
    def __init__(self, host, port, token_file, *, max_tokens=256, timeout=600):
        self.token = Path(token_file).read_text().strip()
        if len(self.token) < 24:
            raise ValueError("a private chat API token is required")
        self.max_tokens, self.timeout = max_tokens, timeout
        self.pending = Queue(maxsize=8)
        self.lock = threading.Lock()
        self.state = {"phase": "training", "ready": False}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def authorized(self):
                expected = "Bearer " + owner.token
                if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
                    self.json_response(
                        401, {"error": {"message": "Unauthorized", "type": "authentication_error"}}
                    )
                    return False
                return True

            def json_response(self, status, body):
                data = json.dumps(body, allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if not self.authorized():
                    return
                if self.path in ("/health", "/v1/models"):
                    state = owner.snapshot()
                    if self.path == "/health":
                        self.json_response(200, state)
                    else:
                        self.json_response(
                            200,
                            {
                                "object": "list",
                                "data": [
                                    {
                                        "id": name,
                                        "object": "model",
                                        "owned_by": "archlab",
                                        "created": 0,
                                    }
                                    for name in MODELS
                                ],
                            },
                        )
                else:
                    self.json_response(404, {"error": {"message": "Not found"}})

            def do_POST(self):
                if not self.authorized():
                    return
                if self.path != "/v1/chat/completions":
                    self.json_response(404, {"error": {"message": "Not found"}})
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 65536:
                        raise ValueError("request body must be 1–65536 bytes")
                    payload = validate_request(
                        json.loads(self.rfile.read(size)), max_tokens=owner.max_tokens
                    )
                except (ValueError, TypeError, KeyError, AttributeError) as error:
                    self.json_response(
                        400, {"error": {"message": str(error), "type": "invalid_request_error"}}
                    )
                    return
                if not owner.snapshot()["ready"]:
                    self.json_response(
                        503,
                        {
                            "error": {
                                "message": f"Checkpoint chat is available during scheduled chat windows; current phase: {owner.snapshot()['phase']}.",
                                "type": "model_not_ready",
                            }
                        },
                    )
                    return
                ticket = Ticket(payload)
                try:
                    owner.pending.put_nowait(ticket)
                except Full:
                    self.json_response(429, {"error": {"message": "Chat queue is full"}})
                    return
                chunks = []
                usage = {}
                reason = "stop"
                fingerprint = None
                started = False
                try:
                    if payload["stream"]:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        started = True
                    while True:
                        event = ticket.events.get(timeout=owner.timeout)
                        if event["type"] == "error":
                            raise RuntimeError(event["message"])
                        if event["type"] == "done":
                            usage = event["usage"]
                            reason = event["finish_reason"]
                            fingerprint = event.get("system_fingerprint")
                            break
                        text = event["text"]
                        chunks.append(text)
                        if payload["stream"]:
                            chunk = {
                                "id": ticket.request_id,
                                "object": "chat.completion.chunk",
                                "created": ticket.created,
                                "model": payload["model"],
                                "choices": [
                                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                                ],
                            }
                            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                            self.wfile.flush()
                    common = {
                        "id": ticket.request_id,
                        "created": ticket.created,
                        "model": payload["model"],
                        "system_fingerprint": fingerprint,
                    }
                    if payload["stream"]:
                        final = {
                            **common,
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                            "usage": usage,
                        }
                        self.wfile.write(
                            ("data: " + json.dumps(final) + "\n\ndata: [DONE]\n\n").encode()
                        )
                        self.wfile.flush()
                    else:
                        self.json_response(
                            200,
                            {
                                **common,
                                "object": "chat.completion",
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "".join(chunks),
                                        },
                                        "finish_reason": reason,
                                    }
                                ],
                                "usage": usage,
                            },
                        )
                except (Empty, RuntimeError) as error:
                    message = str(error) or "Model request timed out"
                    if started:
                        self.wfile.write(
                            (
                                "data: "
                                + json.dumps({"error": {"message": message}})
                                + "\n\ndata: [DONE]\n\n"
                            ).encode()
                        )
                    else:
                        self.json_response(
                            503, {"error": {"message": message, "type": "model_error"}}
                        )
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    ticket.cancelled.set()

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def set_state(self, phase, **metadata):
        with self.lock:
            self.state = {"phase": phase, "ready": phase == "chat", **metadata}

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def close_window(self):
        self.set_state("training")
        while True:
            try:
                ticket = self.pending.get_nowait()
            except Empty:
                break
            ticket.events.put(
                {"type": "error", "message": "The chat window has ended; training is resuming."}
            )

    def close(self):
        self.close_window()
        self.server.shutdown()
        self.server.server_close()
