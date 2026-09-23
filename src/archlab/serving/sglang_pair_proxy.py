"""Keep existing MLflow model names while routing to independent SGLang workers.

This is a transport proxy, with no models or shared generation queue. It keeps
the current model names and forwards thinking traces and tool calls to the
already-patched SGLang workers. GPU workers are separately authenticated with
the same team key. No management API is exposed.
"""

from __future__ import annotations

import argparse
import hmac
import http.client
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from archlab.serving.openai_chat import MODELS

THINKING_HEADING = "### Thinking\n\n"
ANSWER_HEADING = "\n\n### Answer\n\n"


def _visible_text(value):
    return value if isinstance(value, str) else ""


def reveal_message_thinking(message):
    """Keep reasoning_content, and copy it into content so MLflow can display it.

    The OpenAI-compatible gateway adapter drops reasoning_content, and the
    playground renders assistant.content as Markdown. Folding the trace into a
    Thinking heading keeps it visible without changing the SGLang workers.
    """
    if not isinstance(message, dict):
        return message
    reasoning = _visible_text(message.get("reasoning_content")).strip()
    text = message.get("content") if isinstance(message.get("content"), str) else ""
    if not reasoning or text.startswith(THINKING_HEADING):
        return message
    revealed = THINKING_HEADING + reasoning
    if text.strip():
        revealed += ANSWER_HEADING + text
    visible = dict(message)
    visible["content"] = revealed
    return visible


def reveal_completion_thinking(body):
    if not isinstance(body, dict) or not isinstance(body.get("choices"), list):
        return body
    revealed = dict(body)
    revealed["choices"] = []
    for choice in body["choices"]:
        if not isinstance(choice, dict):
            revealed["choices"].append(choice)
            continue
        item = dict(choice)
        if "message" in item:
            item["message"] = reveal_message_thinking(item["message"])
        revealed["choices"].append(item)
    return revealed


class ThinkingStreamRewriter:
    """Rewrite SSE deltas so thinking tokens land in delta.content."""

    def __init__(self):
        self.phase = "start"

    def rewrite_line(self, line):
        if line.endswith(b"\r\n"):
            payload, ending = line[:-2], b"\r\n"
        elif line.endswith(b"\n"):
            payload, ending = line[:-1], b"\n"
        else:
            payload, ending = line, b""
        if not payload.startswith(b"data:"):
            return line
        data = payload[5:].strip()
        if data in (b"", b"[DONE]"):
            return line
        try:
            event = json.loads(data)
        except (ValueError, TypeError):
            return line
        return b"data: " + json.dumps(self.rewrite_event(event), ensure_ascii=False).encode() + ending

    def rewrite_event(self, event):
        if not isinstance(event, dict) or not isinstance(event.get("choices"), list):
            return event
        rewritten = dict(event)
        rewritten["choices"] = [self.rewrite_choice(choice) for choice in event["choices"]]
        return rewritten

    def rewrite_choice(self, choice):
        if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
            return choice
        item = dict(choice)
        item["delta"] = self.rewrite_delta(choice["delta"])
        return item

    def rewrite_delta(self, delta):
        reasoning = _visible_text(delta.get("reasoning_content"))
        text = delta.get("content") if isinstance(delta.get("content"), str) else ""
        pieces = []
        if reasoning:
            if self.phase == "start":
                pieces.append(THINKING_HEADING)
            self.phase = "thinking"
            pieces.append(reasoning)
        if text:
            if self.phase == "thinking":
                pieces.append(ANSWER_HEADING)
            self.phase = "answer"
            pieces.append(text)
        if not pieces:
            return delta
        visible = dict(delta)
        visible["content"] = "".join(pieces)
        return visible


def _thinking_requested(value):
    extra = value.get("extra_body") if isinstance(value.get("extra_body"), dict) else {}
    kwargs = value.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = extra.get("chat_template_kwargs") if isinstance(extra.get("chat_template_kwargs"), dict) else {}
    if "thinking" in kwargs:
        return bool(kwargs["thinking"])
    if "enable_thinking" in kwargs:
        return bool(kwargs["enable_thinking"])
    if "thinking" in value:
        return bool(value["thinking"])
    return True


def validate_request(value, *, max_tokens=8192):
    if not isinstance(value, dict) or value.get("model") not in MODELS:
        raise ValueError("choose a served DeepSeek checkpoint model")
    messages = value.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) > 64:
        raise ValueError("messages must contain 1–64 messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in (
            "system", "user", "assistant", "tool",
        ):
            raise ValueError("unsupported chat message role")
    if messages[-1].get("role") not in ("user", "tool"):
        raise ValueError("the final message must be from the user or a tool")
    thinking = _thinking_requested(value)
    limit = value.get("max_completion_tokens", value.get("max_tokens"))
    if limit is None:
        limit = 4096 if thinking else 128
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
    if value.get("n", 1) != 1:
        raise ValueError("only one completion is supported")
    tools = value.get("tools") or []
    if tools and (not isinstance(tools, list) or len(tools) > 32):
        raise ValueError("tools must be a list of at most 32 function definitions")
    payload = {
        "model": value["model"],
        "variant": MODELS[value["model"]],
        "messages": messages,
        "max_tokens": limit,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": bool(value.get("stream", False)),
        "chat_template_kwargs": {"thinking": thinking},
    }
    if tools:
        payload["tools"] = tools
        if value.get("tool_choice") is not None:
            payload["tool_choice"] = value["tool_choice"]
    if "seed" in value:
        payload["seed"] = value["seed"]
    if isinstance(value.get("stream_options"), dict):
        payload["stream_options"] = value["stream_options"]
    return payload


class SGLangPairProxy:
    def __init__(self, host, port, token_file, backends, *, timeout=1800, admission_file=None):
        self.token = Path(token_file).read_text().strip()
        if len(self.token) < 24 or set(backends) != {"normal", "simplicial"}:
            raise ValueError("both authenticated model backends are required")
        self.backends, self.timeout = backends, timeout
        self.admission_file = Path(admission_file) if admission_file is not None else None
        self.slots = {variant: threading.BoundedSemaphore(8) for variant in backends}
        for target in backends.values():
            url = urlsplit(target["url"])
            if url.scheme != "http" or not url.hostname or url.path not in ("", "/"):
                raise ValueError("backend must be an explicit private HTTP origin")
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, status, body):
                data = json.dumps(body, allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def authorized(self):
                if hmac.compare_digest(self.headers.get("Authorization", "").encode(),
                                       ("Bearer " + owner.token).encode()):
                    return True
                self.respond(401, {"error": {"message": "Unauthorized"}})
                return False

            def do_GET(self):
                if not self.authorized():
                    return
                if self.path == "/health":
                    admitted = owner.admitted()
                    if admitted:
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            states = dict(zip(owner.backends, executor.map(owner.backend_health, owner.backends), strict=True))
                    else:
                        states = dict.fromkeys(owner.backends, False)
                    ready = all(states.values())
                    self.respond(200 if ready else 503, dict(
                        ready=ready, phase="chat" if ready else "loading-and-validating",
                        backend="sglang", available_until_unix=None, workers=states,
                        checkpoints={v: b["cursor"] for v, b in owner.backends.items()}))
                elif self.path == "/v1/models":
                    self.respond(200, {"object": "list", "data": [
                        {"id": name, "object": "model", "owned_by": "archlab"} for name in MODELS]})
                else:
                    self.respond(404, {"error": {"message": "Not found"}})

            def do_POST(self):
                if not self.authorized():
                    return
                if self.path != "/v1/chat/completions":
                    self.respond(404, {"error": {"message": "Not found"}})
                    return
                if not owner.admitted():
                    self.respond(503, {"error": {"type": "model_not_ready", "message":
                        "Chat is temporarily unavailable while the new inference service loads and is validated."}})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 1_000_000:
                        raise ValueError("request body must be between 1 byte and 1 MB")
                    payload = validate_request(json.loads(self.rfile.read(length)))
                except (ValueError, TypeError, AttributeError):
                    self.respond(400, {"error": {"message": "Invalid text-chat request"}})
                    return
                variant = payload.pop("variant")
                semaphore = owner.slots[variant]
                if not semaphore.acquire(blocking=False):
                    self.respond(429, {"error": {"message": "This model's request queue is full"}})
                    return
                connection, sent = None, False
                try:
                    connection = owner.connection(variant)
                    connection.request("POST", "/v1/chat/completions", json.dumps(payload),
                                       {"Authorization": "Bearer " + owner.token,
                                        "Content-Type": "application/json"})
                    response = connection.getresponse()
                    content_type = response.getheader("Content-Type", "application/json")
                    if "text/event-stream" in content_type:
                        self.send_response(response.status)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.send_header("X-Accel-Buffering", "no")
                        self.end_headers()
                        sent = True
                        rewriter = ThinkingStreamRewriter()
                        while line := response.readline():
                            self.wfile.write(rewriter.rewrite_line(line))
                            self.wfile.flush()
                        self.close_connection = True
                    else:
                        body = json.loads(response.read())
                        if response.status == 200:
                            step = owner.backends[variant]["cursor"]["step"]
                            body["system_fingerprint"] = f"sglang-{variant}-step-{step:06d}"
                            body = reveal_completion_thinking(body)
                        self.respond(response.status, body)
                        sent = True
                except (OSError, http.client.HTTPException, ValueError):
                    if not sent:
                        try:
                            self.respond(502, {"error": {"message": "SGLang backend unavailable"}})
                        except OSError:
                            pass
                finally:
                    if connection is not None:
                        connection.close()
                    semaphore.release()

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = None

    def admitted(self):
        if self.admission_file is None:
            return True
        try:
            value = json.loads(self.admission_file.read_text())
            return value.get("passed") is True and value.get("checkpoints") == {
                variant: target["cursor"] for variant, target in self.backends.items()}
        except (OSError, ValueError):
            return False

    def connection(self, variant, *, timeout=None):
        url = urlsplit(self.backends[variant]["url"])
        return http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout or self.timeout)

    def backend_health(self, variant):
        connection = self.connection(variant, timeout=10)
        try:
            connection.request("GET", "/health", headers={"Authorization": "Bearer " + self.token})
            response = connection.getresponse()
            response.read()
            return response.status == 200
        except (OSError, http.client.HTTPException):
            return False
        finally:
            connection.close()

    def start(self):
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        if self.thread:
            self.thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    proxy = SGLangPairProxy(config["host"], config["port"], config["token_file"], config["backends"],
                           admission_file=config["admission_file"])
    proxy.server.serve_forever()


if __name__ == "__main__":
    main()
