import http.client
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from archlab.serving.sglang_pair_proxy import SGLangPairProxy


def test_auth_defaults_streaming_and_independent_model_queues(tmp_path):
    token = "test-key-" * 5
    token_file = tmp_path / "token"
    token_file.write_text(token)
    entered, release = threading.Event(), threading.Event()
    recorded = []

    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            assert self.headers["Authorization"] == "Bearer " + token
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            recorded.append(body)
            if body["model"].endswith("normal-latest"):
                entered.set()
                assert release.wait(5)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if body["stream"] else "application/json")
            self.end_headers()
            if body["stream"]:
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"fixture"}}]}\n\ndata: [DONE]\n\n')
            else:
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "fixture"}}]}).encode())

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    targets = {variant: dict(url=f"http://127.0.0.1:{backend.server_port}", cursor=dict(step=step))
               for variant, step in (("normal", 4537), ("simplicial", 3620))}
    proxy = SGLangPairProxy("127.0.0.1", 0, token_file, targets, timeout=5)
    proxy.start()

    def request(variant, *, stream=False, key=token):
        connection = http.client.HTTPConnection("127.0.0.1", proxy.server.server_port, timeout=5)
        body = dict(model=f"deepseek-v41-{variant}-latest",
                    messages=[dict(role="user", content="fixture")], stream=stream)
        connection.request("POST", "/v1/chat/completions", json.dumps(body),
                           {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    try:
        assert request("simplicial", key="wrong")[0] == 401
        with ThreadPoolExecutor(max_workers=2) as executor:
            normal = executor.submit(request, "normal")
            assert entered.wait(3)
            status, body = request("simplicial")
            assert status == 200 and not normal.done()
            assert json.loads(body)["system_fingerprint"] == "sglang-simplicial-step-003620"
            release.set()
            assert normal.result(timeout=3)[0] == 200
        status, data = request("simplicial", stream=True)
        assert status == 200 and data.endswith(b"data: [DONE]\n\n")
        assert all(row["max_tokens"] == 4096 for row in recorded)
        assert all(row["chat_template_kwargs"] == {"thinking": True} for row in recorded)
        tools = [dict(type="function", function=dict(name="lookup", parameters=dict(type="object")))]
        connection = http.client.HTTPConnection("127.0.0.1", proxy.server.server_port, timeout=5)
        connection.request("POST", "/v1/chat/completions", json.dumps(dict(
            model="deepseek-v41-normal-latest",
            messages=[dict(role="user", content="use the tool")],
            tools=tools, max_tokens=32, chat_template_kwargs=dict(thinking=False),
        )), {"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        assert connection.getresponse().status == 200
        connection.close()
        assert recorded[-1]["tools"] == tools
        assert recorded[-1]["chat_template_kwargs"] == {"thinking": False}
        connection = http.client.HTTPConnection("127.0.0.1", proxy.server.server_port, timeout=5)
        connection.request("POST", "/v1/chat/completions", json.dumps(dict(
            model="deepseek-v41-normal-latest",
            messages=[dict(role="user", content="fixture")],
            max_tokens=32, stream=True, stream_options=dict(include_usage=True),
        )), {"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        assert connection.getresponse().status == 200
        connection.close()
        assert recorded[-1]["stream_options"] == {"include_usage": True}
        assert proxy.backend_health("normal") and proxy.backend_health("simplicial")
    finally:
        release.set()
        proxy.close()
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=5)


def test_thinking_trace_is_folded_into_visible_content(tmp_path):
    token = "test-key-" * 5
    token_file = tmp_path / "token"
    token_file.write_text(token)

    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            if body["stream"]:
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in (
                    {"choices": [{"index": 0, "delta": {"reasoning_content": "plan "}}]},
                    {"choices": [{"index": 0, "delta": {"reasoning_content": "now"}}]},
                    {"choices": [{"index": 0, "delta": {"content": "4"}}]},
                ):
                    self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {
                    "role": "assistant", "content": "4", "reasoning_content": "plan now",
                    "tool_calls": [{"id": "call_1", "type": "function",
                                    "function": {"name": "lookup", "arguments": "{\"q\":\"2+2\"}"}}],
                }}]}).encode())

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    targets = {variant: dict(url=f"http://127.0.0.1:{backend.server_port}", cursor=dict(step=step))
               for variant, step in (("normal", 4537), ("simplicial", 3620))}
    proxy = SGLangPairProxy("127.0.0.1", 0, token_file, targets, timeout=5)
    proxy.start()

    def request(*, stream=False):
        connection = http.client.HTTPConnection("127.0.0.1", proxy.server.server_port, timeout=5)
        connection.request("POST", "/v1/chat/completions", json.dumps(dict(
            model="deepseek-v41-normal-latest",
            messages=[dict(role="user", content="fixture")], max_tokens=32, stream=stream,
        )), {"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        response = connection.getresponse()
        headers = dict(response.getheaders())
        body = response.read()
        connection.close()
        return response.status, headers, body

    try:
        status, _, body = request()
        message = json.loads(body)["choices"][0]["message"]
        assert status == 200
        assert "### Thinking" in message["content"] and "plan now" in message["content"]
        assert message["content"].endswith("4")
        assert message["reasoning_content"] == "plan now"
        assert message["tool_calls"][0]["function"]["name"] == "lookup"
        status, headers, body = request(stream=True)
        header_names = {name.lower(): value for name, value in headers.items()}
        assert status == 200
        assert header_names["x-accel-buffering"] == "no"
        text = "".join(
            json.loads(line[5:].strip())["choices"][0]["delta"].get("content") or ""
            for line in body.decode().splitlines() if line.startswith("data: {")
        )
        assert "### Thinking" in text and "plan now" in text and text.endswith("4")
        assert b"data: [DONE]" in body
    finally:
        proxy.close()
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=5)


def test_production_admission_requires_both_exact_checkpoint_cursors(tmp_path):
    token = tmp_path / "token"
    token.write_text("test-key-" * 5)
    admission = tmp_path / "admission.json"
    targets = {v: dict(url="http://127.0.0.1:1", cursor=dict(step=s))
               for v, s in (("normal", 4537), ("simplicial", 3620))}
    proxy = SGLangPairProxy("127.0.0.1", 0, token, targets, admission_file=admission)
    try:
        assert not proxy.admitted()
        admission.write_text(json.dumps(dict(passed=True, checkpoints={"normal": {"step": 4537}})))
        assert not proxy.admitted()
        admission.write_text(json.dumps(dict(passed=True, checkpoints={v: b["cursor"] for v, b in targets.items()})))
        assert proxy.admitted()
    finally:
        proxy.server.server_close()
