import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from archlab.serving.openai_chat import ChatFront, validate_request


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        key = Path(self.tmp.name) / "key"
        key.write_text("test-private-key-" + "a" * 32)
        self.front = ChatFront("127.0.0.1", 0, key, timeout=5)
        self.addCleanup(self.front.close)
        self.url = f"http://127.0.0.1:{self.front.server.server_port}"
        self.headers = {
            "Authorization": "Bearer " + key.read_text(),
            "Content-Type": "application/json",
        }
        self.payload = {
            "model": "deepseek-v41-normal-latest",
            "messages": [{"role": "user", "content": "fixture"}],
        }

    def test_auth_and_busy_state(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.url + "/v1/models")
        self.assertEqual(error.exception.code, 401)
        request = Request(
            self.url + "/v1/chat/completions", json.dumps(self.payload).encode(), self.headers
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 503)

    def test_text_and_streaming_completion_protocol(self):
        self.front.set_state("chat")
        for stream in (False, True):

            def reply():
                ticket = self.front.pending.get(timeout=5)
                ticket.events.put({"type": "text", "text": "fixture reply"})
                ticket.events.put(
                    {
                        "type": "done",
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                        "system_fingerprint": "checkpoint-fixture",
                    }
                )

            worker = threading.Thread(target=reply)
            worker.start()
            request = Request(
                self.url + "/v1/chat/completions",
                json.dumps({**self.payload, "stream": stream}).encode(),
                self.headers,
            )
            with urlopen(request, timeout=5) as response:
                body = response.read().decode()
            worker.join()
            if stream:
                self.assertIn("data: [DONE]", body)
                self.assertIn("fixture reply", body)
            else:
                self.assertEqual(
                    json.loads(body)["choices"][0]["message"]["content"], "fixture reply"
                )

    def test_request_bounds_and_text_content(self):
        self.assertEqual(validate_request(self.payload)["variant"], "normal")
        for update in (
            {"max_tokens": 0},
            {"max_tokens": 100000},
            {"temperature": float("nan")},
            {"top_p": 0},
            {"model": "unknown"},
            {"seed": 2**64},
            {"seed": -(2**63) - 1},
        ):
            with self.assertRaises(ValueError):
                validate_request({**self.payload, **update})


if __name__ == "__main__":
    unittest.main()
