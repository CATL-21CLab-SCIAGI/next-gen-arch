import json
import sys
import types

import pytest


def _environment(tmp_path, monkeypatch, outcomes, wait=1800.0):
    from archlab.training import sandbox

    calls = []

    class FakeSandbox:
        @staticmethod
        def create(**kwargs):
            calls.append(kwargs["template"])
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    monkeypatch.setitem(sys.modules, "e2b", types.SimpleNamespace(Sandbox=FakeSandbox))
    monkeypatch.setattr(sandbox.time, "sleep", lambda seconds: None)
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"images": {"img": {"source_image": "img", "template_id": "tpl"}}}))
    key = tmp_path / "key"
    key.write_text("secret")
    env = sandbox.E2BEnvironment(image="img", template_map=str(mapping), api_key_file=str(key),
                                 domain="example", capacity_wait_seconds=wait)
    return env, calls


def test_capacity_refusals_wait_then_succeed(tmp_path, monkeypatch):
    env, calls = _environment(tmp_path, monkeypatch, [RuntimeError("503: insufficient resources"),
                                                      RuntimeError("429: rate limit"), "box"])
    env.start()
    assert env.sandbox == "box" and calls == ["tpl"] * 3


def test_other_create_errors_fail_without_retry(tmp_path, monkeypatch):
    from mimoagent.environments import TransportError

    env, calls = _environment(tmp_path, monkeypatch, [RuntimeError("template not found"), "box"])
    with pytest.raises(TransportError):
        env.start()
    assert calls == ["tpl"]


def test_capacity_wait_is_bounded(tmp_path, monkeypatch):
    from mimoagent.environments import TransportError

    env, calls = _environment(tmp_path, monkeypatch, [RuntimeError("503")] * 50, wait=0.0)
    with pytest.raises(TransportError):
        env.start()
    assert calls == ["tpl"]
