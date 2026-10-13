"""E2B is an execution boundary; failures never masquerade as failed tests."""

import json
import sys
import traceback
from types import SimpleNamespace

import pytest

from archlab.artifacts import sha256_file
from archlab.training.sandbox import E2BEnvironment, check_sandbox


@pytest.fixture
def sdk(monkeypatch):
    class Exit(Exception):
        def __init__(self, code=1):
            self.stdout, self.stderr, self.exit_code = "output", "error", code

    class Transport(Exception):
        pass

    monkeypatch.setitem(sys.modules, "e2b.sandbox.commands.command_handle", SimpleNamespace(CommandExitException=Exit))
    monkeypatch.setitem(sys.modules, "mimoagent.environments", SimpleNamespace(TransportError=Transport))
    return Exit, Transport


def test_explicit_image_mapping_and_parent_only_credentials(monkeypatch, tmp_path):
    mapping = tmp_path / "templates.json"
    mapping.write_text(json.dumps({"images": {"image@sha256:one": {
        "source_image": "image@sha256:one", "template_id": "qualified-task-template",
    }}}))
    secret = tmp_path / "key"
    secret.write_text("test-private-key")
    calls = []
    sandbox = SimpleNamespace(kill=lambda: calls.append("killed"))

    def create(**kwargs):
        calls.append(kwargs)
        return sandbox

    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(Sandbox=SimpleNamespace(create=create)))
    env = E2BEnvironment(image="image@sha256:one", template_map=str(mapping),
                         api_key_file=str(secret), domain="sandbox.example.invalid")
    env.start()
    assert calls[0]["template"] == "qualified-task-template"
    assert calls[0]["allow_internet_access"] is False
    assert calls[0]["api_key"] == "test-private-key"
    assert calls[0]["envs"] == {"PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                                "UV_OFFLINE": "1"}
    assert "test-private-key" not in calls[0]["envs"].values()
    assert env.get_template_vars() == {"cwd": "/testbed", "image": "image@sha256:one"}
    env.cleanup()
    env.cleanup()
    assert calls.count("killed") == 1


def test_nonzero_exit_is_a_real_process_result(sdk):
    exit_error, _ = sdk

    def run(*args, **kwargs):
        raise exit_error(2)

    env = E2BEnvironment(image="image")
    env.sandbox = SimpleNamespace(commands=SimpleNamespace(run=run))
    assert env.execute("unused") == {"output": "outputerror", "returncode": 2}


def test_transport_failure_is_not_a_zero_reward(sdk):
    _, transport_error = sdk

    def run(*args, **kwargs):
        raise TimeoutError("request expired")

    env = E2BEnvironment(image="image")
    env.sandbox = SimpleNamespace(commands=SimpleNamespace(run=run))
    with pytest.raises(transport_error, match="TimeoutError"):
        env.execute("unused")


def test_verifier_transfer_preserves_bytes(tmp_path):
    source = tmp_path / "test.patch"
    source.write_bytes(b"exact\x00bytes\r\n")
    writes = []
    env = E2BEnvironment(image="image")
    env.sandbox = SimpleNamespace(files=SimpleNamespace(write=lambda *a, **kw: writes.append((a, kw))))
    env.copy_to(str(source), "/tmp/verifier.patch")
    assert writes[0][0] == ("/tmp/verifier.patch", b"exact\x00bytes\r\n")
    assert writes[0][1]["user"] == "root"


def test_failed_cleanup_can_retry(sdk):
    calls = []

    def kill():
        calls.append("kill")
        if len(calls) == 1:
            raise TimeoutError("request expired")

    env = E2BEnvironment(image="image")
    env.sandbox = SimpleNamespace(kill=kill)
    with pytest.raises(sdk[1], match="TimeoutError"):
        env.cleanup()
    assert env.sandbox is not None
    env.cleanup()
    assert env.sandbox is None
    assert calls == ["kill", "kill"]


def test_create_failure_hides_sdk_authentication_traceback(monkeypatch, tmp_path, sdk):
    mapping = tmp_path / "templates.json"
    mapping.write_text(json.dumps({"images": {"image": {
        "source_image": "image", "template_id": "template"}}}))
    key = tmp_path / "key"
    key.write_text("sensitive-test-value")

    def create(**kwargs):
        raise RuntimeError("SDK request failed with sensitive-test-value")

    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(Sandbox=SimpleNamespace(create=create)))
    env = E2BEnvironment(image="image", template_map=str(mapping), api_key_file=str(key))
    with pytest.raises(sdk[1], match="create infrastructure failure") as raised:
        env.start()
    assert "sensitive-test-value" not in "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert raised.value.__suppress_context__


@pytest.mark.parametrize("operation", ["command", "file transfer", "cleanup"])
def test_sdk_failure_discards_nested_secret_context(sdk, tmp_path, operation):
    def fail(*args, **kwargs):
        try:
            raise ValueError("sensitive-nested-value")
        except ValueError as cause:
            raise RuntimeError("SDK sensitive-test-value") from cause

    env = E2BEnvironment(image="image")
    env.sandbox = SimpleNamespace(commands=SimpleNamespace(run=fail),
                                  files=SimpleNamespace(write=fail), kill=fail)
    source = tmp_path / "fixture.patch"
    source.write_text("fixture")
    with pytest.raises(sdk[1]) as raised:
        if operation == "command":
            env.execute("true")
        elif operation == "file transfer":
            env.copy_to(str(source), "/tmp/fixture.patch")
        else:
            env.cleanup()
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "sensitive" not in "".join(traceback.format_exception(raised.value))


@pytest.mark.parametrize("mutation", [None, "ready_only", "changed_patch", "transport_error", "tampered", "wrong_service", "changed_harness"])
def test_admission_requires_every_task_runtime_receipt(tmp_path, mutation):
    image = "source@sha256:" + "1" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    task = {"instance_id": "task", "resolved_image": image, "test_patch_sha256": "2" * 64}
    (tmp_path / "index.jsonl").write_text(json.dumps(task) + "\n")
    harness = tmp_path / "code_bash.yaml"
    harness.write_text("environment:\n  anti_hack_cleanup: true\n")
    spec = tmp_path / "single_bash.yaml"
    spec.write_text("harnesses:\n  - label: native-bash\n    config: code_bash.yaml\n")
    receipt = {"harness_sha256": sha256_file(harness), "status": "task_runtime_verified", "instance_id": "task", "source_image": image,
               "template_id": "template", "test_patch_sha256": task["test_patch_sha256"],
               "runtime_base_commit": "a" * 40, "verifier_output_sha256": "b" * 64,
               "verifier_returncode": 1, "clean_task_state": True, "history_truncated": True,
               "test_patch_applied": True, "infrastructure_errors": []}
    service = {"service_id": "dedicated", "domain": "sandbox.example.invalid",
               "key_file": str(tmp_path / "sandbox.key")}
    receipt["service"] = {**service, "service_id": "other"} if mutation == "wrong_service" else service
    if mutation == "changed_patch":
        receipt["test_patch_sha256"] = "3" * 64
    if mutation == "transport_error":
        receipt["infrastructure_errors"] = ["timeout"]
    if mutation == "changed_harness":
        harness.write_text("environment:\n  anti_hack_cleanup: false\n")
    proof = tmp_path / "runtime.json"
    proof.write_text(json.dumps(receipt))
    binding = {"source_image": image, "template_id": "template",
               "runtime_verification": {"path": str(proof), "sha256": sha256_file(proof)}}
    if mutation == "ready_only":
        binding.pop("runtime_verification")
    if mutation == "tampered":
        proof.write_text("{}")
    mapping = tmp_path / "templates.json"
    mapping.write_text(json.dumps({"dataset_manifest_sha256": sha256_file(manifest),
                                   "service": service,
                                   "images": {image: binding}}))
    key = tmp_path / "sandbox.key"
    key.write_text("unit-test-only")
    variables = {"ARCHLAB_E2B_TEMPLATE_MAP": str(mapping), "ARCHLAB_E2B_KEY_FILE": str(key),
                 "E2B_DOMAIN": "sandbox.example.invalid", "MIXED_HARNESS_SPEC": str(spec)}
    config = SimpleNamespace(ray_kwargs=SimpleNamespace(ray_init=SimpleNamespace(
        runtime_env=SimpleNamespace(env_vars=variables))))
    experiment = SimpleNamespace(data_manifest=str(manifest))
    if mutation:
        with pytest.raises(ValueError):
            check_sandbox(experiment, config)
    else:
        assert check_sandbox(experiment, config)["tasks"] == 1
