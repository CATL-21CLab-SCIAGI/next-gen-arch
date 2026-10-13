"""E2B is an execution boundary; failures never masquerade as failed tests."""

import json
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
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


def test_directory_transfer_preserves_paths_empty_directories_and_bytes(tmp_path):
    source = tmp_path / "editor strings"
    nested = source / "nested 雪\nname"
    nested.mkdir(parents=True)
    (source / "empty").mkdir()
    payload = b"exact\x00bytes\r\n" + "café\n雪".encode()
    (nested / "old\nstring").write_bytes(payload)
    calls = []
    env = E2BEnvironment(image="image", user="researcher")
    env.sandbox = SimpleNamespace(files=SimpleNamespace(
        make_dir=lambda *a, **kw: calls.append(("mkdir", a, kw)),
        write=lambda *a, **kw: calls.append(("write", a, kw))))
    env.copy_to(str(source), "/tmp/native editor 雪\nstrings", timeout=37)
    assert ("mkdir", ("/tmp/native editor 雪\nstrings/empty",),
            {"user": "researcher", "request_timeout": 37}) in calls
    assert calls[-1] == ("write", ("/tmp/native editor 雪\nstrings/nested 雪\nname/old\nstring", payload),
                         {"user": "researcher", "request_timeout": 37})
    assert all(call[2] == {"user": "researcher", "request_timeout": 37} for call in calls)


def test_directory_transfer_rejects_links_before_any_upload(tmp_path):
    source = tmp_path / "strings"
    source.mkdir()
    (source / "old").write_text("old")
    (source / "link").symlink_to(source / "old")
    calls = []
    env = E2BEnvironment(image="image")
    env.sandbox = SimpleNamespace(files=SimpleNamespace(
        make_dir=lambda *a, **kw: calls.append(a), write=lambda *a, **kw: calls.append(a)))
    with pytest.raises(ValueError, match="regular files and directories"):
        env.copy_to(str(source), "/tmp/native-editor")
    assert calls == []


@pytest.mark.parametrize("failed_operation", ["make_dir", "write"])
def test_directory_transfer_failure_discards_sdk_secret_context(sdk, tmp_path, failed_operation):
    def fail(*args, **kwargs):
        try:
            raise ValueError("sensitive-directory-value")
        except ValueError as cause:
            raise RuntimeError("SDK sensitive-auth-value") from cause

    source = tmp_path / "strings"
    source.mkdir()
    (source / "old").write_text("old")
    methods = {"make_dir": lambda *a, **kw: None, "write": lambda *a, **kw: None}
    methods[failed_operation] = fail
    env = E2BEnvironment(image="image")
    env.sandbox = SimpleNamespace(files=SimpleNamespace(**methods))
    with pytest.raises(sdk[1], match="file transfer infrastructure failure: RuntimeError") as raised:
        env.copy_to(str(source), "/tmp/native-editor")
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "sensitive" not in "".join(traceback.format_exception(raised.value))


def test_actual_upstream_read_write_edit_execute_through_sdk_boundary(monkeypatch, tmp_path):
    upstream = Path(__file__).resolve().parents[1] / "third_party/verl/third_party/mimoagent-osr/src"
    monkeypatch.syspath_prepend(str(upstream))
    from mimoagent.tools.edit import EditTool
    from mimoagent.tools.read import ReadTool
    from mimoagent.tools.write import WriteTool

    class Exit(Exception):
        def __init__(self, result):
            self.stdout, self.stderr, self.exit_code = result.stdout, result.stderr, result.returncode

    monkeypatch.setitem(sys.modules, "e2b.sandbox.commands.command_handle", SimpleNamespace(CommandExitException=Exit))
    directories, commands = [], []

    def make_dir(path, **kwargs):
        directories.append(Path(path))
        Path(path).mkdir(parents=True, exist_ok=True)

    def write(path, data, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)

    def run(command, *, cwd, timeout, user):
        commands.append(command)
        result = subprocess.run(["bash", "-c", command], cwd=cwd, timeout=timeout,
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise Exit(result)
        return SimpleNamespace(stdout=result.stdout, stderr=result.stderr, exit_code=0)

    env = E2BEnvironment(image="image", cwd=str(tmp_path))
    env.sandbox = SimpleNamespace(files=SimpleNamespace(make_dir=make_dir, write=write),
                                  commands=SimpleNamespace(run=run))
    path = tmp_path / "model café\ncase.py"
    old = "alpha = 'café'\r\n"
    new = "alpha = '雪'\r\nnew_line = 3\n"
    initial = old + "beta = 2\n"
    context = {"env": env}
    assert WriteTool().execute({"path": str(path), "file_text": initial}, context).success
    assert path.read_bytes() == initial.encode()
    assert "café" in ReadTool().execute({"path": str(path)}, context).output
    verifier = tmp_path / "verify.py"
    verifier.write_text("import runpy, sys\nassert runpy.run_path(sys.argv[1])['alpha'] == '雪'\n")
    command = " ".join(shlex.quote(str(value)) for value in (sys.executable, verifier, path))
    assert env.execute(command)["returncode"] == 1
    result = EditTool().execute({"path": str(path), "old_str": old, "new_str": new}, context)
    assert result.success, result.output
    assert path.read_bytes() == initial.replace(old, new, 1).encode()
    assert directories and all(not directory.exists() for directory in directories)
    assert env.execute(command)["returncode"] == 0
    assert env.execute("printf 'actual failing command\\n'; exit 7") == {
        "output": "actual failing command\n", "returncode": 7}
    assert len(commands) >= 5


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
