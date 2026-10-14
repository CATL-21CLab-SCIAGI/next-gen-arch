"""MiMo's Environment contract backed by the official E2B SDK (PAI-Instant).

Task images map to prequalified private templates; this adapter never builds an
image, copies credentials into a task, or interprets transport failure as a test
failure. The SDK owns sandbox lifecycle, command execution and file transfer.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from archlab.artifacts import sha256_file


def _env(name):
    return field(default_factory=lambda: os.environ.get(name, ""))


def _capacity_refusal(error):
    """True for pool-capacity refusals (HTTP 503 insufficient resources, 429 rate limit)."""
    text = str(error)
    return any(marker in text for marker in ("503", "429", "insufficient resources"))


def _infrastructure_failure(operation, error):
    from mimoagent.environments import TransportError

    # SDK frames carry authentication kwargs; never chain them into run logs.
    return TransportError(f"E2B {operation} infrastructure failure: {type(error).__name__}")


def check_sandbox(experiment, config):
    """Require image-bound runtime evidence for every indexed training/eval task.

    A Ready template is only an allocation artifact. Each binding additionally
    needs a checksummed task preflight: actual clean git state, hidden-patch
    application and a completed verifier with no infrastructure failure.
    """
    from yaml import safe_load

    variables = config.ray_kwargs.ray_init.runtime_env.env_vars
    spec_path = Path(variables["MIXED_HARNESS_SPEC"])
    harnesses = safe_load(spec_path.read_text())["harnesses"]
    if len(harnesses) != 1:
        raise ValueError("sandbox receipts require one explicit code harness")
    harness_path = Path(harnesses[0]["config"])
    if not harness_path.is_absolute():
        harness_path = spec_path.parent / harness_path
    harness_sha256 = sha256_file(harness_path)
    mapping_path = Path(variables["ARCHLAB_E2B_TEMPLATE_MAP"])
    mapping = json.loads(mapping_path.read_text())
    service = mapping.get("service")
    if service and (service["domain"] != variables.get("E2B_DOMAIN")
                    or Path(service["key_file"]).resolve() != Path(variables["ARCHLAB_E2B_KEY_FILE"]).resolve()):
        raise ValueError("sandbox mapping belongs to another service deployment")
    manifest = Path(experiment.data_manifest)
    if mapping.get("dataset_manifest_sha256") != sha256_file(manifest):
        raise ValueError("sandbox mapping belongs to another dataset contract")
    index = [json.loads(line) for line in (manifest.parent / "index.jsonl").read_text().splitlines()]
    verified = []
    for task in index:
        image = task["resolved_image"]
        binding = mapping["images"].get(image, {})
        proof = binding.get("runtime_verification", {})
        if binding.get("source_image") != image or not binding.get("template_id") or not proof:
            raise ValueError(f"missing qualified sandbox mapping: {task['instance_id']}")
        receipt_path = Path(proof["path"])
        if sha256_file(receipt_path) != proof["sha256"]:
            raise ValueError("sandbox runtime preflight checksum mismatch")
        receipt = json.loads(receipt_path.read_text())
        expected = {"status": "task_runtime_verified", "instance_id": task["instance_id"],
                    "source_image": image, "template_id": binding["template_id"],
                    "test_patch_sha256": task["test_patch_sha256"],
                    "harness_sha256": harness_sha256}
        if service:
            expected["service"] = service
        rc = receipt.get("verifier_returncode")
        if (any(receipt.get(key) != value for key, value in expected.items())
                or not re.fullmatch(r"[0-9a-f]{40}", receipt.get("runtime_base_commit", ""))
                or not re.fullmatch(r"[0-9a-f]{64}", receipt.get("verifier_output_sha256", ""))
                or type(rc) is not int or not 0 <= rc < 124
                or any(receipt.get(key) is not True for key in (
                    "clean_task_state", "history_truncated", "test_patch_applied"))
                or receipt.get("infrastructure_errors") != []):
            raise ValueError(f"invalid task runtime preflight: {task['instance_id']}")
        verified.append(task["instance_id"])
    if not variables.get("E2B_DOMAIN") or not Path(variables["ARCHLAB_E2B_KEY_FILE"]).is_file():
        raise ValueError("sandbox authentication is not configured on this deployment")
    return {"tasks": len(verified), "template_map_sha256": sha256_file(mapping_path),
            "harness_sha256": harness_sha256,
            "scope": "all indexed task runtime receipts; live reference-agent verdict is separate"}


@dataclass
class E2BEnvironmentConfig:
    image: str
    cwd: str = "/testbed"
    timeout: int = 300
    sandbox_timeout: int = 7200
    template_map: str = _env("ARCHLAB_E2B_TEMPLATE_MAP")
    api_key_file: str = _env("ARCHLAB_E2B_KEY_FILE")
    domain: str = _env("E2B_DOMAIN")
    user: str = "root"
    capacity_wait_seconds: float = 1800.0


class E2BEnvironment:
    def __init__(self, *, config_class=E2BEnvironmentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.sandbox = None
        self.template_id = None

    def start(self):
        from e2b import Sandbox

        if self.sandbox is not None:
            raise RuntimeError("sandbox already started")
        config = self.config
        templates = json.loads(Path(config.template_map).read_text())
        service = templates.get("service")
        if service and (service["domain"] != config.domain
                        or Path(service["key_file"]).resolve() != Path(config.api_key_file).resolve()):
            raise ValueError("task/template service binding differs from deployment")
        entry = templates["images"][config.image]
        if entry["source_image"] != config.image or not entry["template_id"]:
            raise ValueError("task/template provenance mismatch")
        self.template_id = entry["template_id"]
        # The key stays in the parent-side SDK; task environments never receive it.
        key = Path(config.api_key_file).read_text().strip() if config.api_key_file else None
        failure = None
        deadline = time.monotonic() + config.capacity_wait_seconds
        delay = 15.0
        try:
            while True:
                try:
                    self.sandbox = Sandbox.create(
                        template=entry["template_id"], timeout=config.sandbox_timeout,
                        domain=config.domain or None, api_key=key, allow_internet_access=False,
                        envs={"PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                              "UV_OFFLINE": "1"},
                        request_timeout=60, metadata={"purpose": "archlab-code-rl"},
                    )
                    break
                except Exception as error:
                    # A shared pool refusing for capacity (503/429) is transient: wait for a slot
                    # within a bounded budget. Every other failure stays an infrastructure error.
                    if _capacity_refusal(error) and time.monotonic() + delay < deadline:
                        time.sleep(delay * (0.5 + random.random()))
                        delay = min(delay * 2, 120.0)
                        continue
                    failure = _infrastructure_failure("create", error)
                    break
        finally:
            del key
        if failure is not None:
            # Raising outside the handler also removes __context__, including
            # for loggers that inspect exception objects instead of tracebacks.
            raise failure from None

    def execute(self, command, cwd="", timeout=None):
        from e2b.sandbox.commands.command_handle import CommandExitException

        if self.sandbox is None:
            raise RuntimeError("sandbox is not started")
        try:
            try:
                result = self.sandbox.commands.run(
                    command, cwd=cwd or self.config.cwd, timeout=timeout or self.config.timeout,
                    user=self.config.user,
                )
            except CommandExitException as result_error:
                result = result_error
            output = {"output": result.stdout + result.stderr, "returncode": result.exit_code}
        except Exception as error:
            # Ordinary nonzero exits were handled above. SDK timeout/network
            # errors invalidate the trajectory instead of inventing reward zero.
            failure = _infrastructure_failure("command", error)
        else:
            return output
        raise failure from None

    def execute_detached(self, command, cwd="", timeout=None, **_):
        return self.execute(command, cwd=cwd, timeout=timeout)

    def copy_to(self, src_path, dest_path, *, timeout=300, max_retries=10):
        """Upload a file, or a directory's contents at the exact destination.

        MiMo's editor sends a fresh directory containing its old/new strings.
        The SDK creates directories and transfers bytes; links and special files
        inside directory uploads are unsupported rather than followed.
        """
        if self.sandbox is None:
            raise RuntimeError("sandbox is not started")
        source = Path(src_path)
        if source.is_file():
            directories = []
            files = [(source, dest_path + source.name if dest_path.endswith("/") else dest_path)]
        elif source.is_dir() and not source.is_symlink():
            entries = sorted(source.rglob("*"))
            if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in entries):
                raise ValueError("directory transfer supports regular files and directories only")
            destination = PurePosixPath(dest_path)
            directories = [destination, *(destination / path.relative_to(source)
                            for path in entries if path.is_dir())]
            files = [(path, str(destination / path.relative_to(source)))
                     for path in entries if path.is_file()]
        else:
            raise ValueError("transfer source must be a file or regular directory")
        failure = None
        try:
            for directory in directories:
                self.sandbox.files.make_dir(str(directory), user=self.config.user, request_timeout=timeout)
            for path, destination in files:
                self.sandbox.files.write(destination, path.read_bytes(),
                                         user=self.config.user, request_timeout=timeout)
        except Exception as error:
            failure = _infrastructure_failure("file transfer", error)
        if failure is not None:
            raise failure from None

    def get_template_vars(self):
        return {"cwd": self.config.cwd, "image": self.config.image}

    def cleanup(self):
        if self.sandbox is not None:
            failure = None
            try:
                self.sandbox.kill()
            except Exception as error:
                failure = _infrastructure_failure("cleanup", error)
            if failure is not None:
                raise failure from None
            self.sandbox = None
