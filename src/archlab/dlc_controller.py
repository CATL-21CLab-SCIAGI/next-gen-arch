from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
LAUNCHER = "scripts/run_qwen38_quarter_fp4_dlc.sh"
DEFAULT_ALLOWED_REPO_ROOT = Path("/mnt/nas/evergreen")
DEFAULT_CONTROL_ROOT = Path(
    "/mnt/nas/evergreen/next-gen-arch/qwen38-quarter-fp4-fineweb100b-control"
)

_GENERATION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_ALLOWED_TOP_LEVEL_KEYS = {
    "schema_version",
    "generation",
    "action",
    "requested_at_utc",
    "requested_from",
    "repository",
    "launcher",
    "environment",
}
_ALLOWED_REPOSITORY_KEYS = {"root", "commit"}
_ALLOWED_ENVIRONMENT_KEYS = {
    "NGA_SOURCE_DATA",
    "NGA_SOURCE_MANIFEST",
    "NGA_DATA_ROOT",
    "NGA_TOKENIZER",
    "NGA_OUTPUT_ROOT",
    "NGA_PYTHON",
    "NGA_MEGATRON_ROOT",
    "NGA_EXPECTED_NODES",
    "NGA_GPUS_PER_NODE",
    "NGA_TOKENIZER_WORKERS",
    "NGA_EXPECTED_SOURCE_SHARDS",
    "NGA_VALIDATION_SOURCE",
    "NGA_EXPECTED_TRAIN_SHARDS",
    "NGA_EXPECTED_TRAIN_PARTS",
    "NGA_DOCUMENT_BATCH_SIZE",
    "NGA_DATA_WAIT_SECONDS",
    "NGA_PREPARE_DATA",
    "NGA_RUNTIME_PREFLIGHT",
    "NGA_PREFLIGHT_DATA_ROOT",
    "NGA_SEQUENCE_LENGTH",
    "NGA_MICRO_BATCH_SIZE",
    "NGA_GLOBAL_BATCH_SIZE",
    "NGA_TARGET_TRAIN_TOKENS",
    "NGA_CHECKPOINT_INTERVAL_TOKENS",
    "NGA_PROBE_STEPS",
    "NGA_CONTAINER_DIGEST",
    "NGA_PRECISION",
}
_REQUIRED_ENVIRONMENT_KEYS = {
    "NGA_SOURCE_DATA",
    "NGA_SOURCE_MANIFEST",
    "NGA_DATA_ROOT",
    "NGA_TOKENIZER",
    "NGA_OUTPUT_ROOT",
    "NGA_EXPECTED_NODES",
    "NGA_GPUS_PER_NODE",
    "NGA_TOKENIZER_WORKERS",
    "NGA_EXPECTED_TRAIN_PARTS",
    "NGA_DATA_WAIT_SECONDS",
    "NGA_PREPARE_DATA",
    "NGA_RUNTIME_PREFLIGHT",
    "NGA_PREFLIGHT_DATA_ROOT",
    "NGA_SEQUENCE_LENGTH",
    "NGA_MICRO_BATCH_SIZE",
    "NGA_GLOBAL_BATCH_SIZE",
    "NGA_TARGET_TRAIN_TOKENS",
    "NGA_CHECKPOINT_INTERVAL_TOKENS",
}
_POSITIVE_INTEGER_KEYS = {
    "NGA_EXPECTED_NODES",
    "NGA_GPUS_PER_NODE",
    "NGA_TOKENIZER_WORKERS",
    "NGA_EXPECTED_SOURCE_SHARDS",
    "NGA_EXPECTED_TRAIN_SHARDS",
    "NGA_EXPECTED_TRAIN_PARTS",
    "NGA_DOCUMENT_BATCH_SIZE",
    "NGA_DATA_WAIT_SECONDS",
    "NGA_SEQUENCE_LENGTH",
    "NGA_MICRO_BATCH_SIZE",
    "NGA_GLOBAL_BATCH_SIZE",
    "NGA_TARGET_TRAIN_TOKENS",
    "NGA_CHECKPOINT_INTERVAL_TOKENS",
}
_NONNEGATIVE_INTEGER_KEYS = {"NGA_PROBE_STEPS"}
_BOOLEAN_KEYS = {"NGA_PREPARE_DATA", "NGA_RUNTIME_PREFLIGHT"}
_PATH_PREFIXES = {
    "NGA_SOURCE_DATA": Path("/mnt/oss-dataset"),
    "NGA_SOURCE_MANIFEST": Path("/mnt/oss/datasets"),
    "NGA_DATA_ROOT": Path("/mnt/oss/datasets"),
    "NGA_TOKENIZER": Path("/mnt/oss/models"),
    "NGA_OUTPUT_ROOT": Path("/mnt/nas/evergreen"),
    "NGA_PREFLIGHT_DATA_ROOT": Path("/mnt/oss/datasets"),
}


@dataclass(frozen=True)
class LaunchRequest:
    generation: str
    repository_root: Path
    commit: str
    environment: dict[str, str]
    payload: dict[str, Any]
    digest: str


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    unknown = set(value) - expected
    if unknown:
        raise ValueError(f"unsupported {label} keys: {sorted(unknown)}")


def _as_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise ValueError(f"{label} must be a non-empty single-line string")
    return value


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _validate_timestamp(value: Any) -> str:
    timestamp = _as_str(value, "requested_at_utc")
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("requested_at_utc must include a timezone")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("requested_at_utc must be expressed in UTC")
    return timestamp


def validate_request(
    payload: Any,
    *,
    allowed_repo_root: Path = DEFAULT_ALLOWED_REPO_ROOT,
    expected_nodes: int | None = None,
    expected_gpus_per_node: int | None = None,
) -> LaunchRequest:
    if not isinstance(payload, dict):
        raise ValueError("launch request must be a JSON object")
    _require_exact_keys(payload, _ALLOWED_TOP_LEVEL_KEYS, "launch request")
    missing_top = _ALLOWED_TOP_LEVEL_KEYS - set(payload)
    if missing_top:
        raise ValueError(f"missing launch request keys: {sorted(missing_top)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if payload["action"] != "run":
        raise ValueError("action must be 'run'")
    generation = _as_str(payload["generation"], "generation")
    if not _GENERATION_RE.fullmatch(generation):
        raise ValueError("generation contains unsupported characters")
    _validate_timestamp(payload["requested_at_utc"])
    if _as_str(payload["requested_from"], "requested_from") != "dsw-evergreen":
        raise ValueError("requested_from must be dsw-evergreen")
    if payload["launcher"] != LAUNCHER:
        raise ValueError(f"launcher must be {LAUNCHER}")

    repository = payload["repository"]
    if not isinstance(repository, dict):
        raise ValueError("repository must be an object")
    _require_exact_keys(repository, _ALLOWED_REPOSITORY_KEYS, "repository")
    if set(repository) != _ALLOWED_REPOSITORY_KEYS:
        raise ValueError("repository must contain root and commit")
    repository_root = Path(_as_str(repository["root"], "repository.root"))
    if not repository_root.is_absolute() or not _is_within(repository_root, allowed_repo_root):
        raise ValueError(f"repository.root must be under {allowed_repo_root}")
    commit = _as_str(repository["commit"], "repository.commit")
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError("repository.commit must be a full lowercase SHA-1")

    raw_environment = payload["environment"]
    if not isinstance(raw_environment, dict):
        raise ValueError("environment must be an object")
    _require_exact_keys(raw_environment, _ALLOWED_ENVIRONMENT_KEYS, "environment")
    missing_environment = _REQUIRED_ENVIRONMENT_KEYS - set(raw_environment)
    if missing_environment:
        raise ValueError(f"missing environment keys: {sorted(missing_environment)}")
    environment = {
        key: _as_str(value, f"environment.{key}") for key, value in raw_environment.items()
    }

    for key in _POSITIVE_INTEGER_KEYS:
        if key in environment and (not environment[key].isdigit() or int(environment[key]) < 1):
            raise ValueError(f"{key} must be a positive integer")
    for key in _NONNEGATIVE_INTEGER_KEYS:
        if key in environment and (not environment[key].isdigit()):
            raise ValueError(f"{key} must be a non-negative integer")
    for key in _BOOLEAN_KEYS:
        if environment[key] not in {"0", "1"}:
            raise ValueError(f"{key} must be 0 or 1")
    if (
        "NGA_PRECISION" in environment
        and environment["NGA_PRECISION"] not in {"bf16", "fp4"}
    ):
        raise ValueError("NGA_PRECISION must be bf16 or fp4")
    for key, root in _PATH_PREFIXES.items():
        path = Path(environment[key])
        if not path.is_absolute() or not _is_within(path, root):
            raise ValueError(f"{key} must be under {root}")
    if "NGA_PYTHON" in environment and environment["NGA_PYTHON"] != "/opt/venv/bin/python":
        raise ValueError("NGA_PYTHON must use the container-owned /opt/venv/bin/python")
    if (
        "NGA_MEGATRON_ROOT" in environment
        and environment["NGA_MEGATRON_ROOT"] != "/opt/Megatron-Bridge/3rdparty/Megatron-LM"
    ):
        raise ValueError("NGA_MEGATRON_ROOT must use the container-owned Megatron checkout")

    nodes = int(environment["NGA_EXPECTED_NODES"])
    gpus_per_node = int(environment["NGA_GPUS_PER_NODE"])
    micro_batch = int(environment["NGA_MICRO_BATCH_SIZE"])
    global_batch = int(environment["NGA_GLOBAL_BATCH_SIZE"])
    target_tokens = int(environment["NGA_TARGET_TRAIN_TOKENS"])
    checkpoint_tokens = int(environment["NGA_CHECKPOINT_INTERVAL_TOKENS"])
    if expected_nodes is not None and nodes != expected_nodes:
        raise ValueError(f"request expects {nodes} nodes; controller owns {expected_nodes}")
    if expected_gpus_per_node is not None and gpus_per_node != expected_gpus_per_node:
        raise ValueError(
            f"request expects {gpus_per_node} GPUs per node; controller owns {expected_gpus_per_node}"
        )
    if global_batch % (nodes * gpus_per_node * micro_batch):
        raise ValueError("global batch must divide by the distributed micro batch")
    if checkpoint_tokens > target_tokens or target_tokens % checkpoint_tokens:
        raise ValueError("checkpoint interval must evenly divide target train tokens")

    return LaunchRequest(
        generation=generation,
        repository_root=repository_root,
        commit=commit,
        environment=environment,
        payload=payload,
        digest=_digest(payload),
    )


def _read_json(path: Path) -> Any:
    if path.stat().st_size > 1_048_576:
        raise ValueError(f"refusing oversized control file: {path}")
    return json.loads(path.read_text())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with temporary.open("x") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def verify_repository(request: LaunchRequest) -> None:
    root = request.repository_root
    if not root.is_dir():
        raise RuntimeError(f"repository does not exist: {root}")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    if head != request.commit:
        raise RuntimeError(f"repository HEAD is {head}, expected {request.commit}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout.splitlines()
    drift = [line for line in status if line not in {"?? .LAUNCH_READY", "?? repo-head.txt"}]
    if drift:
        raise RuntimeError(f"repository is not clean: {drift}")
    launcher = root / LAUNCHER
    if not launcher.is_file():
        raise RuntimeError(f"launcher is missing: {launcher}")


def publish_request(
    request_file: Path, control_root: Path, allowed_repo_root: Path
) -> dict[str, Any]:
    request = validate_request(_read_json(request_file), allowed_repo_root=allowed_repo_root)
    verify_repository(request)
    control_root.mkdir(parents=True, exist_ok=True)
    requests_root = control_root / "requests"
    requests_root.mkdir(parents=True, exist_ok=True)
    archived = requests_root / f"{request.generation}.json"
    canonical = _canonical_json(request.payload)
    try:
        descriptor = os.open(archived, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640)
    except FileExistsError:
        existing = _read_json(archived)
        if _canonical_json(existing) != canonical:
            raise RuntimeError(
                f"generation {request.generation} already has a different request"
            ) from None
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical)
            stream.flush()
            os.fsync(stream.fileno())
    _atomic_write_json(control_root / "desired.json", request.payload)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "generation": request.generation,
        "request_sha256": request.digest,
        "published_at_utc": _utc_now(),
        "published_from_host": socket.gethostname(),
        "desired": str(control_root / "desired.json"),
        "archived": str(archived),
    }
    _atomic_write_json(control_root / "last-publish.json", receipt)
    return receipt


class Controller:
    def __init__(
        self,
        *,
        control_root: Path,
        allowed_repo_root: Path,
        rank: int,
        world_size: int,
        expected_nodes: int,
        expected_gpus_per_node: int,
        poll_seconds: float,
    ) -> None:
        self.control_root = control_root
        self.allowed_repo_root = allowed_repo_root
        self.rank = rank
        self.world_size = world_size
        self.expected_nodes = expected_nodes
        self.expected_gpus_per_node = expected_gpus_per_node
        self.poll_seconds = poll_seconds
        self.stop_requested = False
        self.child: subprocess.Popen[bytes] | None = None
        self.last_error_fingerprint = ""
        self.started_at = _utc_now()

    @property
    def heartbeat_path(self) -> Path:
        return self.control_root / "controllers" / f"node-{self.rank:05d}.json"

    def request_stop(self, signum: int, _frame: Any) -> None:
        self.stop_requested = True
        self._event("controller_signal", signal=signum)
        if self.child is not None and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _event(self, event: str, **fields: Any) -> None:
        print(
            json.dumps(
                {
                    "event": event,
                    "time": _utc_now(),
                    "rank": self.rank,
                    **fields,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _heartbeat(self, state: str, **fields: Any) -> None:
        _atomic_write_json(
            self.heartbeat_path,
            {
                "schema_version": SCHEMA_VERSION,
                "job_id": os.environ.get("DLC_JOB_ID", "unknown"),
                "hostname": socket.gethostname(),
                "rank": self.rank,
                "world_size": self.world_size,
                "controller_started_at": self.started_at,
                "heartbeat_at": _utc_now(),
                "state": state,
                **fields,
            },
        )

    def _result_path(self, request: LaunchRequest) -> Path:
        return self.control_root / "runs" / request.generation / f"node-{self.rank:05d}.json"

    def _already_processed(self, request: LaunchRequest) -> bool:
        result_path = self._result_path(request)
        if not result_path.exists():
            return False
        result = _read_json(result_path)
        if result.get("request_sha256") != request.digest:
            raise RuntimeError(
                f"generation {request.generation} result digest does not match desired request"
            )
        self._heartbeat(
            "waiting_after_run",
            generation=request.generation,
            request_sha256=request.digest,
            prior_exit_code=result.get("exit_code"),
        )
        return True

    def _run(self, request: LaunchRequest) -> None:
        verify_repository(request)
        run_root = self.control_root / "runs" / request.generation
        run_root.mkdir(parents=True, exist_ok=True)
        started_at = _utc_now()
        _atomic_write_json(
            run_root / f"node-{self.rank:05d}.started.json",
            {
                "schema_version": SCHEMA_VERSION,
                "generation": request.generation,
                "request_sha256": request.digest,
                "job_id": os.environ.get("DLC_JOB_ID", "unknown"),
                "rank": self.rank,
                "hostname": socket.gethostname(),
                "started_at": started_at,
            },
        )
        self._heartbeat(
            "running",
            generation=request.generation,
            request_sha256=request.digest,
            child_started_at=started_at,
        )
        self._event(
            "launch_start",
            generation=request.generation,
            request_sha256=request.digest,
            repository=str(request.repository_root),
            commit=request.commit,
        )
        child_environment = os.environ.copy()
        child_environment.update(request.environment)
        child_environment["NGA_REPO_ROOT"] = str(request.repository_root)
        child_environment["NGA_EXPECTED_COMMIT"] = request.commit
        self.child = subprocess.Popen(
            ["bash", str(request.repository_root / LAUNCHER)],
            env=child_environment,
            start_new_session=True,
        )
        last_heartbeat = 0.0
        while self.child.poll() is None:
            now = time.monotonic()
            if now - last_heartbeat >= 30:
                self._heartbeat(
                    "running",
                    generation=request.generation,
                    request_sha256=request.digest,
                    child_pid=self.child.pid,
                    child_started_at=started_at,
                )
                last_heartbeat = now
            if self.stop_requested:
                break
            time.sleep(1)
        if self.stop_requested and self.child.poll() is None:
            deadline = time.monotonic() + 30
            while self.child.poll() is None and time.monotonic() < deadline:
                time.sleep(1)
            if self.child.poll() is None:
                os.killpg(self.child.pid, signal.SIGKILL)
        exit_code = self.child.wait()
        self.child = None
        finished_at = _utc_now()
        result = {
            "schema_version": SCHEMA_VERSION,
            "generation": request.generation,
            "request_sha256": request.digest,
            "job_id": os.environ.get("DLC_JOB_ID", "unknown"),
            "rank": self.rank,
            "hostname": socket.gethostname(),
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": exit_code,
            "succeeded": exit_code == 0,
        }
        _atomic_write_json(self._result_path(request), result)
        self._heartbeat(
            "waiting_after_run",
            generation=request.generation,
            request_sha256=request.digest,
            prior_exit_code=exit_code,
        )
        self._event(
            "launch_exit",
            generation=request.generation,
            request_sha256=request.digest,
            exit_code=exit_code,
        )

    def serve(self) -> int:
        if self.world_size != self.expected_nodes:
            raise RuntimeError(
                f"DLC injected WORLD_SIZE={self.world_size}; controller expects {self.expected_nodes}"
            )
        self.control_root.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        self._event(
            "controller_start",
            control_root=str(self.control_root),
            world_size=self.world_size,
            expected_gpus_per_node=self.expected_gpus_per_node,
        )
        while not self.stop_requested:
            desired = self.control_root / "desired.json"
            if not desired.exists():
                self._heartbeat("waiting_for_request")
                time.sleep(self.poll_seconds)
                continue
            try:
                request = validate_request(
                    _read_json(desired),
                    allowed_repo_root=self.allowed_repo_root,
                    expected_nodes=self.expected_nodes,
                    expected_gpus_per_node=self.expected_gpus_per_node,
                )
                if not self._already_processed(request):
                    self._run(request)
                self.last_error_fingerprint = ""
            except Exception as error:  # keep the paid allocation alive for a corrected request
                fingerprint = f"{type(error).__name__}: {error}"
                if fingerprint != self.last_error_fingerprint:
                    self._event("controller_error", error=fingerprint)
                    self.last_error_fingerprint = fingerprint
                self._heartbeat("control_error", error=fingerprint)
            if not self.stop_requested:
                time.sleep(self.poll_seconds)
        self._heartbeat("stopped")
        self._event("controller_stop")
        return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent, DSW-controlled PAI DLC launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish", help="atomically publish a launch request")
    publish.add_argument("--request", type=Path, required=True)
    publish.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    publish.add_argument("--allowed-repo-root", type=Path, default=DEFAULT_ALLOWED_REPO_ROOT)

    serve = subparsers.add_parser("serve", help="keep a DLC allocation alive and run requests")
    serve.add_argument(
        "--control-root",
        type=Path,
        default=Path(os.environ.get("NGA_CONTROL_ROOT", DEFAULT_CONTROL_ROOT)),
    )
    serve.add_argument(
        "--allowed-repo-root",
        type=Path,
        default=Path(os.environ.get("NGA_CONTROLLER_ALLOWED_REPO_ROOT", DEFAULT_ALLOWED_REPO_ROOT)),
    )
    serve.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "-1")))
    serve.add_argument("--world-size", type=_positive_int, default=os.environ.get("WORLD_SIZE"))
    serve.add_argument(
        "--expected-nodes",
        type=_positive_int,
        default=os.environ.get("NGA_CONTROLLER_EXPECTED_NODES", "4"),
    )
    serve.add_argument(
        "--expected-gpus-per-node",
        type=_positive_int,
        default=os.environ.get("NGA_CONTROLLER_GPUS_PER_NODE", "8"),
    )
    serve.add_argument(
        "--poll-seconds",
        type=_positive_int,
        default=os.environ.get("NGA_CONTROLLER_POLL_SECONDS", "10"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "publish":
        receipt = publish_request(args.request, args.control_root, args.allowed_repo_root)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.rank < 0:
        raise SystemExit("PAI DLC must inject a non-negative RANK")
    controller = Controller(
        control_root=args.control_root,
        allowed_repo_root=args.allowed_repo_root,
        rank=args.rank,
        world_size=args.world_size,
        expected_nodes=args.expected_nodes,
        expected_gpus_per_node=args.expected_gpus_per_node,
        poll_seconds=args.poll_seconds,
    )
    return controller.serve()


if __name__ == "__main__":
    sys.exit(main())
