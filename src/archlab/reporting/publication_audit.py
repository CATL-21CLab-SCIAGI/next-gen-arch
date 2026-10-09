"""Audit a Git tree for publication without reading ignored workspace artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOTS = frozenset({".github", "src", "tests", "recipes", "scripts", "docs"})
ROOT_FILES = frozenset({
    ".gitignore", ".gitmodules", ".python-version", "AGENTS.md", "CITATION.cff",
    "LICENSE", "README.md", "pyproject.toml", "uv.lock",
})
REGISTERED_SUBMODULES = {"verl": "https://github.com/XiaomiMiMo/verl.git"}
FORBIDDEN_PARTS = frozenset({
    ".runtime", "results", "result", ".git", ".aws", ".ssh", ".agents", ".codex",
    "__pycache__", ".venv", "node_modules", "checkpoints", "mlruns", "wandb",
})
FORBIDDEN_SUFFIXES = (
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".idx", ".parquet", ".arrow",
    ".jsonl", ".jsonl.gz", ".log", ".pem", ".key", ".p12", ".pickle", ".pkl",
)
SECRET_MARKERS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "credential_url": re.compile(rb"https?://[^\s/:@]+:[^\s/@]+@"),
    "private_registry": re.compile(rb"[a-z0-9-]+-registry-vpc\.[a-z0-9.-]+"),
}


def inspect_blob(path: str, mode: str, content: bytes) -> list[str]:
    """Return reason names only; never echo suspected credentials."""
    parts = PurePosixPath(path).parts
    reasons = []
    if not parts or (parts[0] not in ROOTS and path not in ROOT_FILES):
        reasons.append("unapproved_root")
    if any(part in FORBIDDEN_PARTS for part in parts):
        reasons.append("workspace_artifact")
    if path.endswith(FORBIDDEN_SUFFIXES) or (parts[-1].startswith(".env") and parts[-1] != ".env.example"):
        reasons.append("data_or_secret_file")
    if mode not in {"100644", "100755"}:
        reasons.append("non_regular_file")
    if len(content) > 5 * 1024 * 1024:
        reasons.append("oversize_blob")
    for name, marker in SECRET_MARKERS.items():
        if marker.search(content):
            reasons.append(name)
    return reasons


def audit_tree(root: Path, revision: str = "HEAD") -> dict:
    def git(*args: str, payload: bytes | None = None) -> bytes:
        return subprocess.run(["git", "-C", str(root), *args], input=payload,
                              capture_output=True, check=True).stdout

    resolved = git("rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()
    entries, gitlinks = [], []
    for record in git("ls-tree", "-rz", resolved).split(b"\0"):
        if not record:
            continue
        header, path = record.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        if kind == "commit" and mode == "160000":
            gitlinks.append((path.decode(), oid))
        elif kind == "blob":
            entries.append((mode, oid, path.decode()))
        else:
            raise ValueError("publication tree contains an unsupported object")
    objects = git("cat-file", "--batch", payload="".join(oid + "\n" for _, oid, _ in entries).encode())
    cursor = 0
    manifest, findings = [], []
    for mode, oid, path in entries:
        end = objects.index(b"\n", cursor)
        header = objects[cursor:end].decode().split()
        if header[:2] != [oid, "blob"]:
            raise ValueError("unexpected Git object response")
        size = int(header[2])
        start = end + 1
        content = objects[start:start + size]
        cursor = start + size + 1
        manifest.append({"path": path, "bytes": size, "sha256": hashlib.sha256(content).hexdigest()})
        reasons = inspect_blob(path, mode, content)
        if reasons:
            findings.append({"path": path, "reasons": reasons})
    submodules = []
    if gitlinks or any(path == ".gitmodules" for _, _, path in entries):
        expected_config = []
        for path, oid in gitlinks:
            url = REGISTERED_SUBMODULES.get(path)
            if url is None:
                findings.append({"path": path, "reasons": ["unapproved_submodule"]})
                continue
            expected_config.extend([
                (f"submodule.{path}.path", path),
                (f"submodule.{path}.url", url),
            ])
            submodules.append({"name": path, "path": path, "url": url, "revision": oid})
        try:
            config = git("config", "--null", "--no-includes", "--blob",
                         f"{resolved}:.gitmodules", "--list")
            actual_config = [tuple(row.decode().split("\n", 1))
                             for row in config.split(b"\0") if row]
            config_matches = bool(gitlinks) and sorted(actual_config) == sorted(expected_config)
        except (subprocess.CalledProcessError, UnicodeDecodeError):
            config_matches = False
        if not config_matches:
            findings.append({"path": ".gitmodules", "reasons": ["invalid_submodule_registration"]})
    return {
        "revision": resolved,
        "passed": not findings,
        "files": len(manifest),
        "bytes": sum(row["bytes"] for row in manifest),
        "findings": findings,
        "manifest": manifest,
        "submodules": submodules,
        "scope": "Selected Git tree and registered submodule pins only; excludes upstream source, "
                 "historical/remote surfaces and ignored local data.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_tree(args.root, args.revision)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("revision", "passed", "files", "bytes", "findings")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
