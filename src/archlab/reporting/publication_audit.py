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
    ".gitignore", ".python-version", "AGENTS.md", "CITATION.cff",
    "LICENSE", "README.md", "pyproject.toml", "uv.lock",
})
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
    entries = []
    for record in git("ls-tree", "-rz", resolved).split(b"\0"):
        if not record:
            continue
        header, path = record.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        if kind != "blob":
            raise ValueError("publication tree contains a submodule or unsupported object")
        entries.append((mode, oid, path.decode()))
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
    return {
        "revision": resolved,
        "passed": not findings,
        "files": len(manifest),
        "bytes": sum(row["bytes"] for row in manifest),
        "findings": findings,
        "manifest": manifest,
        "scope": "Selected Git tree only; excludes historical/remote surfaces and ignored local data.",
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
