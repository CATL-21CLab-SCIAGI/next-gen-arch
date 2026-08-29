"""Content identities for source trees, datasets, tokenizers, and model state."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

DATASET_MANIFEST_SCHEMA = 1
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class ProvenanceError(ValueError):
    """Raised when an artifact identity is unsafe, incomplete, or does not match."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for stable content identities."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, *, chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ProvenanceError(f"manifest path must be safe and relative: {value!r}")
    return path


@dataclass(frozen=True)
class DatasetFile:
    path: str
    size: int | None
    sha256: str

    def validate(self) -> None:
        _safe_relative_path(self.path)
        if self.size is not None and self.size < 0:
            raise ProvenanceError(f"negative size for {self.path}")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256.lower()
        ):
            raise ProvenanceError(f"invalid SHA-256 for {self.path}")


@dataclass(frozen=True)
class DatasetManifest:
    dataset: str
    revision: str
    files: tuple[DatasetFile, ...]
    schema_version: int = DATASET_MANIFEST_SCHEMA
    kind: str = "dataset-content-manifest"

    def validate(self) -> None:
        if self.schema_version != DATASET_MANIFEST_SCHEMA:
            raise ProvenanceError(
                f"unsupported dataset manifest schema: {self.schema_version}"
            )
        if self.kind != "dataset-content-manifest":
            raise ProvenanceError(f"unexpected dataset manifest kind: {self.kind!r}")
        if not self.dataset or not self.revision:
            raise ProvenanceError("dataset and revision must not be empty")
        if not self.files:
            raise ProvenanceError("dataset manifest has no files")
        paths = [item.path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ProvenanceError("dataset manifest paths must be sorted and unique")
        for item in self.files:
            item.validate()

    @property
    def inventory_sha256(self) -> str:
        return stable_json_sha256([asdict(item) for item in self.files])

    @property
    def identity_sha256(self) -> str:
        return stable_json_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "dataset": self.dataset,
            "revision": self.revision,
            "inventory_sha256": self.inventory_sha256,
            "files": [asdict(item) for item in self.files],
        }


def create_dataset_manifest(
    root: str | Path,
    *,
    dataset: str,
    revision: str,
    patterns: Sequence[str] = ("**/*",),
) -> DatasetManifest:
    """Hash a deterministic file inventory rooted at ``root``."""

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = {
        path.resolve()
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file()
    }
    if not paths:
        raise ProvenanceError(f"no files matched under {root}")
    records = []
    for path in sorted(paths):
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ProvenanceError(f"manifest input escapes root: {path}") from error
        records.append(
            DatasetFile(
                path=relative.as_posix(),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    manifest = DatasetManifest(dataset=dataset, revision=revision, files=tuple(records))
    manifest.validate()
    return manifest


def write_dataset_manifest(path: str | Path, manifest: DatasetManifest) -> None:
    manifest.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _manifest_from_json(path: Path) -> DatasetManifest:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ProvenanceError(f"dataset manifest must be an object: {path}")
    files = value.get("files")
    if not isinstance(files, list):
        raise ProvenanceError("dataset manifest files must be a list")
    manifest = DatasetManifest(
        schema_version=int(value.get("schema_version", 0)),
        kind=str(value.get("kind", "")),
        dataset=str(value.get("dataset", "")),
        revision=str(value.get("revision", "")),
        files=tuple(
            DatasetFile(
                path=str(item["path"]),
                size=None if item.get("size") is None else int(item["size"]),
                sha256=str(item["sha256"]).lower(),
            )
            for item in files
        ),
    )
    manifest.validate()
    recorded_inventory = value.get("inventory_sha256")
    if recorded_inventory != manifest.inventory_sha256:
        raise ProvenanceError("dataset manifest inventory_sha256 does not match its records")
    return manifest


def _manifest_from_sha256(path: Path, *, dataset_root_name: str) -> DatasetManifest:
    """Read a GNU sha256sum ledger, including the existing FineWeb relay format."""

    records: list[DatasetFile] = []
    prefix = f"{dataset_root_name}/"
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        digest, separator, raw_name = line.partition("  ")
        if not separator:
            raise ProvenanceError(f"invalid sha256 ledger line {line_number}: {path}")
        name = raw_name.removeprefix(prefix)
        records.append(DatasetFile(path=name, size=None, sha256=digest.lower()))
    manifest = DatasetManifest(
        dataset=dataset_root_name,
        revision=f"sha256-ledger:{sha256_file(path)}",
        files=tuple(sorted(records, key=lambda item: item.path)),
    )
    manifest.validate()
    return manifest


def load_dataset_manifest(
    path: str | Path,
    *,
    dataset_root_name: str = "dataset",
) -> DatasetManifest:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return _manifest_from_json(path)
    return _manifest_from_sha256(path, dataset_root_name=dataset_root_name)


def verify_dataset_manifest(
    root: str | Path,
    manifest_path: str | Path,
    *,
    mode: str = "metadata",
    required_files: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Verify a pinned dataset identity without hiding the verification strength.

    ``metadata`` checks the exact file set and recorded byte sizes. ``full`` also
    rehashes every file. A legacy sha256sum ledger has no size records, but still
    pins content and can be fully verified.
    """

    if mode not in {"metadata", "full"}:
        raise ProvenanceError("dataset verification mode must be metadata or full")
    root = Path(root).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_dataset_manifest(manifest_path, dataset_root_name=root.name)
    expected = {item.path: item for item in manifest.files}
    selected = sorted(set(expected) if required_files is None else set(required_files))
    unknown = set(selected) - set(expected)
    if unknown:
        raise ProvenanceError(
            "required files are absent from the dataset manifest: " + ", ".join(sorted(unknown))
        )
    if required_files is None:
        visible = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        extras = visible - set(expected)
        missing = set(expected) - visible
        if extras or missing:
            raise ProvenanceError(
                "dataset inventory mismatch: "
                f"extra={sorted(extras)[:5]} missing={sorted(missing)[:5]}"
            )
    total_bytes = 0
    for relative in selected:
        item = expected[relative]
        path = (root / _safe_relative_path(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ProvenanceError(f"dataset file escapes root: {relative}") from error
        if not path.is_file():
            raise ProvenanceError(f"dataset file is missing: {relative}")
        size = path.stat().st_size
        total_bytes += size
        if item.size is not None and size != item.size:
            raise ProvenanceError(
                f"dataset size mismatch for {relative}: {size} != {item.size}"
            )
        if mode == "full":
            observed = sha256_file(path)
            if observed != item.sha256:
                raise ProvenanceError(
                    f"dataset content mismatch for {relative}: {observed} != {item.sha256}"
                )
    return {
        "dataset": manifest.dataset,
        "revision": manifest.revision,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_identity_sha256": manifest.identity_sha256,
        "inventory_sha256": manifest.inventory_sha256,
        "verification_mode": mode,
        "content_rehashed": mode == "full",
        "files_verified": len(selected),
        "bytes_verified": total_bytes,
    }


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={root}", *args],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def source_provenance(repository: str | Path) -> dict[str, Any]:
    """Hash tracked changes and untracked file content without exposing either."""

    repository = Path(repository).expanduser().resolve()
    status = _git_output(repository, "status", "--porcelain=v1", "--untracked-files=all")
    diff = subprocess.run(
        ["git", "-c", f"safe.directory={repository}", "diff", "--binary", "HEAD"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    untracked_output = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    untracked = sorted(Path(os.fsdecode(path)) for path in untracked_output.split(b"\0") if path)
    untracked_digest = hashlib.sha256()
    worktree_digest = hashlib.sha256(b"tracked-diff\0" + diff)
    for relative in untracked:
        path = repository / relative
        if path.is_symlink():
            content = b"symlink\0" + os.fsencode(os.readlink(path))
        elif path.is_file():
            content = b"file\0" + path.read_bytes()
        else:
            content = b"other\0"
        name = os.fsencode(relative)
        framed = len(name).to_bytes(8, "big") + name + len(content).to_bytes(8, "big") + content
        untracked_digest.update(framed)
        worktree_digest.update(framed)
    return {
        "source_commit": _git_output(repository, "rev-parse", "HEAD"),
        "source_dirty": bool(status),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_untracked_files": [str(path) for path in untracked],
        "source_untracked_sha256": untracked_digest.hexdigest(),
        "source_worktree_sha256": worktree_digest.hexdigest(),
    }


def hash_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    include_names: set[str] | None = None,
) -> str:
    """Hash tensor names, shapes, dtypes, and bytes in a deterministic order."""

    digest = hashlib.sha256()
    selected = sorted(
        (name, tensor)
        for name, tensor in named_tensors
        if include_names is None or name in include_names
    )
    if not selected:
        raise ProvenanceError("no tensors were selected for initialization hashing")
    for name, tensor in selected:
        detached = tensor.detach().contiguous()
        header = {
            "name": name,
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
        }
        # ``bytes(UntypedStorage)`` avoids a NumPy dependency and preserves raw
        # BF16/FP8 bit patterns. ``contiguous`` above guarantees the storage has
        # no unrelated slice prefix or suffix.
        payload = bytes(detached.view(torch.uint8).cpu().untyped_storage())
        header_bytes = canonical_json_bytes(header)
        digest.update(len(header_bytes).to_bytes(8, "big"))
        digest.update(header_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def hash_tokenizer_vocabulary(tokenizer: Any, *, padded_vocab_size: int | None = None) -> str:
    """Hash every token's decoded bytes, not merely file names or token lengths."""

    vocabulary_size = int(tokenizer.get_vocab_size())
    if padded_vocab_size is not None and padded_vocab_size < vocabulary_size:
        raise ProvenanceError("padded vocabulary is smaller than the tokenizer vocabulary")
    digest = hashlib.sha256()
    digest.update(vocabulary_size.to_bytes(8, "big"))
    digest.update((padded_vocab_size or vocabulary_size).to_bytes(8, "big"))
    encoder = getattr(tokenizer, "enc", None)
    for token_id in range(vocabulary_size):
        if encoder is not None and hasattr(encoder, "decode_single_token_bytes"):
            payload = encoder.decode_single_token_bytes(token_id)
        else:
            payload = tokenizer.decode([token_id]).encode("utf-8")
        digest.update(token_id.to_bytes(8, "big"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
