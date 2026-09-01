"""Build Megatron indexed FineWeb data with a model-native tokenizer.

The public ``fineweb100B-gpt2`` shards are a compact, deterministic text
source, but their uint16 token ids cannot be consumed by Qwen's 151k-token
vocabulary.  This module decodes every GPT-2 document and writes a Megatron
indexed dataset using the pinned Qwen tokenizer.  Work is partitioned into
independent prefixes so a multi-node DLC job can prepare data in parallel and
resume completed parts without merging hundreds of gigabytes.

The FineWeb-Edu ``sample-100BT`` mirror is already available as Parquet.  It is
handled directly rather than round-tripping through GPT-2 token ids.  The last
source shard is held out as validation, and a generated source inventory pins
the ordered file set, byte sizes, row counts, and published token counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FINEWEB_MAGIC = 20_240_520
FINEWEB_VERSION = 1
FINEWEB_HEADER_INTS = 256
GPT2_EOT = 50_256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_tokens(path: Path) -> np.memmap:
    header = np.fromfile(path, dtype=np.int32, count=FINEWEB_HEADER_INTS)
    if len(header) != FINEWEB_HEADER_INTS or int(header[0]) != FINEWEB_MAGIC:
        raise ValueError(f"invalid FineWeb header: {path}")
    if int(header[1]) != FINEWEB_VERSION:
        raise ValueError(f"unsupported FineWeb version in {path}: {int(header[1])}")
    token_count = int(header[2])
    expected_bytes = FINEWEB_HEADER_INTS * 4 + token_count * np.dtype(np.uint16).itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"FineWeb shard length mismatch for {path}: {path.stat().st_size} != {expected_bytes}"
        )
    return np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
        offset=FINEWEB_HEADER_INTS * 4,
        shape=(token_count,),
    )


@dataclass(frozen=True)
class PartJob:
    prefix: str
    marker: str
    sources: tuple[str, ...]
    tokenizer: str
    tokenizer_sha256: str
    source_manifest_sha256: str
    qwen_eos_id: int
    document_batch_size: int


@dataclass(frozen=True)
class ParquetPartJob:
    prefix: str
    marker: str
    sources: tuple[str, ...]
    tokenizer: str
    tokenizer_sha256: str
    source_manifest_sha256: str
    qwen_eos_id: int
    document_batch_size: int
    text_column: str
    source_token_column: str


def _validate_complete_part(job: PartJob) -> dict[str, Any] | None:
    marker = Path(job.marker)
    if not marker.is_file():
        return None
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "sources": list(job.sources),
        "tokenizer_sha256": job.tokenizer_sha256,
        "source_manifest_sha256": job.source_manifest_sha256,
        "qwen_eos_id": job.qwen_eos_id,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"completed part contract changed for {marker}: {key}")
    for suffix in (".bin", ".idx"):
        path = Path(job.prefix + suffix)
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"completed indexed dataset artifact is missing: {path}")
    return payload


def _convert_part(job: PartJob) -> dict[str, Any]:
    complete = _validate_complete_part(job)
    if complete is not None:
        return complete

    import tiktoken
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    from tokenizers import Tokenizer

    prefix = Path(job.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = Path(job.prefix + ".bin")
    idx_path = Path(job.prefix + ".idx")
    # An absent marker means these exact files belong to an interrupted build
    # in this new dataset namespace. IndexedDatasetBuilder truncates the bin;
    # finalize atomically replaces the small index file.
    idx_path.unlink(missing_ok=True)
    builder = IndexedDatasetBuilder(str(bin_path), dtype=np.int32)
    qwen = Tokenizer.from_file(str(Path(job.tokenizer) / "tokenizer.json"))
    gpt2 = tiktoken.get_encoding("gpt2")
    started = time.time()
    input_tokens = 0
    output_tokens = 0
    documents = 0

    for source_name in job.sources:
        source = Path(source_name)
        tokens = _source_tokens(source)
        input_tokens += len(tokens)
        boundaries = np.flatnonzero(tokens == GPT2_EOT)
        starts = np.concatenate((np.array([0], dtype=np.int64), boundaries + 1))
        ends = np.concatenate((boundaries, np.array([len(tokens)], dtype=np.int64)))
        nonempty = ends > starts
        starts, ends = starts[nonempty], ends[nonempty]

        for offset in range(0, len(starts), job.document_batch_size):
            batch_starts = starts[offset : offset + job.document_batch_size]
            batch_ends = ends[offset : offset + job.document_batch_size]
            texts = [
                gpt2.decode(np.asarray(tokens[start:end], dtype=np.int64).tolist())
                for start, end in zip(batch_starts, batch_ends, strict=True)
            ]
            encoded = qwen.encode_batch(texts, add_special_tokens=False)
            lengths = [len(item.ids) + 1 for item in encoded]
            flat = np.fromiter(
                (token for item in encoded for token in (*item.ids, job.qwen_eos_id)),
                dtype=np.int32,
                count=sum(lengths),
            )
            # One IndexedDataset document per batch keeps writes coarse while
            # retaining every source-document length and EOS boundary.
            builder.add_document(flat, lengths)
            output_tokens += len(flat)
            documents += len(lengths)

    builder.finalize(str(idx_path))
    payload = {
        "format": "megatron-indexed-dataset",
        "dtype": "int32",
        "prefix": str(prefix),
        "sources": list(job.sources),
        "source_shards": len(job.sources),
        "source_tokens": input_tokens,
        "output_tokens": output_tokens,
        "documents": documents,
        "tokenizer": str(Path(job.tokenizer)),
        "tokenizer_sha256": job.tokenizer_sha256,
        "source_manifest_sha256": job.source_manifest_sha256,
        "qwen_eos_id": job.qwen_eos_id,
        "bin_bytes": bin_path.stat().st_size,
        "idx_bytes": idx_path.stat().st_size,
        "elapsed_seconds": time.time() - started,
        "completed_at_unix": time.time(),
    }
    _write_json(Path(job.marker), payload)
    return payload


def _validate_complete_parquet_part(job: ParquetPartJob) -> dict[str, Any] | None:
    marker = Path(job.marker)
    if not marker.is_file():
        return None
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "source_format": "parquet",
        "sources": list(job.sources),
        "tokenizer_sha256": job.tokenizer_sha256,
        "source_manifest_sha256": job.source_manifest_sha256,
        "qwen_eos_id": job.qwen_eos_id,
        "text_column": job.text_column,
        "source_token_column": job.source_token_column,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"completed parquet part contract changed for {marker}: {key}")
    for suffix in (".bin", ".idx"):
        path = Path(job.prefix + suffix)
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"completed indexed dataset artifact is missing: {path}")
    return payload


def _convert_parquet_part(job: ParquetPartJob) -> dict[str, Any]:
    complete = _validate_complete_parquet_part(job)
    if complete is not None:
        return complete

    import pyarrow.parquet as pq
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    from tokenizers import Tokenizer

    prefix = Path(job.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = Path(job.prefix + ".bin")
    idx_path = Path(job.prefix + ".idx")
    idx_path.unlink(missing_ok=True)
    builder = IndexedDatasetBuilder(str(bin_path), dtype=np.int32)
    qwen = Tokenizer.from_file(str(Path(job.tokenizer) / "tokenizer.json"))
    started = time.time()
    input_tokens = 0
    output_tokens = 0
    documents = 0

    for source_name in job.sources:
        source = Path(source_name)
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(
            batch_size=job.document_batch_size,
            columns=[job.text_column, job.source_token_column],
            use_threads=False,
        ):
            texts = batch.column(0).to_pylist()
            published_counts = batch.column(1).to_pylist()
            if any(not isinstance(text, str) or not text for text in texts):
                raise RuntimeError(f"empty or non-string text encountered in {source}")
            if any(count is None or int(count) < 0 for count in published_counts):
                raise RuntimeError(f"invalid source token count encountered in {source}")
            input_tokens += sum(int(count) for count in published_counts)
            encoded = qwen.encode_batch(texts, add_special_tokens=False)
            lengths = [len(item.ids) + 1 for item in encoded]
            flat = np.fromiter(
                (token for item in encoded for token in (*item.ids, job.qwen_eos_id)),
                dtype=np.int32,
                count=sum(lengths),
            )
            builder.add_document(flat, lengths)
            output_tokens += len(flat)
            documents += len(lengths)

    builder.finalize(str(idx_path))
    payload = {
        "dataset": "FineWeb-Edu sample-100BT",
        "source_format": "parquet",
        "format": "megatron-indexed-dataset",
        "dtype": "int32",
        "prefix": str(prefix),
        "sources": list(job.sources),
        "source_shards": len(job.sources),
        "source_tokens": input_tokens,
        "output_tokens": output_tokens,
        "documents": documents,
        "tokenizer": str(Path(job.tokenizer)),
        "tokenizer_sha256": job.tokenizer_sha256,
        "source_manifest_sha256": job.source_manifest_sha256,
        "qwen_eos_id": job.qwen_eos_id,
        "text_column": job.text_column,
        "source_token_column": job.source_token_column,
        "bin_bytes": bin_path.stat().st_size,
        "idx_bytes": idx_path.stat().st_size,
        "elapsed_seconds": time.time() - started,
        "completed_at_unix": time.time(),
    }
    _write_json(Path(job.marker), payload)
    return payload


def _assigned_jobs(args: argparse.Namespace) -> list[PartJob]:
    source_root = args.source_root.expanduser().resolve()
    pattern = "fineweb_train_*.bin" if args.split == "train" else "fineweb_val_*.bin"
    sources = sorted(source_root.glob(pattern))
    if len(sources) != args.expected_shards:
        raise RuntimeError(
            f"expected {args.expected_shards} {args.split} shards under {source_root}, "
            f"found {len(sources)}"
        )
    tokenizer = args.tokenizer.expanduser().resolve()
    tokenizer_json = tokenizer / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)
    source_manifest = args.source_manifest.expanduser().resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)

    total_workers = args.nodes * args.workers
    jobs = []
    output = args.output_root.expanduser().resolve() / args.split
    for local_worker in range(args.workers):
        global_worker = args.node_rank * args.workers + local_worker
        assigned = tuple(str(path) for path in sources[global_worker::total_workers])
        if not assigned:
            continue
        prefix = output / f"part-{global_worker:05d}"
        jobs.append(
            PartJob(
                prefix=str(prefix),
                marker=str(prefix.with_suffix(".json")),
                sources=assigned,
                tokenizer=str(tokenizer),
                tokenizer_sha256=_sha256(tokenizer_json),
                source_manifest_sha256=_sha256(source_manifest),
                qwen_eos_id=args.qwen_eos_id,
                document_batch_size=args.document_batch_size,
            )
        )
    return jobs


def _assigned_parquet_jobs(args: argparse.Namespace) -> list[ParquetPartJob]:
    source_root = args.source_root.expanduser().resolve()
    source_manifest = args.source_manifest.expanduser().resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "fineweb-edu-parquet-inventory-v1":
        raise RuntimeError(f"unsupported FineWeb-Edu source manifest: {source_manifest}")
    if Path(manifest.get("source_root", "")).resolve() != source_root:
        raise RuntimeError("FineWeb-Edu source root changed from the pinned inventory")
    validation_source = manifest.get("validation_source")
    entries = manifest.get("shards", [])
    selected = [
        entry
        for entry in entries
        if (entry.get("name") == validation_source) == (args.split == "val")
    ]
    if len(selected) != args.expected_shards:
        raise RuntimeError(
            f"expected {args.expected_shards} {args.split} parquet shards in {source_manifest}, "
            f"found {len(selected)}"
        )
    sources = []
    for entry in selected:
        path = source_root / str(entry["name"])
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"FineWeb-Edu source artifact changed: {path}")
        sources.append(path)

    tokenizer = args.tokenizer.expanduser().resolve()
    tokenizer_json = tokenizer / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)
    total_workers = args.nodes * args.workers
    jobs = []
    output = args.output_root.expanduser().resolve() / args.split
    for local_worker in range(args.workers):
        global_worker = args.node_rank * args.workers + local_worker
        assigned = tuple(str(path) for path in sources[global_worker::total_workers])
        if not assigned:
            continue
        prefix = output / f"part-{global_worker:05d}"
        jobs.append(
            ParquetPartJob(
                prefix=str(prefix),
                marker=str(prefix.with_suffix(".json")),
                sources=assigned,
                tokenizer=str(tokenizer),
                tokenizer_sha256=_sha256(tokenizer_json),
                source_manifest_sha256=_sha256(source_manifest),
                qwen_eos_id=args.qwen_eos_id,
                document_batch_size=args.document_batch_size,
                text_column=args.text_column,
                source_token_column=args.source_token_column,
            )
        )
    return jobs


def _convert(args: argparse.Namespace) -> None:
    if not 0 <= args.node_rank < args.nodes:
        raise ValueError("node-rank must be in [0, nodes)")
    if min(args.nodes, args.workers, args.expected_shards, args.document_batch_size) < 1:
        raise ValueError(
            "nodes, workers, expected-shards, and document-batch-size must be positive"
        )
    jobs = _assigned_jobs(args)
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        payloads = list(pool.map(_convert_part, jobs))
    node_marker = args.output_root.expanduser().resolve() / (
        f"{args.split}.node-{args.node_rank:05d}.json"
    )
    _write_json(
        node_marker,
        {
            "split": args.split,
            "node_rank": args.node_rank,
            "nodes": args.nodes,
            "workers": args.workers,
            "parts": [payload["prefix"] for payload in payloads],
            "source_shards": sum(payload["source_shards"] for payload in payloads),
            "source_tokens": sum(payload["source_tokens"] for payload in payloads),
            "output_tokens": sum(payload["output_tokens"] for payload in payloads),
            "documents": sum(payload["documents"] for payload in payloads),
            "completed_at_unix": time.time(),
        },
    )
    print(json.dumps(json.loads(node_marker.read_text(encoding="utf-8")), sort_keys=True))


def _convert_parquet(args: argparse.Namespace) -> None:
    if not 0 <= args.node_rank < args.nodes:
        raise ValueError("node-rank must be in [0, nodes)")
    if min(args.nodes, args.workers, args.expected_shards, args.document_batch_size) < 1:
        raise ValueError(
            "nodes, workers, expected-shards, and document-batch-size must be positive"
        )
    jobs = _assigned_parquet_jobs(args)
    if not jobs:
        raise RuntimeError(f"node {args.node_rank} was assigned no parquet work")
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        payloads = list(pool.map(_convert_parquet_part, jobs))
    node_marker = args.output_root.expanduser().resolve() / (
        f"{args.split}.node-{args.node_rank:05d}.json"
    )
    _write_json(
        node_marker,
        {
            "split": args.split,
            "source_format": "parquet",
            "node_rank": args.node_rank,
            "nodes": args.nodes,
            "workers": args.workers,
            "parts": [payload["prefix"] for payload in payloads],
            "source_shards": sum(payload["source_shards"] for payload in payloads),
            "source_tokens": sum(payload["source_tokens"] for payload in payloads),
            "output_tokens": sum(payload["output_tokens"] for payload in payloads),
            "documents": sum(payload["documents"] for payload in payloads),
            "completed_at_unix": time.time(),
        },
    )
    print(json.dumps(json.loads(node_marker.read_text(encoding="utf-8")), sort_keys=True))


def _inventory_parquet(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    source_root = args.source_root.expanduser().resolve()
    sources = sorted(source_root.glob(args.pattern))
    if len(sources) != args.expected_shards:
        raise RuntimeError(
            f"expected {args.expected_shards} parquet shards under {source_root}, "
            f"found {len(sources)}"
        )
    if args.validation_source not in {path.name for path in sources}:
        raise RuntimeError(f"validation source is not in the inventory: {args.validation_source}")
    shards = []
    total_rows = 0
    total_source_tokens = 0
    for path in sources:
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        source_tokens = 0
        for batch in parquet.iter_batches(
            batch_size=1_000_000,
            columns=[args.source_token_column],
            use_threads=True,
        ):
            values = batch.column(0).to_pylist()
            if any(value is None or int(value) < 0 for value in values):
                raise RuntimeError(f"invalid source token count encountered in {path}")
            source_tokens += sum(int(value) for value in values)
        shards.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "rows": rows,
                "source_tokens": source_tokens,
            }
        )
        total_rows += rows
        total_source_tokens += source_tokens
    payload = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-100BT",
        "format": "fineweb-edu-parquet-inventory-v1",
        "source_root": str(source_root),
        "pattern": args.pattern,
        "validation_source": args.validation_source,
        "text_column": args.text_column,
        "source_token_column": args.source_token_column,
        "total_shards": len(shards),
        "total_rows": total_rows,
        "total_source_tokens": total_source_tokens,
        "shards": shards,
        "completed_at_unix": time.time(),
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


def _summarize(args: argparse.Namespace) -> None:
    root = args.output_root.expanduser().resolve()
    train_markers = sorted((root / "train").glob("part-*.json"))
    valid_markers = sorted((root / "val").glob("part-*.json"))
    if len(train_markers) != args.train_parts or len(valid_markers) != args.valid_parts:
        raise RuntimeError(
            f"indexed dataset is incomplete: train={len(train_markers)}/{args.train_parts}, "
            f"val={len(valid_markers)}/{args.valid_parts}"
        )
    parts = [json.loads(path.read_text(encoding="utf-8")) for path in train_markers]
    validation = [json.loads(path.read_text(encoding="utf-8")) for path in valid_markers]
    identities = {
        (part["tokenizer_sha256"], part["source_manifest_sha256"], part["qwen_eos_id"])
        for part in (*parts, *validation)
    }
    if len(identities) != 1:
        raise RuntimeError("retokenized parts do not share one tokenizer/source identity")
    for part in (*parts, *validation):
        for suffix in (".bin", ".idx"):
            path = Path(part["prefix"] + suffix)
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"missing indexed artifact: {path}")
    tokenizer_sha256, source_manifest_sha256, qwen_eos_id = identities.pop()
    datasets = {part.get("dataset", "FineWeb100B") for part in (*parts, *validation)}
    if len(datasets) != 1:
        raise RuntimeError("retokenized parts do not share one dataset identity")
    payload = {
        "dataset": datasets.pop(),
        "format": "megatron-indexed-dataset",
        "tokenizer_sha256": tokenizer_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "qwen_eos_id": qwen_eos_id,
        "train_parts": [part["prefix"] for part in parts],
        "valid_parts": [part["prefix"] for part in validation],
        "train_source_shards": sum(part["source_shards"] for part in parts),
        "train_source_tokens": sum(part["source_tokens"] for part in parts),
        "train_tokens": sum(part["output_tokens"] for part in parts),
        "validation_tokens": sum(part["output_tokens"] for part in validation),
        "documents": sum(part["documents"] for part in parts),
        "completed_at_unix": time.time(),
    }
    if payload["train_tokens"] < args.required_train_tokens:
        raise RuntimeError(
            f"retokenized corpus has {payload['train_tokens']} tokens, fewer than required "
            f"{args.required_train_tokens}"
        )
    _write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


def _validate(args: argparse.Namespace) -> None:
    ready = args.ready.expanduser().resolve()
    payload = json.loads(ready.read_text(encoding="utf-8"))
    train_parts = payload.get("train_parts", [])
    valid_parts = payload.get("valid_parts", [])
    if len(train_parts) != args.train_parts or len(valid_parts) != args.valid_parts:
        raise RuntimeError(
            f"ready dataset part count changed: train={len(train_parts)}/{args.train_parts}, "
            f"val={len(valid_parts)}/{args.valid_parts}"
        )
    if payload.get("train_tokens", 0) < args.required_train_tokens:
        raise RuntimeError(
            f"ready dataset has {payload.get('train_tokens')} tokens, fewer than required "
            f"{args.required_train_tokens}"
        )
    expected_hashes = {
        "source_manifest_sha256": _sha256(args.source_manifest.expanduser().resolve()),
        "tokenizer_sha256": _sha256(args.tokenizer.expanduser().resolve()),
    }
    for key, expected in expected_hashes.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"ready dataset identity changed: {key}")
    for prefix in (*train_parts, *valid_parts):
        for suffix in (".bin", ".idx", ".json"):
            artifact = Path(prefix + suffix)
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                raise RuntimeError(f"ready dataset artifact is missing: {artifact}")
    validated = dict(payload)
    validated.update(
        {
            "ready_path": str(ready),
            "validated_at_unix": time.time(),
        }
    )
    _write_json(args.output, validated)
    print(json.dumps(validated, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    convert = sub.add_parser("convert")
    convert.add_argument("--source-root", required=True, type=Path)
    convert.add_argument("--source-manifest", required=True, type=Path)
    convert.add_argument("--output-root", required=True, type=Path)
    convert.add_argument("--tokenizer", required=True, type=Path)
    convert.add_argument("--split", choices=("train", "val"), required=True)
    convert.add_argument("--expected-shards", required=True, type=int)
    convert.add_argument("--nodes", required=True, type=int)
    convert.add_argument("--node-rank", required=True, type=int)
    convert.add_argument("--workers", type=int, default=8)
    convert.add_argument("--document-batch-size", type=int, default=512)
    convert.add_argument("--qwen-eos-id", type=int, default=151_643)
    parquet = sub.add_parser("convert-parquet")
    parquet.add_argument("--source-root", required=True, type=Path)
    parquet.add_argument("--source-manifest", required=True, type=Path)
    parquet.add_argument("--output-root", required=True, type=Path)
    parquet.add_argument("--tokenizer", required=True, type=Path)
    parquet.add_argument("--split", choices=("train", "val"), required=True)
    parquet.add_argument("--expected-shards", required=True, type=int)
    parquet.add_argument("--nodes", required=True, type=int)
    parquet.add_argument("--node-rank", required=True, type=int)
    parquet.add_argument("--workers", type=int, default=8)
    parquet.add_argument("--document-batch-size", type=int, default=512)
    parquet.add_argument("--qwen-eos-id", type=int, default=151_643)
    parquet.add_argument("--text-column", default="text")
    parquet.add_argument("--source-token-column", default="token_count")
    inventory = sub.add_parser("inventory-parquet")
    inventory.add_argument("--source-root", required=True, type=Path)
    inventory.add_argument("--pattern", default="*.parquet")
    inventory.add_argument("--expected-shards", required=True, type=int)
    inventory.add_argument("--validation-source", required=True)
    inventory.add_argument("--text-column", default="text")
    inventory.add_argument("--source-token-column", default="token_count")
    inventory.add_argument("--output", required=True, type=Path)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--output-root", required=True, type=Path)
    summarize.add_argument("--train-parts", required=True, type=int)
    summarize.add_argument("--valid-parts", type=int, default=1)
    summarize.add_argument("--required-train-tokens", type=int, default=100_000_000_000)
    summarize.add_argument("--output", required=True, type=Path)
    validate = sub.add_parser("validate")
    validate.add_argument("--ready", required=True, type=Path)
    validate.add_argument("--source-manifest", required=True, type=Path)
    validate.add_argument("--tokenizer", required=True, type=Path)
    validate.add_argument("--train-parts", required=True, type=int)
    validate.add_argument("--valid-parts", type=int, default=1)
    validate.add_argument("--required-train-tokens", type=int, default=100_000_000_000)
    validate.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.action == "convert":
        _convert(args)
    elif args.action == "convert-parquet":
        _convert_parquet(args)
    elif args.action == "inventory-parquet":
        _inventory_parquet(args)
    elif args.action == "summarize":
        _summarize(args)
    else:
        _validate(args)


if __name__ == "__main__":
    main()
