"""Seal document-preserving DeepSeek-tokenized FineWeb chunks in source order."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def document_windows(tokens, sequence):
    """Each target is predicted once, with one input-token overlap at boundaries."""
    for start in range(0, len(tokens) - 1, sequence):
        yield start, min(sequence, len(tokens) - 1 - start)


def initialize(tokenizer, output, sequence, exclude):
    global TOKENIZER, OUTPUT, SEQUENCE, EXCLUDE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import pyarrow as pa
    from tokenizers import Tokenizer

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    TOKENIZER = Tokenizer.from_file(str(Path(tokenizer) / "tokenizer.json"))
    OUTPUT = Path(output)
    SEQUENCE = sequence
    EXCLUDE = set(exclude)


def cached_source(source):
    import fcntl

    source = Path(source)
    cache = OUTPUT / "source-cache"
    cache.mkdir(exist_ok=True)
    target = cache / source.name
    with (cache / (source.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = target.with_suffix(".source.json")
        if not receipt.exists():
            tmp = target.with_suffix(".copying")
            shutil.copyfile(source, tmp)
            if tmp.stat().st_size != source.stat().st_size:
                raise ValueError("incomplete source copy")
            digest = sha(tmp)
            tmp.replace(target)
            atomic(
                receipt, {"source": str(source), "bytes": target.stat().st_size, "sha256": digest}
            )
    return target


def prepare_chunk(task):
    import numpy as np
    import pyarrow.parquet as pq

    split, index, source, row_group = task
    out = OUTPUT / "chunks" / f"{split}-{index:06d}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        return json.loads((out / "COMPLETE.json").read_text())
    cached = cached_source(source)
    table = pq.ParquetFile(cached).read_row_group(row_group, columns=["text"])
    texts = table["text"].to_pylist()
    hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    selected = [i for i, h in enumerate(hashes) if h not in EXCLUDE and texts[i]]
    # Shuffle whole documents within each immutable source row group.
    order = np.random.default_rng(2234 + index).permutation(selected).tolist()
    records = []
    offset = 0
    targets = 0
    used_hashes = []
    with (out / "tokens.u32.tmp").open("wb") as stream:
        for first in range(0, len(order), 128):
            indices = order[first : first + 128]
            encoded = TOKENIZER.encode_batch([texts[i] for i in indices], add_special_tokens=False)
            for doc, enc in zip(indices, encoded, strict=True):
                tokens = [0] + enc.ids + [1]
                if len(tokens) < 2:
                    continue
                values = np.asarray(tokens, dtype="<u4")
                stream.write(values.tobytes())
                used_hashes.append(hashes[doc])
                for start, count in document_windows(tokens, SEQUENCE):
                    records.append((offset + start, count, targets))
                    targets += count
                offset += len(tokens)
    (out / "tokens.u32.tmp").replace(out / "tokens.u32")
    with (out / "windows.npy.tmp").open("wb") as stream:
        np.save(stream, np.asarray(records, dtype="<u8").reshape(-1, 3), allow_pickle=False)
    (out / "windows.npy.tmp").replace(out / "windows.npy")
    receipt = {
        "format": "v41-scratch-document-chunk-v1",
        "split": split,
        "index": index,
        "source": str(source),
        "row_group": row_group,
        "documents": len(used_hashes),
        "document_sha256": used_hashes,
        "windows": len(records),
        "targets": targets,
        "stored_tokens": offset,
        "sequence": SEQUENCE,
        "tokens_sha256": sha(out / "tokens.u32"),
        "windows_sha256": sha(out / "windows.npy"),
    }
    atomic(out / "COMPLETE.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=10_000_000_000)
    p.add_argument("--validation-tokens", type=int, default=1_000_000)
    p.add_argument("--sequence", type=int, default=2048)
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    import pyarrow.parquet as pq

    a.output.mkdir(parents=True, exist_ok=True)
    sources = sorted(a.source.glob("*.parquet"))
    if len(sources) < 2:
        raise ValueError("need disjoint validation and training sources")
    contract = {
        "format": "v41-scratch-pretraining-data-v1",
        "source_root": str(a.source.resolve()),
        "source_order": [x.name for x in sources],
        "source_bytes": [x.stat().st_size for x in sources],
        "preprocessing_sha256": sha(Path(__file__)),
        "tokenizer_sha256": sha(a.tokenizer / "tokenizer.json"),
        "vocab_size": 129280,
        "bos_id": 0,
        "eos_id": 1,
        "pad_id": 2,
        "training_targets": a.tokens,
        "validation_targets": a.validation_tokens,
        "sequence": a.sequence,
        "seed": 2234,
        "packing": "one contiguous document segment per row; overlap one input token; right padding masked",
        "split": "first source parquet reserved for validation; exact validation document hashes excluded from training",
    }
    cp = a.output / "CONTRACT.json"
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError("existing data contract differs")
    atomic(cp, contract)
    initialize(a.tokenizer, a.output, a.sequence, [])
    validation = []
    total = 0
    for rg in range(pq.ParquetFile(sources[0]).num_row_groups):
        receipt = prepare_chunk(("validation", rg, str(sources[0]), rg))
        validation.append(receipt)
        total += receipt["targets"]
        if total >= a.validation_tokens:
            break
    if total < a.validation_tokens:
        raise ValueError("validation source too short")
    atomic(
        a.output / "VALIDATION_READY.json",
        {"targets": a.validation_tokens, "chunks": len(validation), "available_targets": total},
    )
    exclude = [h for r in validation for h in r["document_sha256"]]

    def tasks():
        number = 0
        for source in sources[1:]:
            for rg in range(pq.ParquetFile(source).num_row_groups):
                yield ("train", number, str(source), rg)
                number += 1

    iterator = iter(tasks())
    pending = deque()
    total = 0
    count = 0
    pool = ProcessPoolExecutor(
        max_workers=a.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(a.tokenizer, a.output, a.sequence, exclude),
    )
    try:
        for _ in range(2 * a.workers):
            task = next(iterator, None)
            if task is not None:
                pending.append(pool.submit(prepare_chunk, task))
        while pending:
            receipt = pending.popleft().result()
            total += receipt["targets"]
            count += 1
            progress = {
                "prepared_at_unix": time.time(),
                "contiguous_training_chunks": count,
                "available_training_targets": total,
                "target": a.tokens,
            }
            atomic(a.output / "PROGRESS.json", progress)
            if count % 10 == 0 or count == 1:
                print(json.dumps(progress), flush=True)
            if total >= a.tokens:
                atomic(a.output / "TRAINING_READY.json", progress)
                break
            task = next(iterator, None)
            if task is not None:
                pending.append(pool.submit(prepare_chunk, task))
        if total < a.tokens:
            raise ValueError("source exhausted before training target budget")
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
