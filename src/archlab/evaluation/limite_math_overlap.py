"""Audit public benchmark text against the actual local SFT/RL data lineage.

This is a bounded exact/format-normalized audit. It cannot rule out paraphrases
or undisclosed pretraining contamination, and raw source matches are checked
against retained documents and the SFT window cursor before being called seen.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import time
import unicodedata
from pathlib import Path

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.preprocessing.nemotron_math import problem_key


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("−", "-")
    text = re.sub(r"\\(?:left|right)\b", "", text)
    # Keep arithmetic signs; the audit does not conflate negative/positive data.
    return "".join(re.findall(r"[a-z0-9]+|[+-]", text))


def audit_rl(cases: list[dict], data: Path) -> dict:
    matches, counts = [], {}
    for split in ("train", "heldout"):
        rows = [json.loads(line) for line in (data / f"{split}.jsonl").read_text().splitlines() if line]
        counts[split] = len(rows)
        for row in rows:
            prompt = normalized_text(row["prompt"])
            for case in cases:
                exact = row["problem_sha256"] == problem_key(case["question"])
                equivalent = normalized_text(case["question"]) in prompt
                if exact or equivalent:
                    matches.append(dict(problem_id=case["id"], split=split, uuid=row["uuid"],
                                        method="exact NFC-whitespace hash" if exact else "format-normalized full question contained in prompt"))
    return dict(rows=counts, matches=matches,
                split_manifest_sha256=sha256_file(data / "SPLIT.json"),
                files={name: sha256_file(data / name) for name in ("train.jsonl", "heldout.jsonl")})


def audit_sft(cases: list[dict], data: Path, *, tokens: int = 10_000_000_000,
              global_windows_per_step: int = 128, raw_source_root: Path | None = None) -> dict:
    import numpy as np
    import pyarrow.parquet as pq

    started = time.perf_counter()
    ready = json.loads((data / "READY.json").read_text())
    source = Path(ready["source"])
    manifest = json.loads((source / "manifest.json").read_text())
    targets = {normalized_text(case["question"]): case for case in cases}
    hashes = {problem_key(case["question"]): case for case in cases}
    candidates, scanned = [], 0
    # Exact hashes are already sealed in retained-document metadata. A moved
    # raw corpus is optional candidate discovery for normalized text only;
    # every candidate still has to match the original sealed metadata hash.
    for spec in (manifest["sources"] if raw_source_root is not None else []):
        path = raw_source_root / Path(spec["path"]).name
        if path.stat().st_size != spec["bytes"]:
            raise ValueError("auxiliary SFT source size changed")
        parquet = pq.ParquetFile(path)
        row_offset = 0
        for batch in parquet.iter_batches(batch_size=32768, columns=["problem"], use_threads=True):
            problems = batch.column(0).to_pylist()
            for index, problem in enumerate(problems):
                case = hashes.get(problem_key(problem))
                method = "exact NFC-whitespace problem hash"
                if case is None:
                    case = targets.get(normalized_text(problem))
                    method = "full format-normalized problem text"
                if case is not None:
                    candidates.append(dict(problem_id=case["id"], source=path.name,
                                           source_row=row_offset + index, method=method,
                                           problem_sha256=problem_key(problem), source_problem=problem))
            row_offset += len(problems)
        if row_offset != spec["rows"]:
            raise ValueError("SFT source row count changed")
        scanned += row_offset
        print(json.dumps(dict(source=path.name, scanned_rows=row_offset,
                              candidates=len(candidates), seconds=time.perf_counter() - started)), flush=True)
    candidate_map = {key: dict(problem_id=case['id'], method='exact sealed NFC-whitespace problem hash')
                     for key, case in hashes.items()}
    candidate_map.update({row["problem_sha256"]: row for row in candidates})
    order = np.load(data / "train.npy", mmap_mode="r")
    if sha256_file(data / "train.npy") != ready["files"]["train.npy"]:
        raise ValueError("sealed SFT window order changed")
    context = ready["context"]
    # The final step reads a full batch even when its remaining loss budget is
    # smaller. Treat that entire input batch as exposed, conservatively.
    cursor = math.ceil(tokens / (context * global_windows_per_step)) * global_windows_per_step
    active_order = order[:cursor]
    retained, seen = [], []
    for prefix_index, prefix in enumerate(ready["prefixes"]):
        if "/train-" not in prefix:
            continue
        offset = 0
        rows = []
        metadata_path = source / (prefix + ".metadata.jsonl.gz")
        part = json.loads((metadata_path.parent / 'READY.json').read_text())
        expected = next(row for row in part['files'] if row['name'] == metadata_path.name)
        if (metadata_path.stat().st_size != expected['bytes']
                or sha256_file(metadata_path) != expected['sha256']):
            raise ValueError('sealed retained SFT metadata changed')
        with gzip.open(metadata_path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                if row["problem_sha256"] in candidate_map:
                    rows.append((row, offset))
                offset += row["tokens"]
        if not rows:
            continue
        indices = np.flatnonzero(active_order[:, 0] == prefix_index)
        offsets = active_order[indices, 1]
        for row, offset in rows:
            case = candidate_map[row["problem_sha256"]]
            selected = indices[(offsets >= offset) & (offsets < offset + row["tokens"] - context)]
            entry = dict(problem_id=case["problem_id"], source=row["source"], source_row=row["source_row"],
                         prefix=prefix, problem_sha256=row["problem_sha256"], document_tokens=row["tokens"],
                         method=case["method"], windows_read_by_10b_cursor=len(selected),
                         earliest_window=int(selected.min()) if len(selected) else None)
            retained.append(entry)
            if len(selected):
                seen.append(entry)
    return dict(source_rows_scanned=scanned, raw_source_matches=candidates,
                normalized_candidate_scan=raw_source_root is not None,
                exact_hash_scan='all checksum-verified retained training metadata',
                retained_train_document_matches=retained, seen_by_10b_cursor=seen,
                seen_problem_ids=sorted({row["problem_id"] for row in seen}),
                sft_tokens=tokens, read_window_cursor=cursor, window_context=context,
                cursor_rule="conservative full final batch; includes at most127 windows without positive loss",
                manifest_sha256=sha256_file(source / "manifest.json"),
                window_ready_sha256=sha256_file(data / "READY.json"),
                train_order_sha256=ready["files"]["train.npy"], seconds=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sft-data", type=Path, required=True)
    parser.add_argument("--rl-data", type=Path, required=True)
    parser.add_argument("--raw-source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(line) for line in (args.bundle / "cases.jsonl").read_text().splitlines() if line]
    report = dict(format="archlab-limite-benchmark-overlap-audit-v1",
                  cases_sha256=sha256_file(args.bundle / "cases.jsonl"),
                  rl=audit_rl(cases, args.rl_data),
                  limitation="Exact and format-normalized matching only. Paraphrases and undisclosed base pretraining cannot be excluded.")
    atomic_write_json(args.output.with_suffix(".rl-partial.json"), report)
    report["sft"] = audit_sft(cases, args.sft_data, raw_source_root=args.raw_source_root)
    atomic_write_json(args.output, report)
    print(json.dumps(dict(rl_matches=len(report["rl"]["matches"]),
                          sft_seen_problem_ids=report["sft"]["seen_problem_ids"])), flush=True)


if __name__ == "__main__":
    main()
