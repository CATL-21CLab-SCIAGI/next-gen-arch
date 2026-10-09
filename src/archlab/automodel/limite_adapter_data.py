"""One sealed order of document-contained Math-v2 windows for both variants."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

WINDOW_POLICIES = ("drop-tail", "overlap-final-window")


def document_window_offsets(length, context, *, tail_policy="drop-tail"):
    """Keep legacy windows, optionally adding one document-ending full window.

    The opt-in policy supervises the actual ending without padding, crossing a
    document boundary or changing the trainer's fixed shape. It duplicates some
    already supervised targets; short documents still require a separate recipe.
    """
    if context < 1 or length < 1 or tail_policy not in WINDOW_POLICIES:
        raise ValueError("invalid document window policy or geometry")
    starts = list(range(0, length - context, context))
    if starts and tail_policy == "overlap-final-window":
        final = length - context - 1
        if starts[-1] != final:
            starts.append(final)
    return starts


def prepare(source, output, heldout, context=2048, *, tail_policy="drop-tail"):
    document_window_offsets(1, context, tail_policy=tail_policy)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "READY.json").exists():
        previous = json.loads((output / "READY.json").read_text())
        if previous.get("tail_policy", "drop-tail") != tail_policy:
            raise ValueError("existing window dataset uses a different tail policy; choose a new output")
        if tail_policy != "drop-tail" and (
            previous["context"] != context or Path(previous["source"]).resolve() != source.resolve()
            or previous["heldout_sha256"] != hashlib.sha256(heldout.read_bytes()).hexdigest()
        ):
            raise ValueError("existing corrected window dataset has a different data contract")
        return
    ready = json.loads((source / "DATA_READY.json").read_text())
    blocked = {json.loads(line)["problem_sha256"] for line in heldout.read_text().splitlines()}
    train = []
    valid = []
    for _kind, prefixes in ready["prefixes"].items():
        for prefix in prefixes:
            if "/train-" in prefix:
                train.append(prefix)
            elif "/validation-" in prefix:
                valid.append(prefix)
    train = sorted(set(train))
    valid = sorted(set(valid))
    prefixes = sorted(set(train + valid))
    index = {p: i for i, p in enumerate(prefixes)}
    excluded = 0
    counts = {}
    records = {}
    coverage = {}
    for split, items in [("train", train), ("validation", valid)]:
        dest = output / f"{split}.npy"
        chunks = []
        totals = dict(documents=0, retained_documents=0, excluded_short_documents=0,
                      excluded_short_tokens=0, unique_target_tokens=0,
                      repeated_target_tokens=0, additional_tail_windows=0)
        for prefix in items:
            offset = 0
            spec = []
            with gzip.open(source / (prefix + ".metadata.jsonl.gz"), "rt") as f:
                for line in f:
                    row = json.loads(line)
                    length = row["tokens"]
                    if row["problem_sha256"] in blocked:
                        excluded += 1
                    else:
                        starts = document_window_offsets(length, context, tail_policy=tail_policy)
                        spec.extend(
                            (index[prefix], offset + start)
                            for start in starts
                        )
                        totals["documents"] += 1
                        if starts:
                            unique = starts[-1] + context
                            totals["retained_documents"] += 1
                            totals["unique_target_tokens"] += unique
                            totals["repeated_target_tokens"] += len(starts) * context - unique
                            totals["additional_tail_windows"] += len(starts) - (length - 1) // context
                        else:
                            totals["excluded_short_documents"] += 1
                            totals["excluded_short_tokens"] += length
                    offset += length
            if (source / (prefix + ".bin")).stat().st_size != offset * 4:
                raise ValueError("metadata/token payload size mismatch: " + prefix)
            if spec:
                chunks.append(np.asarray(spec, dtype=np.int64))
        array = np.concatenate(chunks) if chunks else np.empty((0, 2), dtype=np.int64)
        np.random.default_rng(42 if split == "train" else 43).shuffle(array)
        np.save(dest, array)
        counts[split] = len(array)
        records[dest.name] = hashlib.sha256(dest.read_bytes()).hexdigest()
        coverage[split] = totals
    manifest = dict(
        source=str(source),
        context=context,
        prefixes=prefixes,
        counts=counts,
        excluded_rl_heldout_documents=excluded,
        files=records,
        objective="all-token next-token warmup; no cross-document attention; no padding",
        tokenizer="Limite shared vocabulary; explicit Violetto math/reasoning serialization",
        heldout_sha256=hashlib.sha256(heldout.read_bytes()).hexdigest(),
        source_contract=ready["contract_sha256"],
    )
    if tail_policy != "drop-tail":
        manifest.update(
            format="archlab-limite-math-windows-v2",
            tail_policy=tail_policy,
            coverage=coverage,
            short_document_policy="exclude documents with at most context tokens; no padding",
            objective=("all-token next-token warmup; document-contained fixed windows plus "
                       "one overlapping final window when needed; all retained document endings supervised"),
            compatibility="new data order and repeated targets; never resume a legacy data cursor unchanged",
        )
        contract = {key: manifest[key] for key in (
            "format", "source_contract", "context", "prefixes", "files", "heldout_sha256", "tail_policy",
        )}
        manifest["window_contract_sha256"] = hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest()
    (output / "READY.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "prefixes"}), flush=True)


class MathWindows:
    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.spec = json.loads((self.root / "READY.json").read_text())
        self.order = np.load(self.root / f"{split}.npy", mmap_mode="r")
        self.maps = {}

    @property
    def contract(self):
        """Historical runs retain their contract; corrected order has its own identity."""
        if self.spec.get("format") == "archlab-limite-math-windows-v2" and not self.spec.get("window_contract_sha256"):
            raise ValueError("corrected window dataset lacks its data-order contract")
        return self.spec.get("window_contract_sha256", self.spec["source_contract"])

    def __len__(self):
        return len(self.order)

    def __getitem__(self, i):
        part, start = self.order[i % len(self.order)]
        if part not in self.maps:
            self.maps[part] = np.memmap(
                Path(self.spec["source"]) / (self.spec["prefixes"][part] + ".bin"),
                mode="r",
                dtype="<i4",
            )
            if len(self.maps) > 32:
                self.maps.pop(next(iter(self.maps)))
        return np.array(self.maps[part][start : start + self.spec["context"] + 1], dtype=np.int64)


class WindowPrefetcher:
    """Read one future step without advancing the sealed training cursor.

    Only the reader thread accesses the dataset's mmap cache. Scheduling is
    deterministic and independent of RNG state; resuming needs only the
    checkpoint's existing global step. Outstanding reads never commit a step.
    """

    def __init__(self, dataset, *, step, rank, world_size, microbatch, accumulation):
        if step < 0 or not 0 <= rank < world_size or min(microbatch, accumulation) < 1:
            raise ValueError("invalid window prefetch geometry")
        self.dataset = dataset
        self.rank, self.world = rank, world_size
        self.microbatch, self.accumulation = microbatch, accumulation
        self.step = step
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="math-windows")
        self.pending = self.pool.submit(self._read, step)

    def _read(self, step):
        batch = self.world * self.microbatch * self.accumulation
        return np.stack(
            [
                np.stack(
                    [
                        self.dataset[int(row)]
                        for row in step * batch
                        + (acc * self.world + self.rank) * self.microbatch
                        + np.arange(self.microbatch)
                    ]
                )
                for acc in range(self.accumulation)
            ]
        )

    def get(self, step):
        if step != self.step:
            raise ValueError("window prefetch cursor differs from training step")
        result = self.pending.result()
        self.step += 1
        self.pending = self.pool.submit(self._read, self.step)
        return result

    def close(self):
        self.pending.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--tail-policy", choices=WINDOW_POLICIES, default="drop-tail")
    a = p.parse_args()
    prepare(a.source, a.output, a.heldout, a.context, tail_policy=a.tail_policy)


if __name__ == "__main__":
    main()
