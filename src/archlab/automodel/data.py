"""One-pass raw-token windows using AutoModel's existing indexed-data reader.

This is an explicit new data-order contract, not a change to frozen speedrun or
the earlier cyclic rank readers. Targets traverse the manifest-ordered stream
exactly once; adjacent windows share one input/target boundary token.
"""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from nemo_automodel.components.datasets.llm.megatron.indexed_dataset import IndexedDataset


class OnePassTokenWindows:
    def __init__(self, prefixes: list[Path], sequence_length: int):
        if not prefixes or sequence_length < 1:
            raise ValueError("nonempty prefixes and positive sequence length required")
        self.sequence_length = sequence_length
        self.datasets = [IndexedDataset(str(prefix), mmap=True) for prefix in prefixes]
        self.ends = [0]
        for prefix, dataset in zip(prefixes, self.datasets):
            if dataset.index.dtype != np.int32:
                raise ValueError("the agreed raw token corpus must use int32")
            count = int(dataset.sequence_lengths.sum(dtype=np.int64))
            if count < 1 or Path(str(prefix) + ".bin").stat().st_size != count * 4:
                raise ValueError(f"indexed data size mismatch: {prefix}")
            self.ends.append(self.ends[-1] + count)
        self.total_tokens = self.ends[-1]

    def __len__(self):
        return (self.total_tokens - 1) // self.sequence_length

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError("one-pass corpus exhausted; wrapping is forbidden")
        start = index * self.sequence_length
        remaining = self.sequence_length + 1
        pieces = []
        while remaining:
            part = bisect_right(self.ends, start) - 1
            offset = start - self.ends[part]
            take = min(remaining, self.ends[part + 1] - start)
            dataset = self.datasets[part]
            pieces.append(dataset.bin_reader.read(dataset.index.dtype, take, offset * 4))
            start += take
            remaining -= take
        values = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        return torch.from_numpy(np.array(values, dtype=np.int64, copy=True))

    def full_microbatches(self, world_size: int, micro_batch: int) -> int:
        if world_size < 1 or micro_batch < 1:
            raise ValueError("world and micro batch must be positive")
        return len(self) // (world_size * micro_batch)

    def batch(self, cursor: int, *, rank: int, world_size: int, micro_batch: int, device="cpu"):
        if not 0 <= rank < world_size or not 0 <= cursor < self.full_microbatches(world_size, micro_batch):
            raise IndexError("invalid rank/cursor or one-pass corpus exhausted")
        first = (cursor * world_size + rank) * micro_batch
        raw = torch.stack([self[first + i] for i in range(micro_batch)]).to(device=device)
        return {"input_ids": raw[:, :-1].contiguous(), "labels": raw[:, 1:].contiguous()}

    def accounting(self, world_size: int, micro_batch: int) -> dict:
        batches = self.full_microbatches(world_size, micro_batch)
        consumed_targets = batches * world_size * micro_batch * self.sequence_length
        return {"source_tokens": self.total_tokens, "full_microbatches": batches,
                "consumed_target_tokens": consumed_targets,
                "unused_final_target_tokens": self.total_tokens - 1 - consumed_targets,
                "initial_context_only_tokens": 1, "wrapped_tokens": 0}


def load_fineweb_windows(root: Path, checkpoint: Path, sequence_length: int):
    root = root.resolve()
    raw = (root / "DATA_READY.json").read_bytes()
    manifest = json.loads(raw)
    tokenizer_sha = hashlib.sha256((checkpoint / "tokenizer.json").read_bytes()).hexdigest()
    if manifest.get("tokenizer_sha256") != tokenizer_sha:
        raise ValueError("FineWeb token IDs do not match the pretrained tokenizer")
    splits = []
    declared_splits = []
    for directory, key, count_key in (("train", "train_parts", "train_tokens"),
                                      ("val", "valid_parts", "validation_tokens")):
        values = manifest.get(key)
        if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
            raise ValueError(f"manifest is missing {key}")
        prefixes = [(Path(v) if Path(v).is_absolute() else root / v).resolve() for v in values]
        discovered = {p.with_suffix("").resolve() for p in (root / directory).glob("*.idx")}
        if len(set(prefixes)) != len(prefixes) or set(prefixes) != discovered:
            raise ValueError(f"manifest membership mismatch: {key}")
        dataset = OnePassTokenWindows(prefixes, sequence_length)
        if dataset.total_tokens != manifest.get(count_key):
            raise ValueError(f"manifest token count mismatch: {count_key}")
        splits.append(dataset)
        declared_splits.append(prefixes)
    if set(declared_splits[0]) & set(declared_splits[1]):
        raise ValueError("training/validation overlap")
    provenance = {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
                  "tokenizer_sha256": tokenizer_sha, "order": "manifest-order-contiguous-unpadded-windows",
                  "train_prefixes": list(map(str, declared_splits[0])),
                  "validation_prefixes": list(map(str, declared_splits[1]))}
    return *splits, provenance
