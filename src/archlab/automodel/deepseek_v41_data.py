"""Read only a sealed native-math pilot, with one label shift and exact masks."""

from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from nemo_automodel.components.datasets.llm.megatron.indexed_dataset import IndexedDataset


class MathPilot:
    def __init__(self, root: Path, *, expected_split, expected_budget=None, order_seed=2234):
        self.root = root.resolve(strict=True)
        self.manifest = json.loads((root / "PILOT_READY.json").read_text())
        if self.manifest["format"] != "archlab-native-math-window-pilot-v1" or self.manifest["split"] != expected_split:
            raise ValueError("wrong pilot format or split")
        raw = (root / "windows.jsonl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.manifest["windows_sha256"]:
            raise ValueError("pilot windows changed after sealing")
        self.windows = [json.loads(line) for line in raw.splitlines()]
        # Mix effort/tool strata at optimizer-step scale instead of training
        # through long homogeneous runs of source parts. Source selection and
        # exact masks remain those of the sealed pilot.
        self.order_seed = order_seed
        random.Random(order_seed).shuffle(self.windows)
        self.context = self.manifest["context"]
        self.source = Path(self.manifest["source"]).resolve(strict=True)
        ready_sha = hashlib.sha256((self.source / "DATA_READY.json").read_bytes()).hexdigest()
        if ready_sha != self.manifest["source_ready_sha256"]:
            raise ValueError("source corpus changed after pilot preparation")
        if len(self.windows) != self.manifest["windows"]:
            raise ValueError("pilot window count mismatch")
        total = sum(r["targets"] for r in self.windows)
        if total != self.manifest["supervised_tokens"] or (expected_budget is not None and total != expected_budget):
            raise ValueError("pilot does not match the requested supervised-token budget")
        self.readers = OrderedDict()

    def __len__(self):
        return len(self.windows)

    def batch(self, index, *, device, smoke_context=None, pad_to_full=False):
        context = self.context if smoke_context is None else smoke_context
        if index >= len(self):
            context = context if pad_to_full else min(context, 128)
            inputs = torch.zeros(1, context, dtype=torch.int64, device="cpu")
            labels = torch.full((1, context), -100, dtype=torch.int64, device="cpu")
            return inputs.to(device), labels.to(device), 0
        if index < 0 or context < 2 or context > self.context:
            raise ValueError("invalid pilot index/context")
        record = self.windows[index]
        prefix = (self.source / record["prefix"]).resolve()
        if self.source not in prefix.parents:
            raise ValueError("pilot data escapes the tokenized corpus")
        if prefix not in self.readers:
            self.readers[prefix] = IndexedDataset(str(prefix), mmap=True)
            if self.readers[prefix].index.dtype != np.int32:
                raise ValueError("expected native int32 tokens")
            while len(self.readers) > 4:
                self.readers.popitem(last=False)
        self.readers.move_to_end(prefix)
        start = 0 if smoke_context is None else max(0, record["assistant_label_spans"][0][0] - 16)
        length = min(context, record["length"] - start)
        if not pad_to_full and smoke_context is None:
            context = min(context, max(128, 1 << (length - 1).bit_length()))
        inputs = torch.zeros(1, context, dtype=torch.int64, device="cpu")
        labels = torch.full((1, context), -100, dtype=torch.int64, device="cpu")
        raw = self.readers[prefix].get(record["sequence"], offset=record["start"] + start, length=length + 1)
        tokens = torch.from_numpy(np.asarray(raw, dtype=np.int64).copy())
        if tokens.numel() != length + 1 or bool(((tokens < 0) | (tokens >= 129280)).any()):
            raise ValueError("invalid native token window")
        inputs[0, :length] = tokens[:-1]
        count = 0
        for lo, hi in record["assistant_label_spans"]:
            lo, hi = max(0, lo - start), min(length, hi - start)
            if lo < hi:
                labels[0, lo:hi] = tokens[lo + 1:hi + 1]
                count += hi - lo
        if count != int((labels != -100).sum()) or (smoke_context is None and count != record["targets"]):
            raise ValueError("assistant targets were lost or counted twice")
        return inputs.to(device), labels.to(device), count
