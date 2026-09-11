"""Deterministic raw-int32 batches; caller supplies the data-parallel ownership."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


class BinaryTokenBatches:
    """Rank-local deterministic iterator over raw int32 Megatron ``.bin`` tokens."""

    def __init__(
        self,
        prefixes: list[Path],
        *,
        batch_size: int,
        sequence_len: int,
        start_batch: int,
        device: torch.device,
        repeat_window_batches: int | None = None,
    ):
        if not prefixes:
            raise ValueError("at least one indexed-data prefix is required")
        self.arrays = [np.memmap(f"{prefix}.bin", mode="r", dtype=np.int32) for prefix in prefixes]
        if any(array.size <= sequence_len for array in self.arrays):
            raise ValueError("each indexed-data part must contain more than one sequence")
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.batch_index = start_batch
        self.start_batch = start_batch
        if repeat_window_batches is not None and repeat_window_batches < 1:
            raise ValueError("repeat_window_batches must be positive")
        self.repeat_window_batches = repeat_window_batches
        self.device = device
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future | None = None
        if self.device.type == "cuda":
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="token-prefetch")
            self._future = self._executor.submit(
                self._cpu_batch, self._source_batch_index(self.batch_index)
            )

    @staticmethod
    def _cyclic_slice(array: np.memmap, start: int, length: int) -> np.ndarray:
        start %= array.size
        if start + length <= array.size:
            return np.asarray(array[start : start + length])
        first = np.asarray(array[start:])
        remainder = length - first.size
        chunks = [first]
        while remainder >= array.size:
            chunks.append(np.asarray(array[:]))
            remainder -= array.size
        if remainder:
            chunks.append(np.asarray(array[:remainder]))
        return np.concatenate(chunks)

    def __iter__(self):
        return self

    def _source_batch_index(self, batch_index: int) -> int:
        return self._source_index(batch_index)

    def _source_index(self, batch_index: int) -> int:
        # Keep the extension point used by the strided simplicial pilots.
        if self.repeat_window_batches is None:
            return batch_index
        return self.start_batch + (
            (batch_index - self.start_batch) % self.repeat_window_batches
        )

    def _cpu_batch(self, batch_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        array = self.arrays[batch_index % len(self.arrays)]
        flat_tokens = self.batch_size * self.sequence_len
        start = (batch_index // len(self.arrays)) * flat_tokens
        window = self._cyclic_slice(array, start, flat_tokens + 1)
        # Copy because memmap slices are read-only and then pin for nonblocking H2D.
        tensor = torch.from_numpy(np.array(window, dtype=np.int64, copy=True))
        if self.device.type == "cuda":
            tensor = tensor.pin_memory()
        tokens = tensor[:-1].view(self.batch_size, self.sequence_len)
        labels = tensor[1:].view(self.batch_size, self.sequence_len)
        return tokens, labels

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._future is None:
            tokens, labels = self._cpu_batch(self._source_batch_index(self.batch_index))
        else:
            tokens, labels = self._future.result()
        self.batch_index += 1
        if self._executor is not None:
            self._future = self._executor.submit(
                self._cpu_batch, self._source_batch_index(self.batch_index)
            )
        return {
            "tokens": tokens.to(self.device, non_blocking=True),
            "labels": labels.to(self.device, non_blocking=True),
        }

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def partition_prefixes(
    prefixes: list[Path],
    rank: int,
    world_size: int,
    *,
    require_distinct: bool = True,
) -> list[Path]:
    if not prefixes:
        raise ValueError("at least one indexed-data prefix is required")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid rank/world size: {rank}/{world_size}")
    if require_distinct and len(prefixes) < world_size:
        raise ValueError(
            f"training requires at least one distinct indexed-data part per rank: "
            f"parts={len(prefixes)}, world_size={world_size}"
        )
    assigned = prefixes[rank::world_size]
    if assigned:
        return assigned
    return [prefixes[rank % len(prefixes)]]


# Historical name, identical ordering and state cursor.
DPRankTokenBatches = BinaryTokenBatches


def partition_prefixes_for_dp_rank(prefixes, data_parallel_rank, data_parallel_world_size, *, require_distinct=True):
    return partition_prefixes(prefixes, data_parallel_rank, data_parallel_world_size, require_distinct=require_distinct)
