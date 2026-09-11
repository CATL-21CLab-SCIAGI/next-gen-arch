"""Deterministic FineWeb binary data and fixed-window replay shared by backends."""

from pathlib import Path

import numpy as np
import torch

from archlab.distributed import get_dist_info

FINEWEB_HEADER_INTS = 256
FINEWEB_HEADER_BYTES = FINEWEB_HEADER_INTS * 4
FINEWEB_MAGIC = 20_240_520
FINEWEB_VERSION = 1


def inspect_fineweb_shard(path: Path) -> int:
    """Validate one modded-nanogpt/llm.c token shard and return its token count."""
    with path.open("rb") as handle:
        raw_header = handle.read(FINEWEB_HEADER_BYTES)
    if len(raw_header) != FINEWEB_HEADER_BYTES:
        raise ValueError(f"truncated FineWeb header: {path}")
    header = np.frombuffer(raw_header, dtype=np.int32)
    if int(header[0]) != FINEWEB_MAGIC:
        raise ValueError(f"FineWeb magic mismatch in {path}")
    if int(header[1]) != FINEWEB_VERSION:
        raise ValueError(f"unsupported FineWeb shard version in {path}: {int(header[1])}")
    token_count = int(header[2])
    expected_bytes = FINEWEB_HEADER_BYTES + 2 * token_count
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"FineWeb shard length mismatch in {path}: {actual_bytes} != {expected_bytes}"
        )
    return token_count


def inspect_fineweb_dataset(
    root: str | Path,
    *,
    expected_train_shards: int = 103,
    required_train_tokens: int | None = None,
) -> dict[str, int | bool]:
    """Validate the FineWeb prefix required by a run and report visible inventory."""
    root = Path(root)
    train = sorted(root.glob("fineweb_train_*.bin"))
    validation = sorted(root.glob("fineweb_val_*.bin"))
    expected_validation = [root / "fineweb_val_000000.bin"]
    if not train:
        raise ValueError("FineWeb has no visible train shards")
    if validation != expected_validation:
        raise ValueError("FineWeb requires exactly fineweb_val_000000.bin")
    if required_train_tokens is None and len(train) != expected_train_shards:
        raise ValueError(
            f"complete FineWeb inventory requires {expected_train_shards} train shards; "
            f"found {len(train)}"
        )
    required_train_tokens = required_train_tokens or 0
    train_tokens = 0
    validated_train_shards = 0
    paths_to_validate = (
        [root / f"fineweb_train_{index:06d}.bin" for index in range(1, expected_train_shards + 1)]
        if required_train_tokens == 0
        else train
    )
    for expected_index, path in enumerate(paths_to_validate, start=1):
        if path.name != f"fineweb_train_{expected_index:06d}.bin" or not path.is_file():
            raise ValueError(f"FineWeb is missing required train shard {expected_index:06d}")
        train_tokens += inspect_fineweb_shard(path)
        validated_train_shards += 1
        if train_tokens >= required_train_tokens and required_train_tokens > 0:
            break
    if train_tokens < required_train_tokens:
        raise ValueError(
            f"FineWeb prefix has {train_tokens} validated tokens, fewer than the "
            f"required {required_train_tokens}"
        )
    validation_tokens = sum(inspect_fineweb_shard(path) for path in validation)
    return {
        "visible_train_shards": len(train),
        "validated_train_shards": validated_train_shards,
        "validation_shards": len(validation),
        "validated_train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "complete_inventory": len(train) == expected_train_shards,
    }


class FineWebBinaryLoader:
    """Deterministic rank slices over public GPT-2 FineWeb binary shards.

    All ranks advance one shared logical cursor.  A global batch is one
    contiguous token range, split into equal rank-local ranges; the trailing
    target token overlaps the next range exactly as in modded-nanogpt.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        local_batch_size: int,
        sequence_length: int,
        *,
        rank: int,
        world_size: int,
        device: str | torch.device = "cuda",
        start_batch_index: int = 0,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if local_batch_size <= 0 or sequence_length <= 0:
            raise ValueError("batch size and sequence length must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be inside a positive world size")
        if start_batch_index < 0:
            raise ValueError("start batch index must be non-negative")
        self.files = sorted(Path(root).glob(f"fineweb_{split}_*.bin"))
        if not self.files:
            raise FileNotFoundError(f"no FineWeb {split} shards under {Path(root)}")
        self.local_batch_size = local_batch_size
        self.sequence_length = sequence_length
        self.local_tokens = local_batch_size * sequence_length
        self.global_tokens = self.local_tokens * world_size
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.shard_index = 0
        self.global_position = 0
        self.tokens: np.memmap | None = None
        self.token_count = 0
        self.batch_index = 0
        self._open_shard()
        if not any(inspect_fineweb_shard(path) > self.global_tokens for path in self.files):
            raise ValueError("no FineWeb shard is large enough for one distributed batch")
        self.seek_batch(start_batch_index)

    def _open_shard(self) -> None:
        path = self.files[self.shard_index]
        self.token_count = inspect_fineweb_shard(path)
        self.tokens = np.memmap(
            path,
            dtype=np.uint16,
            mode="r",
            offset=FINEWEB_HEADER_BYTES,
            shape=(self.token_count,),
        )
        self.global_position = 0

    def _advance_shard(self) -> None:
        self.shard_index = (self.shard_index + 1) % len(self.files)
        self._open_shard()

    def seek_batch(self, batch_index: int) -> None:
        """Seek to an exact distributed-microbatch cursor without reading payloads."""

        if batch_index < 0:
            raise ValueError("batch index must be non-negative")
        self.shard_index = 0
        self._open_shard()
        self.batch_index = 0
        remaining = batch_index
        while remaining:
            available = max(0, (self.token_count - self.global_position - 1) // self.global_tokens)
            if remaining < available:
                self.global_position += remaining * self.global_tokens
                self.batch_index += remaining
                return
            remaining -= available
            self.batch_index += available
            self._advance_shard()

    def state_dict(self) -> dict[str, int]:
        return {
            "shard_index": self.shard_index,
            "global_position": self.global_position,
            "batch_index": self.batch_index,
        }

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        while self.global_position + self.global_tokens + 1 > self.token_count:
            self._advance_shard()
        start = self.global_position + self.rank * self.local_tokens
        stop = start + self.local_tokens + 1
        assert self.tokens is not None
        # np.array makes a writable, aligned owner before the asynchronous H2D copy.
        buffer = torch.from_numpy(np.array(self.tokens[start:stop], dtype=np.int64, copy=True))
        inputs = buffer[:-1].view(self.local_batch_size, self.sequence_length)
        labels = buffer[1:].view(self.local_batch_size, self.sequence_length)
        self.global_position += self.global_tokens
        self.batch_index += 1
        if self.device.type == "cuda":
            inputs = inputs.pin_memory().to(self.device, non_blocking=True)
            labels = labels.pin_memory().to(self.device, non_blocking=True)
        else:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
        return inputs, labels


def fineweb_distributed_data_loader(
    root: str | Path,
    split: str,
    local_batch_size: int,
    sequence_length: int,
    *,
    device: str | torch.device = "cuda",
    start_batch_index: int = 0,
):
    """Yield rank-local FineWeb batches using the initialized torchrun topology."""
    _ddp, rank, _local_rank, world_size = get_dist_info()
    yield from FineWebBinaryLoader(
        root,
        split,
        local_batch_size,
        sequence_length,
        rank=rank,
        world_size=world_size,
        device=device,
        start_batch_index=start_batch_index,
    )


def fixed_fineweb_validation_loader(
    root: str | Path,
    local_batch_size: int,
    sequence_length: int,
    *,
    window_batches: int,
    device: str | torch.device = "cuda",
):
    """Replay the identical validation token window at every evaluation."""

    if window_batches < 1:
        raise ValueError("validation window must contain at least one batch")
    while True:
        source = fineweb_distributed_data_loader(
            root,
            "val",
            local_batch_size,
            sequence_length,
            device=device,
        )
        for _ in range(window_batches):
            yield next(source)
