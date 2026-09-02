from pathlib import Path

import numpy as np
import torch

from archlab.megatron.qwen38_train import (
    BinaryTokenBatches,
    _data_prefixes,
    _partition_prefixes,
)


def _part(root: Path, split: str, index: int, tokens: list[int]) -> Path:
    prefix = root / split / f"part-{index:05d}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(tokens, dtype=np.int32).tofile(f"{prefix}.bin")
    Path(f"{prefix}.idx").write_bytes(b"idx")
    Path(f"{prefix}.json").write_text("{}")
    return prefix


def test_binary_batches_shift_labels_and_wrap(tmp_path):
    prefix = _part(tmp_path, "train", 0, list(range(10)))
    batches = BinaryTokenBatches(
        [prefix],
        batch_size=1,
        sequence_len=4,
        start_batch=2,
        device=torch.device("cpu"),
    )

    first = next(batches)
    second = next(batches)
    assert first["tokens"].tolist() == [[8, 9, 0, 1]]
    assert first["labels"].tolist() == [[9, 0, 1, 2]]
    assert second["tokens"].tolist() == [[2, 3, 4, 5]]


def test_prefix_validation_and_rank_partition(tmp_path):
    prefixes = [_part(tmp_path, "train", i, list(range(8))) for i in range(4)]
    assert _data_prefixes(tmp_path, "train") == prefixes
    assert _partition_prefixes(prefixes, 1, 2) == [prefixes[1], prefixes[3]]
    assert _partition_prefixes(prefixes, 7, 8) == [prefixes[3]]
