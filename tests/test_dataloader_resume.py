import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

if "pyarrow.parquet" not in sys.modules:
    pyarrow = types.ModuleType("pyarrow")
    pyarrow.__path__ = []
    sys.modules.setdefault("pyarrow", pyarrow)
    sys.modules["pyarrow.parquet"] = types.ModuleType("pyarrow.parquet")
if "archlab.speedrun.dataset" not in sys.modules:
    dataset = types.ModuleType("archlab.speedrun.dataset")
    dataset.list_parquet_files = lambda **_: []
    sys.modules["archlab.speedrun.dataset"] = dataset

import archlab.speedrun.dataloader as dataloader


def _write_fineweb_shard(path: Path, tokens: list[int]) -> None:
    header = np.zeros(dataloader.FINEWEB_HEADER_INTS, dtype=np.int32)
    header[:3] = (dataloader.FINEWEB_MAGIC, dataloader.FINEWEB_VERSION, len(tokens))
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(np.asarray(tokens, dtype=np.uint16).tobytes())


class TinyTokenizer:
    def get_bos_token_id(self):
        return 1

    def encode(self, texts, prepend, num_threads):
        return [[prepend, *[2 + (ord(char) % 17) for char in text]] for text in texts]


def deterministic_documents(split, resume_state_dict, tokenizer_batch_size, **_kwargs):
    index = 0
    while True:
        texts = [f"doc-{index}-{offset}" for offset in range(tokenizer_batch_size)]
        yield texts, (index // 10, index, 1 + index // 100)
        index += 1


def test_exact_bestfit_resume_replays_the_prefetched_batch(monkeypatch):
    monkeypatch.setattr(dataloader, "_document_batches", deterministic_documents)
    kwargs = dict(
        tokenizer=TinyTokenizer(),
        B=2,
        T=8,
        split="train",
        tokenizer_threads=1,
        tokenizer_batch_size=4,
        device="cpu",
        buffer_size=8,
    )
    uninterrupted = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(**kwargs)
    batches = []
    for _ in range(7):
        inputs, targets, state = next(uninterrupted)
        batches.append((inputs.clone(), targets.clone(), dict(state)))

    resume_state = batches[4][2]
    assert resume_state["batch_index"] == 4
    resumed = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        **kwargs,
        resume_state_dict=resume_state,
    )
    resumed_inputs, resumed_targets, resumed_state = next(resumed)
    torch.testing.assert_close(resumed_inputs, batches[4][0], rtol=0, atol=0)
    torch.testing.assert_close(resumed_targets, batches[4][1], rtol=0, atol=0)
    assert resumed_state == resume_state

    next_inputs, next_targets, next_state = next(resumed)
    torch.testing.assert_close(next_inputs, batches[5][0], rtol=0, atol=0)
    torch.testing.assert_close(next_targets, batches[5][1], rtol=0, atol=0)
    assert next_state["batch_index"] == 5


def test_repacking_one_global_batch_preserves_historical_row_order(monkeypatch):
    monkeypatch.setattr(dataloader, "_document_batches", deterministic_documents)
    kwargs = dict(
        tokenizer=TinyTokenizer(),
        T=8,
        split="train",
        tokenizer_threads=1,
        tokenizer_batch_size=4,
        device="cpu",
        buffer_size=8,
        distributed=False,
    )
    historical = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        B=4, **kwargs
    )
    historical_inputs = []
    historical_targets = []
    for _ in range(3):
        inputs, targets, _state = next(historical)
        historical_inputs.append(inputs.clone())
        historical_targets.append(targets.clone())

    global_loader = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        B=12, **kwargs
    )
    global_inputs, global_targets, _state = next(global_loader)

    torch.testing.assert_close(global_inputs, torch.cat(historical_inputs), rtol=0, atol=0)
    torch.testing.assert_close(global_targets, torch.cat(historical_targets), rtol=0, atol=0)


def test_balanced_replicated_slice_keeps_exact_192_sequences_on_15_ranks():
    slices = [
        dataloader.balanced_replicated_batch_slice(192, rank, 15) for rank in range(15)
    ]

    assert [active for _start, active, _local in slices] == [13] * 12 + [12] * 3
    assert {local for _start, _active, local in slices} == {13}
    assert sum(active for _start, active, _local in slices) == 192
    assert [start for start, _active, _local in slices] == [
        0,
        13,
        26,
        39,
        52,
        65,
        78,
        91,
        104,
        117,
        130,
        143,
        156,
        168,
        180,
    ]


def test_fineweb_binary_loader_partitions_one_contiguous_global_batch(tmp_path):
    shard = tmp_path / "fineweb_train_000001.bin"
    _write_fineweb_shard(shard, list(range(100)))

    rank0 = dataloader.FineWebBinaryLoader(
        tmp_path, "train", 2, 4, rank=0, world_size=2, device="cpu"
    )
    rank1 = dataloader.FineWebBinaryLoader(
        tmp_path, "train", 2, 4, rank=1, world_size=2, device="cpu"
    )
    inputs0, labels0 = next(rank0)
    inputs1, labels1 = next(rank1)

    assert inputs0.flatten().tolist() == list(range(8))
    assert labels0.flatten().tolist() == list(range(1, 9))
    assert inputs1.flatten().tolist() == list(range(8, 16))
    assert labels1.flatten().tolist() == list(range(9, 17))
    assert next(rank0)[0].flatten().tolist() == list(range(16, 24))


def test_fineweb_shard_validation_rejects_truncated_payload(tmp_path):
    shard = tmp_path / "fineweb_val_000000.bin"
    _write_fineweb_shard(shard, [1, 2, 3])
    with shard.open("ab") as handle:
        handle.write(b"x")

    with pytest.raises(ValueError, match="length mismatch"):
        dataloader.inspect_fineweb_shard(shard)


def test_fineweb_inventory_accepts_a_validated_prefix_large_enough_for_run(tmp_path):
    _write_fineweb_shard(tmp_path / "fineweb_train_000001.bin", list(range(32)))
    _write_fineweb_shard(tmp_path / "fineweb_val_000000.bin", list(range(16)))

    summary = dataloader.inspect_fineweb_dataset(tmp_path, required_train_tokens=17)

    assert summary == {
        "visible_train_shards": 1,
        "validated_train_shards": 1,
        "validation_shards": 1,
        "validated_train_tokens": 32,
        "validation_tokens": 16,
        "complete_inventory": False,
    }


def test_fineweb_inventory_rejects_an_insufficient_prefix(tmp_path):
    _write_fineweb_shard(tmp_path / "fineweb_train_000001.bin", list(range(8)))
    _write_fineweb_shard(tmp_path / "fineweb_val_000000.bin", list(range(8)))

    with pytest.raises(ValueError, match="fewer than the required"):
        dataloader.inspect_fineweb_dataset(tmp_path, required_train_tokens=9)
