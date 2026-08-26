import sys
import types

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
