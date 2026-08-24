import sys
import types

import torch

if "pyarrow.parquet" not in sys.modules:
    pyarrow = types.ModuleType("pyarrow")
    pyarrow.__path__ = []
    sys.modules.setdefault("pyarrow", pyarrow)
    sys.modules["pyarrow.parquet"] = types.ModuleType("pyarrow.parquet")
if "nanochat.dataset" not in sys.modules:
    dataset = types.ModuleType("nanochat.dataset")
    dataset.list_parquet_files = lambda **_: []
    sys.modules["nanochat.dataset"] = dataset

import nanochat.dataloader as dataloader


class TinyTokenizer:
    def get_bos_token_id(self):
        return 1

    def encode(self, texts, prepend, num_threads):
        return [[prepend, *[2 + (ord(char) % 17) for char in text]] for text in texts]


def deterministic_documents(split, resume_state_dict, tokenizer_batch_size):
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
