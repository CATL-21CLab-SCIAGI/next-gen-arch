import sys
from types import SimpleNamespace

import torch

from archlab.rl.limite_generation import graph_generate


def test_compaction_retains_logical_samples_probabilities_and_generator_clock(monkeypatch):
    """Different EOS times must not reorder rows or consume a different RNG stream."""
    calls = []

    def logits(rows, step):
        values = torch.full((len(rows), 8), -torch.inf)
        for index, row in enumerate(rows.tolist()):
            values[index, 7 if step >= row * 2 else 1] = 0
        return values

    class Decoder:
        def __init__(self, model, source, token, capacity):
            self.rows = source.layers[0].keys[:, 0, 0, 0].long()
            self.cache = source

        def __call__(self, token, position):
            calls.append(len(self.rows))
            return logits(self.rows, position)

    monkeypatch.setitem(sys.modules, "archlab.architectures.limite_decode", SimpleNamespace(GraphDecoder=Decoder))

    class Model:
        config = SimpleNamespace(max_position_embeddings=20)

        def __call__(self, input_ids, **kwargs):
            rows = input_ids[:, 0]
            cache = SimpleNamespace(layers=[SimpleNamespace(keys=rows[:, None, None, None], values=rows[:, None, None, None])],
                                    get_seq_length=lambda: 1)
            return SimpleNamespace(logits=logits(rows, 0)[:, None], past_key_values=cache)

    ids = torch.arange(4)[:, None]
    config = SimpleNamespace(max_new_tokens=10, eos_token_id=[7])
    tokenizer = SimpleNamespace(pad_token_id=0)
    plain_rng, compact_rng = torch.Generator().manual_seed(123), torch.Generator().manual_seed(123)
    full = graph_generate(Model(), ids, config, tokenizer, lambda: False, generator=plain_rng)
    full_rows = sum(calls)
    calls.clear()
    compact = graph_generate(Model(), ids, config, tokenizer, lambda: False, generator=compact_rng, compact=True)
    assert torch.equal(full[0], compact[0])
    assert torch.equal(full[1], compact[1])
    assert full[2:4] == compact[2:4] == ([1, 3, 5, 7], ["eos"] * 4)
    assert torch.equal(plain_rng.get_state(), compact_rng.get_state())
    assert sum(calls) == 12 < full_rows == 24


def test_native_context_budget_is_prompt_adjusted_and_final_generated_token_is_retained(monkeypatch):
    calls = []
    class Decoder:
        def __init__(self, model, source, token, capacity):
            assert capacity == 20
        def __call__(self, token, position):
            calls.append(position)
            return torch.tensor([[-torch.inf, 0.]])
    monkeypatch.setitem(sys.modules, "archlab.architectures.limite_decode", SimpleNamespace(GraphDecoder=Decoder))
    class Model:
        config = SimpleNamespace(max_position_embeddings=20)
        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[[-torch.inf, 0.]]]), past_key_values=object())
    config = SimpleNamespace(max_new_tokens=20, archlab_budget_mode="native_context", eos_token_id=[0])
    output = graph_generate(Model(), torch.ones(1, 3, dtype=torch.long), config,
                            SimpleNamespace(pad_token_id=0), lambda: False)
    assert output[0].tolist() == [[1] * 17]
    assert output[2:4] == ([17], ["length"])
    assert calls == list(range(3, 19))
