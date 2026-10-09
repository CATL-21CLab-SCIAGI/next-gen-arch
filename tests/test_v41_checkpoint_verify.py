import torch

from archlab.megatron.miles_v41_checkpoint_verify import fingerprint


def test_fingerprint_covers_bf16_weights_and_optimizer_bytes():
    state = dict(weight=torch.arange(32, dtype=torch.bfloat16),
                 optimizer={"momentum": torch.zeros(32), "step": 1})
    before = fingerprint(state)
    assert before["bytes"] == 192
    state["optimizer"]["momentum"][17] = 1
    assert fingerprint(state) != before
    state["optimizer"]["momentum"][17] = 0
    assert fingerprint(state) == before
    state["weight"][31] += 1
    assert fingerprint(state) != before
