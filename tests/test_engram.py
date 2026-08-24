import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.model_factory import build_model_from_config_kwargs


def test_engram_shared_initialization_is_bit_identical_to_baseline():
    common = dict(
        sequence_len=8,
        vocab_size=256,
        n_layer=2,
        n_head=1,
        n_kv_head=1,
        n_embd=128,
        window_pattern="L",
    )
    torch.manual_seed(42)
    with torch.device("meta"):
        baseline = GPT(GPTConfig(**common))
    baseline.to_empty(device="cpu")
    baseline.init_weights()

    torch.manual_seed(42)
    with torch.device("meta"):
        treatment = GPT(GPTConfig(
            **common,
            arch_family="engram",
            engram_layers=(0, 1),
            engram_ngram_orders=(2, 3),
            engram_num_heads=4,
            engram_dim=64,
            engram_vocab_multiplier=1,
        ))
    treatment.to_empty(device="cpu")
    treatment.init_weights()

    treatment_state = treatment.state_dict()
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, treatment_state[name], rtol=0, atol=0)
    assert torch.equal(treatment.smear_lambda, torch.zeros_like(treatment.smear_lambda))
    assert torch.equal(treatment.backout_lambda, torch.full_like(treatment.backout_lambda, 0.2))


def test_engram_factory_forward_backward_is_finite():
    config = {
        "sequence_len": 8,
        "vocab_size": 128,
        "n_layer": 2,
        "n_head": 1,
        "n_kv_head": 1,
        "n_embd": 128,
        "window_pattern": "L",
        "arch_family": "engram",
        "engram_layers": [0, 1],
        "engram_ngram_orders": [2, 3],
        "engram_num_heads": 4,
        "engram_dim": 64,
        "engram_vocab_multiplier": 1,
    }
    with torch.device("meta"):
        model, restored = build_model_from_config_kwargs(config)
    model.to_empty(device="cpu")
    model.init_weights()
    model.configure_engram_token_map(torch.arange(128), pad_id=0)
    tokens = torch.randint(0, 128, (2, 8))
    loss = model(tokens, targets=tokens.roll(-1, dims=1))
    loss.backward()
    assert torch.isfinite(loss)
    assert restored.engram_layers == (0, 1)
    assert all(parameter.grad is not None for parameter in model.engrams.parameters())
