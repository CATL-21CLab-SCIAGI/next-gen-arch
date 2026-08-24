import torch

from nanochat.gpt import GPT, GPTConfig, MHCConnection
from nanochat.model_factory import build_model_from_config_kwargs


def make_connection():
    config = GPTConfig(
        n_embd=16,
        n_head=2,
        n_kv_head=2,
        n_layer=2,
        mhc_num_streams=4,
        mhc_sinkhorn_iterations=20,
    )
    connection = MHCConnection(config)
    connection.init_weights()
    return connection


def test_mhc_mapping_constraints_and_gradients():
    connection = make_connection()
    streams = torch.randn(2, 3, 4, 16, dtype=torch.bfloat16, requires_grad=True)
    h_pre, h_post, h_res = connection.compute_mappings(streams)

    assert torch.all((h_pre > 0) & (h_pre < 1))
    assert torch.all((h_post > 0) & (h_post < 2))
    torch.testing.assert_close(h_res.sum(-1), torch.ones_like(h_res.sum(-1)), atol=2e-5, rtol=0)
    torch.testing.assert_close(h_res.sum(-2), torch.ones_like(h_res.sum(-2)), atol=2e-5, rtol=0)

    branch_input, post, residual = connection.prepare(streams)
    output = connection.combine(streams, branch_input, post, residual)
    output.float().square().mean().backward()
    assert streams.grad is not None
    assert connection.mapping_proj.weight.grad is not None
    assert connection.alpha.grad is not None


def test_mhc_zero_bias_initialization_matches_paper_parameterization():
    connection = make_connection()
    assert torch.equal(connection.bias, torch.zeros_like(connection.bias))
    torch.testing.assert_close(
        connection.alpha,
        torch.full_like(connection.alpha, 0.01),
    )


def test_mhc_shared_initialization_is_bit_identical_to_baseline():
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
        treatment = GPT(GPTConfig(**common, arch_family="mhc"))
    treatment.to_empty(device="cpu")
    treatment.init_weights()

    treatment_state = treatment.state_dict()
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, treatment_state[name], rtol=0, atol=0)
    assert torch.equal(treatment.smear_lambda, torch.zeros_like(treatment.smear_lambda))
    assert torch.equal(treatment.backout_lambda, torch.full_like(treatment.backout_lambda, 0.2))


def test_mhc_factory_forward_backward_is_finite():
    config = {
        "sequence_len": 8,
        "vocab_size": 128,
        "n_layer": 2,
        "n_head": 1,
        "n_kv_head": 1,
        "n_embd": 128,
        "window_pattern": "L",
        "arch_family": "mhc",
    }
    with torch.device("meta"):
        model, restored = build_model_from_config_kwargs(config)
    model.to_empty(device="cpu")
    model.init_weights()
    tokens = torch.randint(0, 128, (2, 8))
    loss = model(tokens, targets=tokens.roll(-1, dims=1))
    loss.backward()
    assert torch.isfinite(loss)
    assert restored.arch_family == "mhc"
    assert all(parameter.grad is not None for parameter in model.mhc_connections.parameters())
