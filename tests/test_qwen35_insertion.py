import pytest
import torch

transformers = pytest.importorskip("transformers")
qwen3_5 = pytest.importorskip("transformers.models.qwen3_5")

from archlab.architectures import qwen35_insertion as insertion  # noqa: E402


def tiny_model():
    config = transformers.AutoConfig.for_model(
        "qwen3_5_text", vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, layer_types=["linear_attention"] * 3 + ["full_attention"]
        + ["linear_attention"] * 3 + ["full_attention"], tie_word_embeddings=True)
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(config).float().eval()


def branch_config(mechanism, **overrides):
    return insertion.InsertionConfig(hidden_size=64, mechanism=mechanism, query_heads=4, kv_heads=2, head_dim=16,
                                     short_window=4, long_window=8, **overrides)


def test_sites_are_stratified_by_layer_type():
    types = ["linear_attention"] * 3 + ["full_attention"]
    sites = insertion.stratified_sites(types * 8, 0.25)
    kinds = [(types * 8)[s] for s in sites]
    assert len(sites) == 8 and kinds.count("full_attention") == 2 and kinds.count("linear_attention") == 6


@pytest.mark.parametrize("mechanism", ["simplicial", "local"])
def test_zero_initialized_branches_keep_the_parent_function_and_train_alone(mechanism):
    model = tiny_model()
    tokens = torch.randint(0, 97, (2, 12))
    with torch.no_grad():
        parent = model(tokens).logits
    handles = insertion.install(model, branch_config(mechanism), [1, 3, 6], backend="reference")
    with torch.no_grad():
        assert torch.equal(model(tokens).logits, parent)  # exact: zero output projection
    counts = insertion.freeze_backbone(model)
    assert counts["trainable"] == sum(p.numel() for p in model.inserted_branches.parameters())
    model.train()
    model(tokens).logits.float().square().mean().backward()
    for name, parameter in model.named_parameters():
        if name.startswith("inserted_branches."):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None
    assert model.inserted_branches["1"].output.weight.grad.abs().sum() > 0  # the zero projection still learns
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        assert torch.equal(model.eval()(tokens).logits, parent)


def test_nonzero_branches_change_the_function():
    model = tiny_model()
    tokens = torch.randint(0, 97, (1, 10))
    with torch.no_grad():
        parent = model(tokens).logits
        insertion.install(model, branch_config("simplicial", output_initialization="normal"), [2], backend="reference")
        assert not torch.allclose(model(tokens).logits, parent)
