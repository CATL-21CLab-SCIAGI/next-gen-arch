"""Numerical, causal, initialization, and optimizer contracts for the control."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from test_deepseek_v41_official_adapter import _official_model, _skeleton, _small_config
from test_deepseek_v41_official_recipe import mesh_receipts as mesh_receipts

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter
from archlab.architectures.deepseek_v41_normal_adapter import (
    V41NormalAttentionAdapter,
    normal_adapter_parameter_count,
)
from archlab.architectures.local_attention import (
    deterministic_local_attention,
    reference_local_attention,
)
from archlab.automodel.deepseek_v41_official_adapter import install_official_adapters
from archlab.automodel.deepseek_v41_official_recipe import OfficialV41Config, admit_mesh


def test_normal_control_shares_initial_weights_and_preserves_rng():
    config = _small_config()
    before = torch.get_rng_state().clone()
    baseline = V41SimplicialAdapter(config, seed=53, backend="reference")
    control = V41NormalAttentionAdapter(config, seed=53, backend="reference")
    assert torch.equal(before, torch.get_rng_state())
    reference = baseline.state_dict()
    for key, tensor in control.state_dict().items():
        mapped = key.replace("k.", "k2.").replace("v.", "v2.").replace("k_norm.", "k2_norm.")
        torch.testing.assert_close(tensor, reference[mapped], rtol=0, atol=0)
    assert sum(p.numel() for p in control.parameters()) == normal_adapter_parameter_count(config)
    assert normal_adapter_parameter_count(V41AdapterConfig()) * 8 == 146_843_712


def test_reference_matches_dense_masked_attention_and_is_causal():
    torch.manual_seed(19)
    q = torch.randn(2, 7, 4, 16, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 7, 2, 16, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    actual = reference_local_attention(q, k, v, 3)
    qh, kh, vh = (
        x.transpose(1, 2) for x in (q, k.repeat_interleave(2, 2), v.repeat_interleave(2, 2))
    )
    indices = torch.arange(7)
    mask = (indices[:, None] >= indices[None, :]) & (indices[:, None] - indices[None, :] < 3)
    expected = torch.nn.functional.scaled_dot_product_attention(
        qh, kh, vh, attn_mask=mask
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    grad = torch.randn_like(actual)
    a = torch.autograd.grad(actual, (q, k, v), grad, retain_graph=True)
    b = torch.autograd.grad(expected, (q, k, v), grad)
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)
    changed_k, changed_v = k.detach().clone(), v.detach().clone()
    changed_k[:, 5:] += 100
    changed_v[:, 5:] -= 100
    torch.testing.assert_close(
        reference_local_attention(q, changed_k, changed_v, 3)[:, :5], actual[:, :5], rtol=0, atol=0
    )
    changed_k, changed_v = k.detach().clone(), v.detach().clone()
    changed_k[:, :3] += 100
    changed_v[:, :3] -= 100
    torch.testing.assert_close(
        reference_local_attention(q, changed_k, changed_v, 3)[:, 5:], actual[:, 5:], rtol=0, atol=0
    )


def test_normal_insertion_identity_two_updates_and_exact_resume():
    torch.manual_seed(67)
    model = _official_model()
    tokens = torch.randint(0, 64, (1, 12))
    frozen = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        before = model(tokens).logits
    adapters = install_official_adapters(
        model, _small_config(), layer_indices=(0, 4), backend="reference", variant="normal"
    )
    torch.testing.assert_close(model(tokens).logits, before, rtol=0, atol=0)
    optim = torch.optim.AdamW(
        [p for a in adapters.values() for p in a.parameters()], lr=0.01, foreach=False
    )
    for step in range(2):
        optim.zero_grad(set_to_none=True)
        model(tokens, labels=tokens).loss.backward()
        for a in adapters.values():
            for name, p in a.named_parameters():
                assert p.grad is not None and bool(p.grad.isfinite().all()), name
                assert bool(p.grad.count_nonzero()) == (step > 0 or name == "output.weight"), name
        optim.step()
    for name, p in model.named_parameters():
        if name in frozen:
            assert p.grad is None
            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
    restored = copy.deepcopy(model)
    other = torch.optim.AdamW(
        [p for p in restored.parameters() if p.requires_grad], lr=0.01, foreach=False
    )
    other.load_state_dict(copy.deepcopy(optim.state_dict()))
    for candidate, optimizer in ((model, optim), (restored, other)):
        optimizer.zero_grad(set_to_none=True)
        candidate(tokens, labels=tokens).loss.backward()
        optimizer.step()
    torch.testing.assert_close(restored(tokens).logits, model(tokens).logits, rtol=0, atol=0)


def test_normal_production_optimizer_covers_each_parameter_once():
    from archlab.automodel.deepseek_v41_execution import adapter_optimizers

    model = _skeleton()
    adapters = install_official_adapters(
        model, backend="flash-attn-deterministic", variant="normal", device="meta"
    )
    optimizers = adapter_optimizers(model, adapters)
    groups = [group for optimizer in optimizers for group in optimizer.param_groups]
    params = [p for group in groups for p in group["params"]]
    assert len({id(p) for p in params}) == len(params)
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    assert sum(p.numel() for p in params) == 146_843_712
    assert len(groups[0]["params"]) == 16 and groups[0]["head_dim"] == 128
    assert len(groups[1]["params"]) == 24 and groups[1]["head_dim"] is None
    assert groups[2]["weight_decay"] == 0.1 and groups[3]["weight_decay"] == 0


def test_normal_recipe_only_changes_the_adapter_variant_and_core():
    root = Path(__file__).parents[1] / "recipes/experiments"
    baseline = yaml.safe_load((root / "deepseek_v41_simplicial_math_1b_official.yaml").read_text())
    normal = yaml.safe_load((root / "deepseek_v41_normal_math_1b_official.yaml").read_text())
    config = OfficialV41Config(**normal)
    assert normal.pop("adapter_variant") == "normal"
    assert normal["backend"].pop("adapter_core") == "flash-attn-deterministic-bf16-local512"
    baseline["backend"].pop("adapter_core")
    assert normal == baseline
    with pytest.raises(ValueError):
        replace(config, adapter_variant="simplicial")


def test_normal_admission_rejects_simplicial_mesh_receipts(mesh_receipts):
    path, hashes = mesh_receipts
    with pytest.raises(ValueError):
        admit_mesh(path, hashes, adapter_variant="normal")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the B300 container")
@pytest.mark.parametrize(
    "length,heads,kv_heads,dim,window", [(17, 4, 2, 32, 5), (513, 8, 2, 128, 512)]
)
def test_flash_attention_forward_backward_oracle_and_repeatability(
    length, heads, kv_heads, dim, window
):
    torch.manual_seed(47)
    q = torch.randn(1, length, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(
        1, length, kv_heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)
    inputs = (q, k, v)
    reference_inputs = tuple(x.detach().float().requires_grad_() for x in inputs)
    actual = deterministic_local_attention(*inputs, window)
    expected = reference_local_attention(*reference_inputs, window)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad.float())
    for measured, oracle in zip((actual, *actual_grads), (expected, *expected_grads), strict=True):
        relative = (measured.float() - oracle).norm() / oracle.norm()
        assert relative < 0.008, float(relative)
        torch.testing.assert_close(measured.float(), oracle, rtol=0.04, atol=0.035)
    replay = deterministic_local_attention(*inputs, window)
    replay_grads = torch.autograd.grad(replay, inputs, grad)
    for first, second in zip((actual, *actual_grads), (replay, *replay_grads), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
