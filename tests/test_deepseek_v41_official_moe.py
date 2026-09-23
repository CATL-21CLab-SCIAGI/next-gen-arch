"""Precision regression tests using the pinned official AutoModel MoE kernels."""

import importlib.util
import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from archlab.automodel.deepseek_v41_official_moe import install_official_fp32_moe


def test_variable_gather_backward_never_modifies_an_incoming_gradient(monkeypatch):
    from types import SimpleNamespace

    from archlab.automodel.deepseek_v41_official_moe import _NonMutatingVarlenGather

    incoming = torch.arange(35, dtype=torch.float32).reshape(5, 7)
    original = incoming.clone()
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value, **kwargs: value.mul_(2))
    context = SimpleNamespace(group=None, gathered_lens=[2, 3], rank=1)
    result, *_ = _NonMutatingVarlenGather.backward(context, incoming)
    assert torch.equal(incoming, original)
    assert torch.equal(result, original[2:] * 2)
    result.zero_()
    assert torch.equal(incoming, original)


def _official_moe(*, device="cpu", **overrides):
    if importlib.util.find_spec("nemo_automodel") is None:
        pytest.skip("requires pinned official AutoModel checkout on PYTHONPATH")
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.layers import MoE

    settings = dict(
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        dim=32,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=True,
        swiglu_limit=1.0,
        router_weights_fp32=True,
        force_e_score_correction_bias=True,
        dtype=torch.bfloat16,
    )
    settings.update(overrides)
    backend = BackendConfig(
        attn="eager",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch_mm",
        dispatcher="torch",
        gate_precision=None,
        enable_hf_state_dict_adapter=False,
    )
    model = MoE(MoEConfig(**settings), backend).to(device)
    model.requires_grad_(False)
    with torch.no_grad():
        model.init_weights(torch.device(device), init_std=0.2)
    return model


def _assert_unchanged(model, parameters, values, state):
    current = dict(model.named_parameters())
    assert current.keys() == parameters.keys()
    assert model.state_dict().keys() == state.keys()
    for name, parameter in parameters.items():
        assert current[name] is parameter, name
        assert not parameter.requires_grad and parameter.grad is None, name
        torch.testing.assert_close(parameter, values[name], rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def test_install_preserves_official_modules_parameters_buffers_and_checkpoint_keys():
    torch.manual_seed(712)
    model = nn.Sequential(_official_moe(), _official_moe())
    modules = dict(model.named_modules())
    parameters = dict(model.named_parameters())
    values = {name: value.detach().clone() for name, value in parameters.items()}
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()

    report = install_official_fp32_moe(model)

    assert report["modules"] == ["0", "1"]
    assert report["original_parameters_preserved"] is True
    assert dict(model.named_modules()) == modules
    assert torch.equal(torch.get_rng_state(), rng)
    _assert_unchanged(model, parameters, values, state)
    restored = nn.Sequential(_official_moe(), _official_moe())
    install_official_fp32_moe(restored)
    restored.load_state_dict(state, strict=True)
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def test_install_rejects_missing_moe_reinstallation_and_unfrozen_base():
    _official_moe()  # Skip consistently when the optional upstream is unavailable.
    with pytest.raises(ValueError, match="no official MoE"):
        install_official_fp32_moe(nn.Linear(32, 32))
    model = _official_moe()
    install_official_fp32_moe(model)
    with pytest.raises(ValueError, match="already installed"):
        install_official_fp32_moe(model)
    model = _official_moe()
    model.experts.down_projs.requires_grad_(True)
    with pytest.raises(ValueError, match="freeze"):
        install_official_fp32_moe(model)
    assert "forward" not in model.experts.__dict__


@pytest.mark.parametrize(
    "overrides",
    [
        {"apply_router_weight_after_down": True},
        {"expert_bias": True},
        {"expert_activation": "relu2"},
        {"swiglu_limit": 0.0},
        {"n_shared_experts": 0},
        {"shared_expert_gate": True},
        {"moe_latent_size": 16},
    ],
)
def test_unsupported_geometry_rejected_before_any_module_is_modified(overrides):
    model = nn.Sequential(_official_moe(), _official_moe(**overrides))
    with pytest.raises(ValueError, match="MoE geometry"):
        install_official_fp32_moe(model)
    assert all("forward" not in layer.experts.__dict__ for layer in model)
    assert all(not layer._forward_hooks for layer in model)


@pytest.mark.parametrize("field,value", [("dispatcher", "hybridep"), ("experts", "torch_mm_mxfp8")])
def test_backend_contract_rejected_before_installation(field, value):
    model = _official_moe()
    setattr(model.backend, field, value)
    with pytest.raises(ValueError, match="torch dispatcher and torch_mm"):
        install_official_fp32_moe(model)
    assert "forward" not in model.experts.__dict__


@pytest.fixture(params=["cpu", "cuda"])
def compute_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires frozen-container GPU")
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("requires container-owned torch._grouped_mm")
    return request.param


@pytest.mark.parametrize("keyword_input", [False, True])
def test_routed_fp32_sum_and_shared_add_match_captured_expert_oracle(
    compute_device,
    keyword_input,
    monkeypatch,
):
    torch.manual_seed(713)
    model = _official_moe(device=compute_device)
    install_official_fp32_moe(model)
    captured = {}
    original = torch._grouped_mm

    def capture_expert_outputs(*args, **kwargs):
        output = original(*args, **kwargs)
        captured["expert_outputs"] = output.detach().clone()
        offsets = args[2]
        captured["expert_counts"] = (offsets - F.pad(offsets[:-1], (1, 0))).tolist()
        return output

    monkeypatch.setattr(torch, "_grouped_mm", capture_expert_outputs)
    handles = [
        model.gate.register_forward_hook(
            lambda module, args, output: captured.update(indices=output[1].detach().clone())
        ),
        model.shared_experts.register_forward_hook(
            lambda module, args, output: captured.update(shared=output.detach().clone())
        ),
        model.experts.register_forward_hook(
            lambda module, args, output: captured.update(routed=output.detach().clone())
        ),
    ]
    x = torch.randn(2, 11, 32, device=compute_device, dtype=torch.bfloat16)
    padding = torch.zeros(2, 11, device=compute_device, dtype=torch.bool)
    padding[0, 3] = padding[1, 8] = True
    try:
        with torch.no_grad():
            output = model(x=x, padding_mask=padding) if keyword_input else model(x, padding)
    finally:
        for handle in handles:
            handle.remove()

    assert captured["expert_outputs"].dtype == torch.bfloat16
    assert captured["shared"].dtype == torch.bfloat16
    assert captured["routed"].dtype == torch.float32
    assert output.dtype == x.dtype and output.shape == x.shape
    # Reconstruct the sum from the actual BF16 down-projection outputs. This
    # isolates combination precision from differing grouped/individual GEMMs.
    expected_routed = torch.zeros_like(x.reshape(-1, 32), dtype=torch.float32)
    cursor = 0
    for expert in range(4):
        token_ids, _ = torch.where((captured["indices"] == expert) & (~padding).flatten()[:, None])
        assert token_ids.numel() == captured["expert_counts"][expert]
        for token in token_ids.tolist():
            expected_routed[token] += captured["expert_outputs"][cursor].float()
            cursor += 1
    assert cursor == captured["expert_outputs"].shape[0]
    torch.testing.assert_close(captured["routed"], expected_routed, rtol=0, atol=0)
    expected = (expected_routed + captured["shared"].float()).to(x.dtype).reshape_as(x)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    prematurely_rounded = (expected_routed.to(x.dtype) + captured["shared"]).reshape_as(x)
    assert not torch.equal(output, prematurely_rounded), (
        "fixture must expose the previous BF16 subtotal rounding"
    )
    torch.testing.assert_close(
        output[padding], captured["shared"].reshape_as(x)[padding], rtol=0, atol=0
    )


def _explicit_moe(model, x, padding):
    """Independent native equations decoded from the official parameter layout."""
    flat = x.reshape(-1, model.dim)
    config = model.experts.config
    scores = torch.sqrt(F.softplus(F.linear(flat, model.gate.weight).float()))
    indices = (
        (scores + model.gate.e_score_correction_bias)
        .topk(config.n_activated_experts, dim=-1)
        .indices
    )
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * config.route_scale
    routed = torch.zeros_like(flat, dtype=torch.float32)
    for expert in range(config.n_routed_experts):
        token_ids, slots = torch.where((indices == expert) & (~padding).flatten()[:, None])
        if not token_ids.numel():
            continue
        projection = model.experts.gate_and_up_projs[expert]
        first, second = projection.chunk(2, dim=-1)
        gate = F.linear(flat[token_ids], first.T.contiguous()).float()
        up = F.linear(flat[token_ids], second.T.contiguous()).float()
        gate = gate.clamp(max=config.swiglu_limit)
        up = up.clamp(min=-config.swiglu_limit, max=config.swiglu_limit)
        activated = (F.silu(gate) * up * weights[token_ids, slots, None]).to(x.dtype)
        expert_output = F.linear(activated, model.experts.down_projs[expert].T.contiguous())
        routed = routed.index_add(0, token_ids, expert_output.float())
    shared = model.shared_experts
    gate = F.linear(flat, shared.gate_proj.weight).float().clamp(max=config.swiglu_limit)
    up = (
        F.linear(flat, shared.up_proj.weight)
        .float()
        .clamp(
            min=-config.swiglu_limit,
            max=config.swiglu_limit,
        )
    )
    shared_output = F.linear((F.silu(gate) * up).to(x.dtype), shared.down_proj.weight)
    return (routed + shared_output.float()).to(x.dtype).reshape_as(x)


def test_input_gradient_matches_native_equations_with_frozen_weights(compute_device):
    torch.manual_seed(714)
    model = _official_moe(device=compute_device)
    parameters = dict(model.named_parameters())
    values = {name: value.detach().clone() for name, value in parameters.items()}
    state = {name: value.clone() for name, value in model.state_dict().items()}
    install_official_fp32_moe(model)
    x = torch.randn(2, 13, 32, device=compute_device, dtype=torch.bfloat16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    padding = torch.zeros(2, 13, device=compute_device, dtype=torch.bool)
    padding[0, 7] = True
    cotangent = torch.randn_like(x, dtype=torch.float32)
    actual = model(x, padding)
    expected = _explicit_moe(model, reference_x, padding)
    (actual.float() * cotangent).sum().backward()
    (expected.float() * cotangent).sum().backward()
    assert actual.isfinite().all() and x.grad.isfinite().all()
    assert x.grad.count_nonzero() and reference_x.grad.count_nonzero()
    # Individual and grouped GEMMs may round differently, especially their
    # BF16 gradient accumulation. Bound aggregate error relative to the oracle.
    for value, reference in ((actual, expected), (x.grad, reference_x.grad)):
        relative_l2 = (
            value.detach().float() - reference.detach().float()
        ).norm() / reference.detach().float().norm()
        assert float(relative_l2) < 0.01, float(relative_l2)
    _assert_unchanged(model, parameters, values, state)


def _uneven_ep_worker(rank, directory):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.set_num_threads(2)
    torch.manual_seed(715)
    model = _official_moe()
    reference = _official_moe()
    reference.load_state_dict(model.state_dict(), strict=True)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        for name in ("gate_and_up_projs", "down_projs"):
            parameter = getattr(model.experts, name)
            sharded = distribute_tensor(parameter.detach(), mesh, [Shard(0)])
            setattr(model.experts, name, nn.Parameter(sharded, requires_grad=False))
        parameters = dict(model.named_parameters())
        install_official_fp32_moe(model)
        assert all(
            dict(model.named_parameters())[name] is value for name, value in parameters.items()
        )
        captured = {}
        handle = model.experts.register_forward_hook(
            lambda module, args, output: captured.update(routed_dtype=str(output.dtype))
        )
        torch.manual_seed(716 + rank)
        x = torch.randn(1, 3 + rank * 2, 32, dtype=torch.bfloat16, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_(True)
        padding = torch.zeros(x.shape[:-1], dtype=torch.bool)
        padding[0, -1] = rank == 1
        cotangent = torch.randn_like(x, dtype=torch.float32)
        actual = model(x, padding)
        expected = _explicit_moe(reference, reference_x, padding)
        (actual.float() * cotangent).sum().backward()
        (expected.float() * cotangent).sum().backward()
        handle.remove()
        assert captured["routed_dtype"] == "torch.float32"
        assert actual.dtype == x.dtype and actual.shape == x.shape
        assert x.grad.isfinite().all() and x.grad.count_nonzero()
        assert all(parameter.grad is None for parameter in model.parameters())
        errors = {}
        for name, value, oracle in (
            ("output_relative_l2", actual, expected),
            ("input_gradient_relative_l2", x.grad, reference_x.grad),
        ):
            error = float(
                (value.detach().float() - oracle.detach().float()).norm()
                / oracle.detach().float().norm()
            )
            errors[name] = error
            assert error < 0.01, (rank, name, error)
        Path(directory, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "local_tokens": x.shape[1],
                    "passed": True,
                    **errors,
                }
            )
        )
    finally:
        dist.destroy_process_group()


def test_unequal_token_ep_gather_reduce_and_input_gradients_on_two_cpu_ranks(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    _official_moe()
    if not dist.is_gloo_available() or not hasattr(torch, "_grouped_mm"):
        pytest.skip("requires container-owned Gloo and torch._grouped_mm")
    if dist.is_initialized():
        pytest.skip("requires isolated process groups")
    context = mp.spawn(_uneven_ep_worker, args=(str(tmp_path),), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 120
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("two-rank CPU EP test exceeded 120 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
    receipts = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert [receipt["local_tokens"] for receipt in receipts] == [3, 5]
    assert all(receipt["passed"] for receipt in receipts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires frozen-container GPU FSDP2")
def test_install_after_one_rank_fsdp_preserves_state_and_input_gradients(tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor

    if dist.is_initialized():
        pytest.skip("requires an isolated one-rank process group")
    torch.cuda.set_device(0)
    model = _official_moe(device="cuda")
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    try:
        model = nn.Sequential(model)
        mesh = init_device_mesh("cuda", (1,))
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=None,
                cast_forward_inputs=False,
            ),
        )
        parameters = dict(model.named_parameters())
        assert all(isinstance(parameter, DTensor) for parameter in parameters.values())
        install_official_fp32_moe(model)
        assert all(
            dict(model.named_parameters())[name] is parameter
            for name, parameter in parameters.items()
        )
        assert all(not parameter.requires_grad for parameter in model.parameters())
        for _ in range(2):
            x = torch.randn(1, 11, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            output = model(x)
            assert output.dtype == x.dtype and output.isfinite().all()
            output.float().square().mean().backward()
            assert x.grad.isfinite().all() and x.grad.count_nonzero()
            assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        dist.destroy_process_group()
