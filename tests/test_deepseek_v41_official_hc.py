"""Native mHC forward precision and frozen-input backward integration checks."""

import copy
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import archlab.automodel.deepseek_v41_official_hc as native_hc
from archlab.architectures.deepseek_v41_math import hc_split_sinkhorn


def _official_hc(device="cpu"):
    if importlib.util.find_spec("nemo_automodel") is None:
        pytest.skip("requires pinned official AutoModel checkout on PYTHONPATH")
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
    from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41HyperConnection

    config = DeepseekV41TextConfig(hidden_size=32, hc_mult=4, hc_sinkhorn_iters=20)
    model = DeepseekV41HyperConnection(config, sinkhorn_backend="torch").to(device)
    with torch.no_grad():
        model.fn.normal_(std=0.17)
        model.scale.copy_(torch.tensor([0.71, -0.83, 1.13], device=device))
        model.base.normal_(std=0.31)
    model.requires_grad_(False)
    return model


@pytest.fixture
def cpu_native(monkeypatch, tmp_path):
    # The actual native kernel is CUDA-only. Keep real official module structure
    # and substitute the independent math oracle only at the loader boundary.
    _official_hc()
    monkeypatch.setattr(native_hc, "_native_hc_from_assets", lambda assets: hc_split_sinkhorn)
    return tmp_path


@pytest.fixture
def cuda_assets():
    if not torch.cuda.is_available():
        pytest.skip("requires frozen-container GPU and pinned native reference")
    _official_hc()
    default = (
        Path(__file__).resolve().parents[1] / "results/deepseek-v41-launch-20260913T165222Z/assets"
    )
    assets = Path(os.environ.get("ARCHLAB_DEEPSEEK_V41_ASSETS", str(default)))
    if not (assets / "inference/kernel.py").is_file():
        pytest.skip("set ARCHLAB_DEEPSEEK_V41_ASSETS to the pinned native reference")
    return assets


def _coefficients(output):
    from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41Mix

    assert type(output) is DeepseekV41Mix
    values = output.pre, output.post, output.comb
    assert all(value.dtype == torch.float32 and value.isfinite().all() for value in values)
    return values


def _oracle(model, hidden, kernel=hc_split_sinkhorn):
    flat = hidden.flatten(2).float()
    mixes = F.linear(flat, model.fn) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + model.norm_eps
    )
    return kernel(mixes, model.scale, model.base, model.streams, model.iterations, model.eps)


def _assert_frozen_state(model, parameters, state):
    current = dict(model.named_parameters())
    assert current.keys() == parameters.keys()
    assert model.state_dict().keys() == state.keys()
    for name, parameter in parameters.items():
        assert current[name] is parameter, name
        assert parameter.dtype == torch.float32 and not parameter.requires_grad, name
        assert parameter.grad is None, name
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def test_install_preserves_modules_parameters_state_dtype_and_rng(cpu_native):
    torch.manual_seed(831)
    model = nn.ModuleList([_official_hc(), _official_hc()])
    model.register_buffer("retained_buffer", torch.tensor([3.25]))
    modules = dict(model.named_modules())
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    report = native_hc.install_official_native_hc(model, cpu_native)
    assert report["modules"] == ["0", "1"]
    assert report["original_parameters_preserved"] is True
    assert dict(model.named_modules()) == modules
    assert dict(model.named_buffers()) == buffers
    assert torch.equal(torch.get_rng_state(), rng)
    _assert_frozen_state(model, parameters, state)
    for connection in model:
        assert connection.forward.__self__ is connection
    restored = nn.ModuleList([_official_hc(), _official_hc()])
    restored.register_buffer("retained_buffer", torch.tensor([-7.0]))
    native_hc.install_official_native_hc(restored, cpu_native)
    restored.load_state_dict(state, strict=True)
    hidden = torch.randn(2, 7, 4, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        for left, right in zip(model, restored, strict=True):
            for actual, expected in zip(
                _coefficients(left(hidden)), _coefficients(right(hidden)), strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_install_rejects_missing_modules_and_reinstallation(cpu_native):
    with pytest.raises(ValueError, match="[Hh]yper[Cc]onnection|[Hh][Cc]|mHC"):
        native_hc.install_official_native_hc(nn.Linear(32, 32), cpu_native)
    model = _official_hc()
    native_hc.install_official_native_hc(model, cpu_native)
    with pytest.raises(ValueError, match="already|adapted"):
        native_hc.install_official_native_hc(model, cpu_native)


@pytest.mark.parametrize("parameter_name", ["fn", "scale", "base"])
@pytest.mark.parametrize("invalid_kind", ["trainable", "bfloat16"])
def test_invalid_frozen_fp32_contract_rejected_before_any_forward_changes(
    cpu_native,
    parameter_name,
    invalid_kind,
):
    model = nn.ModuleList([_official_hc(), _official_hc()])
    parameter = getattr(model[1], parameter_name)
    if invalid_kind == "trainable":
        parameter.requires_grad_(True)
    else:
        setattr(
            model[1],
            parameter_name,
            nn.Parameter(parameter.to(torch.bfloat16), requires_grad=False),
        )
    with pytest.raises((ValueError, TypeError), match="frozen|freeze|FP32|float32"):
        native_hc.install_official_native_hc(model, cpu_native)
    assert all("forward" not in connection.__dict__ for connection in model)


@pytest.mark.parametrize("field,value", [("streams", 2), ("iterations", 19), ("eps", 1e-5)])
def test_unsupported_geometry_is_rejected_before_installation(cpu_native, field, value):
    model = nn.ModuleList([_official_hc(), _official_hc()])
    setattr(model[1], field, value)
    with pytest.raises(ValueError, match="geometry"):
        native_hc.install_official_native_hc(model, cpu_native)
    assert all("forward" not in connection.__dict__ for connection in model)


def test_native_loader_rechecks_source_hashes_and_caches_canonical_fixture_path(
    monkeypatch, tmp_path
):
    assets = tmp_path / "assets"
    inference = assets / "inference"
    inference.mkdir(parents=True)
    digests = {}
    for name in ("kernel.py", "model.py"):
        contents = f"# pinned {name}\n".encode()
        (inference / name).write_bytes(contents)
        digests[name] = hashlib.sha256(contents).hexdigest()
    fixture = tmp_path / "prefix/inference"
    fixture.mkdir(parents=True)
    for name in digests:
        (fixture / name).symlink_to(inference / name)
    loads = []

    def verified_loader(path, *, module_name):
        loads.append((path, module_name))
        return SimpleNamespace(hc_split_sinkhorn=hc_split_sinkhorn)

    monkeypatch.setattr(native_hc, "REFERENCE_DIGESTS", digests)
    monkeypatch.setattr(native_hc, "_NATIVE_HC_BY_SOURCE", {})
    monkeypatch.setattr(native_hc, "load_native_reference", verified_loader)
    path_before = sys.path[:]
    dtype_before, device_before = torch.get_default_dtype(), torch.get_default_device()
    assert native_hc._native_hc_from_assets(fixture.parent) is hc_split_sinkhorn
    assert native_hc._native_hc_from_assets(assets) is hc_split_sinkhorn
    assert len(loads) == 1 and loads[0][0] == assets
    assert sys.path == path_before
    assert torch.get_default_dtype() == dtype_before and torch.get_default_device() == device_before
    (inference / "kernel.py").write_text("# changed after first cached import\n")
    with pytest.raises(ValueError, match="unreviewed"):
        native_hc._native_hc_from_assets(assets)
    assert len(loads) == 1


@pytest.mark.parametrize("used_outputs", [(0, 1, 2), (0,), (1,), (2,)])
def test_frozen_hidden_gradient_and_unused_coefficient_outputs(cpu_native, used_outputs):
    torch.manual_seed(832)
    model = _official_hc()
    parameters = dict(model.named_parameters())
    state = {name: value.clone() for name, value in model.state_dict().items()}
    native_hc.install_official_native_hc(model, cpu_native)
    hidden = torch.randn(2, 7, 4, 32, dtype=torch.bfloat16, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    actual = _coefficients(model(hidden))
    expected = _oracle(model, reference_hidden)
    cotangents = [torch.randn_like(value) for value in expected]
    sum((actual[i] * cotangents[i]).sum() for i in used_outputs).backward()
    sum((expected[i] * cotangents[i]).sum() for i in used_outputs).backward()
    assert hidden.grad.isfinite().all() and hidden.grad.count_nonzero()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    _assert_frozen_state(model, parameters, state)


def _checkpoint_comparison(model):
    hidden = torch.randn(
        2, 7, 4, 32, device=model.fn.device, dtype=torch.bfloat16, requires_grad=True
    )
    replay_hidden = hidden.detach().clone().requires_grad_(True)
    direct = _coefficients(model(hidden))
    replay = checkpoint(
        lambda value: _coefficients(model(value)), replay_hidden, use_reentrant=False
    )
    cotangents = [torch.randn_like(value) for value in direct]
    sum((value * weight).sum() for value, weight in zip(direct, cotangents, strict=True)).backward()
    sum((value * weight).sum() for value, weight in zip(replay, cotangents, strict=True)).backward()
    for value, expected in zip(replay, direct, strict=True):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    torch.testing.assert_close(replay_hidden.grad, hidden.grad, rtol=0, atol=0)
    assert hidden.grad.isfinite().all() and hidden.grad.count_nonzero()
    assert all(parameter.grad is None for parameter in model.parameters())


def test_nonreentrant_checkpoint_preserves_forward_and_frozen_input_gradients(cpu_native):
    torch.manual_seed(833)
    model = _official_hc()
    native_hc.install_official_native_hc(model, cpu_native)
    _checkpoint_comparison(model)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_cuda_all_coefficients_exactly_match_verified_native_kernel(cuda_assets, requires_grad):
    torch.manual_seed(834)
    model = _official_hc("cuda")
    kernel = native_hc._native_hc_from_assets(cuda_assets)
    native_hc.install_official_native_hc(model, cuda_assets)
    hidden = torch.randn(
        2, 17, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=requires_grad
    )
    with torch.set_grad_enabled(requires_grad):
        actual = _coefficients(model(hidden))
    with torch.no_grad():
        expected = _oracle(model, hidden, kernel)
    for value, reference in zip(actual, expected, strict=True):
        assert value.requires_grad == requires_grad
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_cuda_native_input_gradient_matches_math_and_official_backward(cuda_assets):
    torch.manual_seed(835)
    model = _official_hc("cuda")
    official = copy.deepcopy(model)
    native_hc.install_official_native_hc(model, cuda_assets)
    hidden = torch.randn(2, 17, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    math_hidden = hidden.detach().clone().requires_grad_(True)
    official_hidden = hidden.detach().clone().requires_grad_(True)
    actual = _coefficients(model(hidden))
    expected = _oracle(model, math_hidden)
    official_expected = _coefficients(official(official_hidden))
    cotangents = [torch.randn_like(value) for value in expected]
    for outputs in (actual, expected, official_expected):
        sum(
            (value * weight).sum() for value, weight in zip(outputs, cotangents, strict=True)
        ).backward()
    assert hidden.grad.isfinite().all() and hidden.grad.count_nonzero()
    # The native forward retains kernel rounding; the derivative is the reviewed
    # FP32 equation. An official derivative can differ at BF16 rounding edges.
    torch.testing.assert_close(hidden.grad, math_hidden.grad, rtol=0, atol=0)
    relative_l2 = (
        hidden.grad.float() - official_hidden.grad.float()
    ).norm() / official_hidden.grad.float().norm()
    assert float(relative_l2) < 0.01, float(relative_l2)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cuda_native_nonreentrant_checkpoint(cuda_assets):
    torch.manual_seed(836)
    model = _official_hc("cuda")
    native_hc.install_official_native_hc(model, cuda_assets)
    _checkpoint_comparison(model)


def test_cuda_install_after_fsdp_preserves_saved_scale_base_through_backward(cuda_assets, tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor

    if dist.is_initialized():
        pytest.skip("requires an isolated one-rank process group")
    torch.manual_seed(837)
    torch.cuda.set_device(0)
    model = _official_hc("cuda")
    reference = copy.deepcopy(model)
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cuda", (1,))
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                output_dtype=None,
                cast_forward_inputs=False,
            ),
        )
        # Make the HC a nested FSDP unit so its all-gather storage is released
        # after forward. Backward must safely retrieve saved scale/base values.
        root = nn.Sequential(model)
        fully_shard(root, mesh=mesh, reshard_after_forward=False)
        parameters = dict(root.named_parameters())
        assert all(isinstance(value, DTensor) for value in parameters.values())
        state = {name: value.to_local().clone() for name, value in root.state_dict().items()}
        native_hc.install_official_native_hc(root, cuda_assets)
        for _ in range(2):
            hidden = torch.randn(
                2, 7, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            reference_hidden = hidden.detach().clone().requires_grad_(True)
            actual = _coefficients(root(hidden))
            expected = _oracle(reference, reference_hidden)
            cotangents = [torch.randn_like(value) for value in expected]
            sum(
                (value * weight).sum() for value, weight in zip(actual, cotangents, strict=True)
            ).backward()
            sum(
                (value * weight).sum() for value, weight in zip(expected, cotangents, strict=True)
            ).backward()
            torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
            assert hidden.grad.isfinite().all() and hidden.grad.count_nonzero()
            assert all(parameter.grad is None for parameter in root.parameters())
        current = dict(root.named_parameters())
        assert all(current[name] is value for name, value in parameters.items())
        for name, value in root.state_dict().items():
            torch.testing.assert_close(value.to_local(), state[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
