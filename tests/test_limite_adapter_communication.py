from __future__ import annotations

import pytest
import torch

from archlab.automodel.limite_adapter_communication import (
    StaticGradientGroups,
    check_warmup_resume,
    communication_contract,
    fp32_mean_hook,
    synchronize_gradients,
)


def test_capacity_tracks_actual_trainable_gradient_bytes_by_dtype():
    train = torch.nn.Parameter(torch.empty(1024, dtype=torch.float32))
    frozen = torch.nn.Parameter(torch.empty(1024, dtype=torch.bfloat16), requires_grad=False)
    contract = communication_contract([train, frozen], 0.003)
    assert contract["gradient_bytes_by_dtype"] == {"torch.float32": 4096}
    assert not contract["fits_one_bucket_per_dtype"]
    assert communication_contract([train, frozen], 0.004)["fits_one_bucket_per_dtype"]
    assert not communication_contract([train])["explicit_bucket_capacity"]


def test_overlapping_fp32_schedule_preserves_native_gradient_storage():
    parameters = [
        torch.nn.Parameter(torch.empty(128, dtype=dtype))
        for dtype in (torch.float32, torch.bfloat16)
    ]
    contract = communication_contract(parameters, 128, schedule="bucketed-fp32")
    assert contract["reduction"] == "mean"
    assert contract["gradient_reduction_dtype"] == "float32"
    assert not contract["no_backward_overlap"]
    assert contract["gradient_as_bucket_view"]
    assert contract["gradient_bytes_by_dtype"] == {"torch.float32": 512, "torch.bfloat16": 256}


def test_fp32_hook_downcasts_after_mean_and_retains_bucket_storage(monkeypatch):
    import torch.distributed as dist

    native = torch.tensor([0.5, -1.25, 0.015625], dtype=torch.bfloat16)
    remote = torch.tensor([0.25, 0.0625, 0.00390625], dtype=torch.bfloat16)
    expected = ((native.float() + remote.float()) / 2).to(native.dtype)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)

    def collective(flat, *, group, async_op):
        assert flat.dtype == torch.float32
        assert async_op
        flat.add_(remote.float() / 2)
        completed = torch.futures.Future()
        completed.set_result([flat])
        return type("Work", (), {"get_future": lambda self: completed})()

    monkeypatch.setattr(dist, "all_reduce", collective)
    bucket = type("Bucket", (), {"buffer": lambda self: native})()
    actual = fp32_mean_hook(object(), bucket).wait()
    assert actual is native
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("capacity", [0, -1, float("nan"), float("inf")])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError, match="finite and positive"):
        communication_contract([], capacity)


def test_resume_scheduling_change_preserves_data_order_and_mean_gradient():
    prior = {"global_batch": 128, "context": 2048, "data_contract": "sealed"}
    check_warmup_resume(prior, global_batch=128, context=2048, data_contract="sealed")
    prior["communication"] = {"reduction": "mean", "bucket_cap_mib": 25}
    check_warmup_resume(prior, global_batch=128, context=2048, data_contract="sealed")
    for overrides in ({"global_batch": 64}, {"context": 1024}, {"data_contract": "other"}):
        selected = dict(global_batch=128, context=2048, data_contract="sealed")
        selected.update(overrides)
        with pytest.raises(ValueError, match="sealed warmup data order"):
            check_warmup_resume(prior, **selected)
    prior["communication"]["reduction"] = "sum"
    with pytest.raises(ValueError, match="gradient reduction"):
        check_warmup_resume(prior, global_batch=128, context=2048, data_contract="sealed")


def _parameters_with_gradients():
    gradients = [
        torch.tensor([[0.5, -1.25], [0.015625, 2.0]], dtype=torch.bfloat16),
        torch.tensor([[1.0, -2.0], [0.03125, 0.125]], dtype=torch.float32).t(),
        torch.tensor([0.25, -0.0625], dtype=torch.bfloat16),
    ]
    parameters = [torch.nn.Parameter(torch.zeros_like(gradient)) for gradient in gradients]
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient
    return parameters


@pytest.mark.parametrize("static", [False, True])
def test_deferred_mean_preserves_dtype_group_order_coordinates_and_native_storage(
    monkeypatch, static
):
    import torch.distributed as dist

    parameters = _parameters_with_gradients()
    gradients = [parameter.grad for parameter in parameters]
    storage = [(id(gradient), gradient.data_ptr(), gradient.stride()) for gradient in gradients]
    originals = [gradient.clone() for gradient in gradients]
    groups = [
        [0, 2],
        [1],
    ]  # First-seen dtype order, with native coordinate order within each dtype.
    remote = [
        torch.tensor([1.0, -0.75, 0.5, 0.125, -2.0, 0.0625]),
        torch.tensor([0.25, -1.5, 0.375, 0.75]),
    ]
    payloads = []
    monkeypatch.setattr(dist, "get_world_size", lambda: 3)
    if static:

        def forbidden_device_drain(*args, **kwargs):
            raise AssertionError("static synchronization must preserve current-stream ordering")

        monkeypatch.setattr(torch.cuda, "synchronize", forbidden_device_drain)

    def collective(flat):
        if flat.dtype == torch.int32:
            assert not static
            torch.testing.assert_close(flat, torch.ones(3, dtype=torch.int32))
            flat.mul_(3)
            return
        group = groups[len(payloads)]
        local = torch.cat([originals[index].reshape(-1) for index in group]).float()
        torch.testing.assert_close(flat, local / 3, rtol=0, atol=0)
        flat.add_(remote[len(payloads)] / 3)
        payloads.append(flat.clone())

    monkeypatch.setattr(dist, "all_reduce", collective)
    if static:
        cached = StaticGradientGroups(parameters)
        cached.synchronize()
        cached.validate()
    else:
        synchronize_gradients(parameters)
    assert len(payloads) == 2
    for group, payload in zip(groups, payloads, strict=True):
        parts = payload.split([originals[index].numel() for index in group])
        for index, part in zip(group, parts, strict=True):
            torch.testing.assert_close(
                parameters[index].grad,
                part.view_as(originals[index]).to(originals[index].dtype),
                rtol=0,
                atol=0,
            )
    assert [
        (id(gradient), gradient.data_ptr(), gradient.stride()) for gradient in gradients
    ] == storage
    assert not parameters[1].grad.is_contiguous()


def test_static_single_rank_validates_and_preserves_gradients_without_collectives(monkeypatch):
    import torch.distributed as dist

    parameters = _parameters_with_gradients()
    originals = [parameter.grad.clone() for parameter in parameters]

    def forbidden(*args, **kwargs):
        raise AssertionError("the qualified static path must not use a collective or device drain")

    monkeypatch.setattr(dist, "all_reduce", forbidden)
    monkeypatch.setattr(dist, "get_world_size", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    cached = StaticGradientGroups(parameters)
    cached.synchronize(world=1)
    for parameter, original in zip(parameters, originals, strict=True):
        torch.testing.assert_close(parameter.grad, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "change",
    ["unused", "object", "storage", "dtype", "shape", "stride", "parameter", "coverage", "frozen"],
)
def test_static_changed_gradient_contract_rejects_before_any_payload_mutation(monkeypatch, change):
    import torch.distributed as dist

    parameters = _parameters_with_gradients()
    cached = StaticGradientGroups(parameters)
    if change == "unused":
        parameters[-1].grad = None
    elif change == "object":
        # A replacement object aliases the same bytes, so pointer checks alone are insufficient.
        parameters[-1].grad = parameters[-1].grad.detach()
    elif change == "storage":
        parameters[-1].grad.data = parameters[-1].grad.clone()
    elif change == "dtype":
        parameters[-1].grad.data = parameters[-1].grad.float()
    elif change == "shape":
        parameters[-1].grad.resize_(1, 2)
    elif change == "stride":
        parameters[0].grad.transpose_(0, 1)
    elif change == "parameter":
        replacement = torch.nn.Parameter(parameters[-1].detach())
        replacement.grad = parameters[-1].grad
        cached.parameters = (*cached.parameters[:-1], replacement)
    elif change == "coverage":
        cached.parameters = cached.parameters[:-1]
    else:
        parameters[-1].requires_grad_(False)

    def forbidden(*args, **kwargs):
        raise AssertionError("contract validation must finish before issuing payload collectives")

    monkeypatch.setattr(dist, "all_reduce", forbidden)
    with pytest.raises(RuntimeError, match="unused|changed"):
        cached.synchronize(world=2)


def test_static_changed_parameter_storage_rejects_even_when_gradient_is_unchanged(monkeypatch):
    import torch.distributed as dist

    parameters = _parameters_with_gradients()
    cached = StaticGradientGroups(parameters)
    parameters[-1].data = parameters[-1].detach().clone()
    calls = []
    monkeypatch.setattr(dist, "all_reduce", lambda flat: calls.append(flat))
    with pytest.raises(RuntimeError, match="parameter identity or storage changed"):
        cached.synchronize(world=2)
    assert calls == []


def test_static_construction_requires_unique_all_used_dense_parameters():
    parameters = _parameters_with_gradients()
    with pytest.raises(ValueError, match="trainable parameters"):
        StaticGradientGroups([])
    with pytest.raises(ValueError, match="duplicates"):
        StaticGradientGroups([parameters[0], parameters[0]])
    parameters[-1].grad = None
    with pytest.raises(RuntimeError, match="unused"):
        StaticGradientGroups(parameters)
    sparse_parameter = torch.nn.Parameter(torch.zeros(2))
    sparse_parameter.grad = torch.sparse_coo_tensor([[0]], [1.0], (2,))
    with pytest.raises(ValueError, match="dense gradients"):
        StaticGradientGroups([sparse_parameter])


@pytest.mark.parametrize("world", [0, -1, 1.5])
def test_static_invalid_world_size_is_rejected(world):
    with pytest.raises(ValueError, match="positive world size"):
        StaticGradientGroups(_parameters_with_gradients()).synchronize(world=world)


def test_dynamic_path_retains_unused_gradient_agreement_and_collective(monkeypatch):
    import torch.distributed as dist

    parameters = _parameters_with_gradients()
    parameters[-1].grad = None
    calls = []
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def collective(flat):
        calls.append(flat.dtype)
        if flat.dtype == torch.int32:
            torch.testing.assert_close(flat, torch.tensor([1, 1, 0], dtype=torch.int32))
            flat.mul_(2)
        else:
            flat.mul_(2)  # Equal local/remote gradients preserve the original values.

    monkeypatch.setattr(dist, "all_reduce", collective)
    originals = [parameter.grad.clone() for parameter in parameters[:-1]]
    synchronize_gradients(parameters)
    assert calls == [torch.int32, torch.float32, torch.float32]
    assert parameters[-1].grad is None
    for parameter, original in zip(parameters[:-1], originals, strict=True):
        torch.testing.assert_close(parameter.grad, original, rtol=0, atol=0)


def test_dynamic_path_rejects_rank_presence_disagreement_before_payload(monkeypatch):
    import torch.distributed as dist

    parameters = _parameters_with_gradients()
    calls = []
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def collective(flat):
        calls.append(flat.dtype)
        flat.mul_(2)
        flat[-1] = 1  # Only one of the two ranks used the final parameter.

    monkeypatch.setattr(dist, "all_reduce", collective)
    with pytest.raises(RuntimeError, match="presence differs across ranks"):
        synchronize_gradients(parameters)
    assert calls == [torch.int32]


def test_static_contract_is_explicit_and_legacy_contract_values_remain_unchanged():
    parameters = _parameters_with_gradients()
    legacy = communication_contract(parameters, schedule="deferred")
    assert (
        communication_contract(parameters, schedule="deferred", static_gradient_sync=False)
        == legacy
    )
    assert not legacy["static_graph"]
    assert "gradient_presence" not in legacy
    static = communication_contract(parameters, schedule="deferred", static_gradient_sync=True)
    assert static["static_graph"]
    assert static["gradient_presence"] == "qualified_all_used_manual_graph"
    assert {
        key: value
        for key, value in static.items()
        if key not in ("static_graph", "gradient_presence")
    } == {key: value for key, value in legacy.items() if key != "static_graph"}
    for schedule in ("bucketed", "bucketed-fp32"):
        with pytest.raises(ValueError, match="requires the deferred schedule"):
            communication_contract(parameters, schedule=schedule, static_gradient_sync=True)
