"""Explicit DDP scheduling contracts without changing warmup model arithmetic."""

from __future__ import annotations

import math


def communication_contract(
    parameters, bucket_cap_mb=None, *, schedule="bucketed", static_gradient_sync=False
):
    if schedule not in ("bucketed", "bucketed-fp32", "deferred"):
        raise ValueError("invalid warmup communication schedule")
    if static_gradient_sync and schedule != "deferred":
        raise ValueError("static gradient synchronization requires the deferred schedule")
    if bucket_cap_mb is not None and (not math.isfinite(bucket_cap_mb) or bucket_cap_mb <= 0):
        raise ValueError("DDP bucket capacity must be finite and positive")
    gradient_bytes_by_dtype = {}
    for parameter in parameters:
        if parameter.requires_grad:
            dtype = str(parameter.dtype)
            gradient_bytes_by_dtype[dtype] = gradient_bytes_by_dtype.get(dtype, 0) + (
                parameter.numel() * parameter.element_size()
            )
    capacity = 25.0 if bucket_cap_mb is None else float(bucket_cap_mb)
    result = {
        "backend": "torch.nn.parallel.DistributedDataParallel",
        "reduction": "mean",
        "static_graph": True,
        "gradient_as_bucket_view": True,
        "bucket_cap_mib": capacity,
        "explicit_bucket_capacity": bucket_cap_mb is not None,
        "gradient_bytes_by_dtype": gradient_bytes_by_dtype,
        "fits_one_bucket_per_dtype": all(
            size <= capacity * 2**20 for size in gradient_bytes_by_dtype.values()
        ),
        "schedule": schedule,
        "gradient_reduction_dtype": "native",
    }
    if schedule == "deferred":
        result.update(
            backend="torch.distributed.all_reduce",
            bucket_cap_mib=None,
            explicit_bucket_capacity=False,
            fits_one_bucket_per_dtype=None,
            no_backward_overlap=True,
            gradient_reduction_dtype="float32",
            static_graph=False,
            gradient_as_bucket_view=False,
        )
    elif schedule == "bucketed-fp32":
        result.update(gradient_reduction_dtype="float32", no_backward_overlap=False)
    if static_gradient_sync:
        result.update(
            static_graph=True,
            gradient_presence="qualified_all_used_manual_graph",
        )
    return result


def fp32_mean_hook(process_group, bucket):
    """Public DDP hook: overlap FP32 mean reduction with native backward.

    BF16 model/gradient storage is retained. Like the deferred path, division
    precedes the FP32 collective and the result is cast only after reduction.
    Return the original bucket so DDP's gradient views remain intact.
    """
    import torch.distributed as dist

    group = process_group if process_group is not None else dist.group.WORLD
    native = bucket.buffer()
    flat = native.float()
    flat.div_(dist.get_world_size(group))
    future = dist.all_reduce(flat, group=group, async_op=True).get_future()

    def restore(completed):
        native.copy_(completed.value()[0])
        return native

    return future.then(restore)


def _synchronize_gradient_groups(groups, world):
    """Shared native cat -> FP32 divide -> collective -> native copy ordering."""
    import torch
    import torch.distributed as dist

    for gradients in groups:
        flat = torch.cat([gradient.reshape(-1) for gradient in gradients]).float()
        # Newly trainable BF16 parameters retain BF16 gradient storage, while
        # distributed accumulation and division use FP32 before casting back.
        flat.div_(world)
        dist.all_reduce(flat)
        offsets = flat.split([gradient.numel() for gradient in gradients])
        torch._foreach_copy_(
            gradients, [part.view_as(grad) for part, grad in zip(offsets, gradients, strict=True)]
        )


def _tensor_storage(tensor):
    return (
        id(tensor),
        tensor.data_ptr(),
        tensor.dtype,
        tensor.device,
        tensor.shape,
        tensor.stride(),
        tensor.layout,
    )


class StaticGradientGroups:
    """Cache the qualified manual graph's all-used persistent gradient objects.

    Construct only after backward has proved the native model's parameter and
    gradient coverage. Validation reads storage metadata and never drains a
    device or performs a collective. Current-stream and NCCL dependencies order
    backward, packing, reduction and copies; the numerical payload is identical
    to the general dynamic path. A changed or unused gradient is an error, not a
    silent switch to a different synchronization contract.
    """

    def __init__(self, parameters):
        import torch

        self.parameters = tuple(parameters)
        if not self.parameters:
            raise ValueError("static gradient synchronization requires trainable parameters")
        if len({id(parameter) for parameter in self.parameters}) != len(self.parameters):
            raise ValueError("static gradient parameter coverage contains duplicates")
        groups = {}
        for parameter in self.parameters:
            if not parameter.requires_grad or parameter.grad is None:
                raise RuntimeError("static all-used synchronization found an unused parameter")
            if parameter.grad.layout != torch.strided:
                raise ValueError("static gradient synchronization requires dense gradients")
            if parameter.device != self.parameters[0].device:
                raise ValueError("static gradient synchronization requires one device")
            groups.setdefault(parameter.grad.dtype, []).append(parameter.grad)
        self._parameter_storage = tuple(_tensor_storage(parameter) for parameter in self.parameters)
        self._gradient_storage = tuple(
            _tensor_storage(parameter.grad) for parameter in self.parameters
        )
        self._groups = tuple(groups.values())

    def validate(self):
        if len(self.parameters) != len(self._parameter_storage):
            raise RuntimeError("static gradient parameter coverage changed")
        for parameter, parameter_storage, gradient_storage in zip(
            self.parameters, self._parameter_storage, self._gradient_storage, strict=True
        ):
            if not parameter.requires_grad or _tensor_storage(parameter) != parameter_storage:
                raise RuntimeError("static gradient parameter identity or storage changed")
            if parameter.grad is None:
                raise RuntimeError("static all-used synchronization found an unused parameter")
            if _tensor_storage(parameter.grad) != gradient_storage:
                raise RuntimeError("static gradient object or storage changed")

    def synchronize(self, world=None):
        import torch.distributed as dist

        self.validate()
        if world is None:
            world = dist.get_world_size()
        if not isinstance(world, int) or world < 1:
            raise ValueError("static gradient synchronization requires a positive world size")
        if world > 1:
            _synchronize_gradient_groups(self._groups, world)


def synchronize_gradients(parameters):
    """Coalesce native-dtype gradients only after all local backward work ends."""
    import torch
    import torch.distributed as dist

    parameters = list(parameters)
    if not parameters:
        raise ValueError("gradient synchronization requires trainable parameters")
    world = dist.get_world_size()
    if world == 1:
        return
    if parameters[0].is_cuda:
        torch.cuda.synchronize(parameters[0].device)
    # A native conditional parameter may be unused, but every rank must agree
    # before issuing dtype-dependent collectives.
    presence = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        dtype=torch.int32,
        device=parameters[0].device,
    )
    dist.all_reduce(presence)
    if bool(((presence != 0) & (presence != world)).any()):
        raise RuntimeError("warmup gradient presence differs across ranks")
    groups = {}
    for parameter in parameters:
        if parameter.grad is not None:
            groups.setdefault(parameter.grad.dtype, []).append(parameter.grad)
    _synchronize_gradient_groups(groups.values(), world)


def check_warmup_resume(prior, *, global_batch, context, data_contract, warmup_schedule=None):
    """Scheduling may change; the sealed data order and objective may not."""
    if (
        prior.get("global_batch") != global_batch
        or prior.get("context") != context
        or prior.get("data_contract") != data_contract
    ):
        raise ValueError("resume would change the sealed warmup data order")
    saved = prior.get("communication")
    if saved is not None and saved.get("reduction") != "mean":
        raise ValueError("resume would change the warmup gradient reduction")
    if prior.get("trainable_mode", "adapter") == "full":
        if warmup_schedule is None or prior.get("warmup_schedule") != warmup_schedule:
            raise ValueError(
                "resume would change the full-weight warmup budget or learning-rate schedule"
            )
