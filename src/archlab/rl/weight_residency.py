"""Temporarily retain FSDP-gathered policy weights during no-grad rollouts.

Only public FSDP mutation APIs are used. The pinned Torch state is read because
there is no public getter for original forward-reshard policies or padded gather
allocation sizes. Unsupported automatic/partial policies fail closed. This is
weight residency, not KV caching and not a remedy for policy replay differences.
"""

from __future__ import annotations

from contextlib import contextmanager
import math

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

_GIB = 1024 ** 3


def _collective_check(errors, *, memory=False):
    packets = [errors]
    if dist.is_initialized():
        packets = [None] * dist.get_world_size()
        dist.all_gather_object(packets, errors)
    if any(packets):
        kind = MemoryError if memory else ValueError
        raise kind(f"weight residency admission failed: {packets}")


def _free_memory(device):
    if device.type != "cuda":
        raise ValueError("weight residency requires a CUDA actor")
    return int(torch.cuda.mem_get_info(device)[0])


def _unused_allocator_cache(device):
    if device.type != "cuda":
        return 0
    return max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))


def _release_unused_cache(device):
    # Entry-only call, after no-gradient/fully-sharded admission. Remeasure
    # driver free memory afterward; reserved-minus-allocated is not a guarantee.
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


def _local(parameter):
    return parameter.to_local() if isinstance(parameter, DTensor) else parameter


def _inventory(model):
    result = {}
    for name, parameter in model.named_parameters():
        local = _local(parameter)
        mesh = None
        if isinstance(parameter, DTensor):
            mesh = (tuple(parameter.device_mesh.mesh.flatten().tolist()),
                    tuple(str(placement) for placement in parameter.placements))
        result[name] = (id(parameter), tuple(parameter.shape), tuple(local.shape),
                        str(parameter.dtype), str(parameter.device), parameter.requires_grad, mesh)
    return result


def _state_plan(model):
    """Read only the installed, qualified Torch FSDP state layout."""
    plan, seen = [], set()
    for name, module in model.named_modules():
        if not isinstance(module, FSDPModule):
            continue
        state = module._get_fsdp_state()
        if id(state) in seen:
            continue
        seen.add(id(state))
        groups = state._fsdp_param_groups
        if not groups:
            continue
        if state._auto_reshard_after_forward is not False:
            raise ValueError("use explicit boolean FSDP forward-reshard policies before residency")
        policies = []
        extra = 0
        for group in groups:
            if group._sharded_state.name != "SHARDED":
                raise ValueError("enter residency with every managed group fully sharded")
            if group.post_forward_mesh_info is None:
                policies.append(False)
            elif group.post_forward_mesh_info is group.mesh_info:
                policies.append(True)
            else:
                raise ValueError("partial/integer forward-reshard policies are unsupported")
            for parameter in group.fsdp_params:
                if parameter.offload_to_cpu:
                    raise ValueError("CPU-offloaded parameters are outside the residency contract")
                if getattr(parameter, "_extensions_data", None) is not None:
                    raise ValueError("custom all-gather extensions need a separate memory contract")
                unsharded = getattr(parameter, "_unsharded_param", None)
                if (getattr(parameter, "unsharded_accumulated_grad", None) is not None
                        or unsharded is not None and unsharded.grad is not None):
                    raise ValueError("release accumulated and unsharded gradients before residency")
                if getattr(parameter.sharded_param, "_nemo_model_owned_grad_divisor", None) is not None:
                    raise ValueError("a model-owned table must not be managed by FSDP")
                dtype = group.mp_policy.param_dtype or parameter.sharded_param.dtype
                element_size = torch.empty((), dtype=dtype).element_size()
                # This includes FSDP padding and preserves existing TP/EP ownership.
                extra += parameter._sharded_param_data.numel() * parameter.mesh_info.shard_mesh_size * element_size
        if len(set(policies)) != 1:
            raise ValueError("one FSDP module has mixed per-group reshard policies")
        plan.append({"name": name, "module": module, "state": state,
                     "groups": groups, "original_policy": policies[0], "extra_bytes": extra})
    if not plan:
        raise ValueError("actor has no FSDP-managed parameters")
    return plan


class WeightResidency:
    """Context-owned inference-head callback; never closes an individual head."""

    def __init__(self, plan, device, reserve):
        self.plan = plan
        self.device = device
        self.reserve = reserve
        self.active = False
        self.receipt = {
            "format": "archlab-retained-fsdp-weights-v1", "torch": torch.__version__,
            "scope": "no-grad weight residency; no KV cache or operator change",
            "estimated_gather_bytes": sum(item["extra_bytes"] for item in plan),
            "minimum_free_bytes": reserve, "minimum_observed_free_bytes": None,
            "entry_cache_release_attempted": False, "entry_cache_reclaimed_bytes": 0,
            "entry_driver_free_before_cache_release": None,
            "managed_modules": [{"name": item["name"], "original_reshard_after_forward": item["original_policy"],
                                 "estimated_gather_bytes": item["extra_bytes"]} for item in plan],
            "cleanup_verified": False,
        }

    def _guard(self, *, allow_entry_cache_release=False):
        remaining = sum(item["extra_bytes"] for item in self.plan
                        if any(group._sharded_state.name == "SHARDED" for group in item["groups"]))
        errors = []
        try:
            free = _free_memory(self.device)
            if (allow_entry_cache_release and free - remaining < self.reserve
                    and _unused_allocator_cache(self.device) > 0):
                self.receipt["entry_cache_release_attempted"] = True
                self.receipt["entry_driver_free_before_cache_release"] = free
                _release_unused_cache(self.device)
                after = _free_memory(self.device)
                self.receipt["entry_cache_reclaimed_bytes"] = max(0, after - free)
                free = after
            previous = self.receipt["minimum_observed_free_bytes"]
            self.receipt["minimum_observed_free_bytes"] = free if previous is None else min(previous, free)
            if free - remaining < self.reserve:
                errors.append(f"free={free}, remaining_gathers={remaining}, reserve={self.reserve}")
        except (ValueError, RuntimeError) as error:
            errors.append(str(error))
        _collective_check(errors, memory=True)

    def _before_forward(self, module, args):
        if not self.active or torch.is_grad_enabled():
            raise RuntimeError("retained-weight forwards must run inside the no-grad residency context")
        state = module._get_fsdp_state()
        if any(group._sharded_state.name == "SHARDED" for group in state._fsdp_param_groups):
            self._guard()

    @contextmanager
    def inference_head(self, head):
        """Drop-in callback for the sampler's no-grad FP32 vocabulary projection."""
        if not self.active or torch.is_grad_enabled():
            raise RuntimeError("inference head requires the active no-grad residency context")
        if not any(item["module"] is head for item in self.plan):
            raise ValueError("head is not managed by this residency context")
        self._before_forward(head, ())
        head.unshard()  # Public and synchronous; already-gathered heads are a no-op.
        if isinstance(head.weight, DTensor) or head.weight.dtype != torch.float32:
            raise ValueError("expected the existing gathered FP32 vocabulary head")
        yield head
        # Outer context owns cleanup, so repeated decoding does not reshard here.


@contextmanager
def retained_fsdp_weights(model, *, minimum_free_gib=16.):
    """Retain gathered weights lazily, then restore parameter ownership/policies.

    Enter only between optimizer operations with no live gradients. A conservative
    admission check reserves all remaining padded gathers plus >=16 GiB; the same
    check runs collectively immediately before each first module gather. It does
    not guarantee activation/workspace peaks, which require separate qualification.
    No parallel/eager unshard is launched. Module prehooks retain ordinary forward
    order; the caller passes ``residency.inference_head`` to its rollout sampler.

    Original explicit True/False policies are read from pinned Torch state and
    restored using public setters. Auto-selected or partial/integer policies are
    rejected rather than changing private policy state. All managed modules are
    explicitly reshared in finally, including after an exception. The parameter
    inventory, TP/EP/owner placement and requires_grad flags must match on exit.
    """
    errors = []
    try:
        if not math.isfinite(minimum_free_gib) or minimum_free_gib < 16:
            raise ValueError("minimum_free_gib must be finite and at least 16")
        if getattr(model, "_archlab_weight_residency_active", False):
            raise ValueError("nested weight residency is not supported")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("release all gradients before entering weight residency")
        plan = _state_plan(model)
        before = _inventory(model)
        device = next(model.parameters()).device
        if any(parameter.device != device for parameter in model.parameters()):
            raise ValueError("all actor parameters must reside on the local device")
        residency = WeightResidency(plan, device, math.ceil(minimum_free_gib * _GIB))
    except (AttributeError, TypeError, ValueError, StopIteration) as error:
        errors.append(str(error))
    _collective_check(errors)
    # A previous completed rollout may leave its freed gather buffers cached in
    # the allocator. Release only unused cache if necessary, then trust measured
    # driver memory rather than assuming every cached block can be reused.
    residency._guard(allow_entry_cache_release=True)
    hooks, changed = [], []
    original_flag = getattr(model, "_archlab_weight_residency_active", None)
    body_error = None
    try:
        model._archlab_weight_residency_active = True
        for item in plan:
            module = item["module"]
            module.set_reshard_after_forward(False, recurse=False)
            changed.append(item)
            hooks.append(module.register_forward_pre_hook(residency._before_forward, prepend=True))
        residency.active = True
        with torch.no_grad():
            yield residency
    except BaseException as error:
        body_error = error
        raise
    finally:
        residency.active = False
        cleanup_errors = []
        if any(parameter.grad is not None for parameter in model.parameters()):
            cleanup_errors.append("a gradient appeared inside the no-grad residency context")
        for name, parameter in model.named_parameters():
            if name in before and parameter.requires_grad != before[name][5]:
                cleanup_errors.append(f"requires_grad changed during residency: {name}")
        for hook in hooks:
            hook.remove()
        for item in reversed(changed):
            try:
                item["module"].reshard()
            except Exception as error:
                cleanup_errors.append(f"reshard {item['name']}: {type(error).__name__}")
            try:
                item["module"].set_reshard_after_forward(item["original_policy"], recurse=False)
            except Exception as error:
                cleanup_errors.append(f"restore policy {item['name']}: {type(error).__name__}")
        if original_flag is None:
            delattr(model, "_archlab_weight_residency_active")
        else:
            model._archlab_weight_residency_active = original_flag
        if _inventory(model) != before:
            cleanup_errors.append("parameter identity, ownership, dtype or gradient flags changed")
        if any(parameter.grad is not None for parameter in model.parameters()):
            cleanup_errors.append("live gradients remained after residency cleanup")
        for item in plan:
            state = item["module"]._get_fsdp_state()
            if (state._auto_reshard_after_forward is not False
                    or any(group._sharded_state.name != "SHARDED" for group in item["groups"])):
                cleanup_errors.append(f"FSDP ownership not restored: {item['name']}")
            for group in item["groups"]:
                expected = group.mesh_info if item["original_policy"] else None
                if group.post_forward_mesh_info is not expected:
                    cleanup_errors.append(f"FSDP policy not restored: {item['name']}")
        cleanup_packets = [cleanup_errors]
        if dist.is_initialized():
            cleanup_packets = [None] * dist.get_world_size()
            dist.all_gather_object(cleanup_packets, cleanup_errors)
        residency.receipt["cleanup_verified"] = not any(cleanup_packets)
        if any(cleanup_packets):
            raise RuntimeError(f"weight residency cleanup failed: {cleanup_packets}") from body_error
