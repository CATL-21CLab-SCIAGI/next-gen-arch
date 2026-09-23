"""Lower replay activation peaks without changing native kernels or ownership.

A private copy of the project-owned MoE orchestration accumulates unique-token slots in
place, avoiding a full FP32 token buffer copy on every expert slot. Container
modules, global dispatchers, and parameter registration remain unchanged.
"""

import ast
import inspect
import textwrap
from types import FunctionType, MethodType

import torch
from torch.nn import functional as F


def _unpack_hc_activation(saved):
    if isinstance(saved, torch.Tensor):
        return saved
    device, dtype, host, shape, stride, offset = saved
    return host.to(device=device, dtype=dtype).as_strided(shape, stride, offset)


def _hc_forward_with_offload(self, hidden_states):
    from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41Mix

    from archlab.automodel.deepseek_v41_full_boundaries import (
        NativeTrainableHC,
        trainable_hc_forward,
    )

    if not torch.is_grad_enabled():
        return trainable_hc_forward(self, hidden_states)
    if hidden_states.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("HC activation offload requires an exactly reversible FP32 promotion")
    with torch.autocast(hidden_states.device.type, enabled=False):
        flat = hidden_states.flatten(2).float()
        if not flat.is_contiguous():
            raise ValueError("HC activation offload requires contiguous residual streams")
        key = flat.untyped_storage().data_ptr()
        host = None

        def pack(tensor):
            nonlocal host
            if tensor.device != flat.device or tensor.untyped_storage().data_ptr() != key:
                return tensor  # Projection weights and all other saved tensors stay on GPU.
            if host is None:
                # flat is exclusively hidden_states.float(): storing the original
                # activation dtype and promoting back is lossless, not quantization.
                host = torch.empty(
                    flat.numel(), dtype=hidden_states.dtype, device="cpu", pin_memory=True
                )
                host.copy_(flat.detach().reshape(-1))
                self._archlab_hc_offload_stats["tensor_copies"] += 1
                self._archlab_hc_offload_stats["copied_bytes"] += host.numel() * host.element_size()
            return (
                tensor.device,
                tensor.dtype,
                host,
                tensor.shape,
                tensor.stride(),
                tensor.storage_offset() - flat.storage_offset(),
            )

        with torch.autograd.graph.saved_tensors_hooks(pack, _unpack_hc_activation):
            mixes = F.linear(flat, self.fn) * torch.rsqrt(
                flat.square().mean(-1, keepdim=True) + self.norm_eps
            )
        scale, base = self.scale.clone(), self.base.clone()
        values = NativeTrainableHC.apply(
            mixes, scale, base, self.streams, self.iterations, self.eps, self._archlab_native_hc
        )
    return DeepseekV41Mix(*values)


def install_hc_activation_offload(model):
    from archlab.automodel.deepseek_v41_full_boundaries import trainable_hc_forward

    selected = [
        (name, module)
        for name, module in model.named_modules()
        if getattr(module.forward, "__func__", None) is trainable_hc_forward
    ]
    if not selected:
        raise ValueError("expected the qualified trainable native HC forward")
    before = {name: id(parameter) for name, parameter in model.named_parameters()}
    for _, module in selected:
        module._archlab_hc_offload_stats = {"tensor_copies": 0, "copied_bytes": 0}
        module.forward = MethodType(_hc_forward_with_offload, module)
    if before != {name: id(parameter) for name, parameter in model.named_parameters()}:
        raise RuntimeError("HC activation offload changed parameter ownership")
    return {
        "enabled": True,
        "kind": "HC-projection-activation-exact-host-roundtrip-v1",
        "modules": [name for name, _ in selected],
        "weights_offloaded": False,
        "deduplicated_saved_views": True,
        "parameter_identity_preserved": True,
    }


def hc_offload_statistics(model):
    rows = [
        module._archlab_hc_offload_stats
        for module in model.modules()
        if hasattr(module, "_archlab_hc_offload_stats")
    ]
    return {key: sum(row[key] for row in rows) for key in ("tensor_copies", "copied_bytes")}


def inplace_native_expert_function():
    from archlab.automodel.deepseek_v41_official_moe import _native_up_grouped_down

    parsed = ast.parse(textwrap.dedent(inspect.getsource(_native_up_grouped_down)))
    changed = 0
    for node in ast.walk(parsed):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "result"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "result"
            and node.value.func.attr == "index_add"
        ):
            node.value.func.attr = "index_add_"
            changed += 1
    if changed != 1:
        raise ValueError("native MoE must contain exactly one reviewed ordered result.index_add")
    namespace = dict(_native_up_grouped_down.__globals__)
    exec(compile(ast.fix_missing_locations(parsed), __file__, "exec"), namespace)
    return namespace[_native_up_grouped_down.__name__]


def install_inplace_moe_accumulation(model):
    from archlab.automodel.deepseek_v41_official_moe import _fp32_grouped_experts_forward

    selected = [
        (name, module)
        for name, module in model.named_modules()
        if getattr(module.forward, "__func__", None) is _fp32_grouped_experts_forward
    ]
    if not selected:
        raise ValueError("expected the project-qualified native FP32 expert orchestration")
    before = {name: id(p) for name, p in model.named_parameters()}
    namespace = dict(_fp32_grouped_experts_forward.__globals__)
    namespace["_native_up_grouped_down"] = inplace_native_expert_function()
    original = _fp32_grouped_experts_forward
    forward = FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    for _, module in selected:
        module.forward = MethodType(forward, module)
    if before != {name: id(p) for name, p in model.named_parameters()}:
        raise RuntimeError("MoE accumulation changed parameter ownership")
    return {
        "enabled": True,
        "kind": "ordered-native-FP32-inplace-expert-sum-v1",
        "modules": [name for name, _ in selected],
        "container_code_changed": False,
        "parameter_identity_preserved": True,
    }
