"""Lower replay activation peaks without changing native kernels or ownership.

A private copy of the project-owned MoE orchestration accumulates unique-token slots in
place, avoiding a full FP32 token buffer copy on every expert slot. Container
modules, global dispatchers, and parameter registration remain unchanged.
"""

import ast
import inspect
import textwrap
from types import FunctionType, MethodType


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
