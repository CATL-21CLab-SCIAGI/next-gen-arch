"""Explicit opt-in contract for the fresh V4.1 performance experiment."""

import json
from pathlib import Path


def read_performance_contract(path):
    if path is None:
        return None
    value = json.loads(Path(path).read_text())
    required = {
        "format",
        "grouped_experts",
        "scale_engram",
        "engram_optimizer",
        "table_lr_scale",
        "batched_updates",
        "optimized_synchronization",
        "trim_alignment",
        "head_loss_chunk",
        "router_bias_rate",
        "retain_activations",
        "compile_hc_backward",
        "expert_dispatcher",
        "microbatch",
    }
    version = value.get("format")
    if version == "archlab-v41-performance-v2":
        required.add("engram_anchor_width")
    if set(value) - {"simplicial_backend"} != required or version not in (
        "archlab-v41-performance-v1", "archlab-v41-performance-v2"
    ):
        raise ValueError("unrecognized V4.1 performance contract")
    if version == "archlab-v41-performance-v2" and (
        type(value["engram_anchor_width"]) is not int or value["engram_anchor_width"] != 128
    ):
        raise ValueError("the fixed Engram cohort requires the d128 lookup budget")
    for key in (
        "grouped_experts",
        "scale_engram",
        "batched_updates",
        "optimized_synchronization",
        "retain_activations",
        "compile_hc_backward",
    ):
        if type(value[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if value["engram_optimizer"] != "sinkhorn-algorithm-1" or not value["scale_engram"]:
        raise ValueError("this Sinkhorn contract requires scaled Engram tables")
    if type(value["trim_alignment"]) is not int or value["trim_alignment"] not in (
        128,
        256,
        512,
        1024,
        2048,
    ):
        raise ValueError("trimming must preserve compression alignment")
    if type(value["head_loss_chunk"]) is not int or not 128 <= value["head_loss_chunk"] <= 2048:
        raise ValueError("invalid full-vocabulary head chunk")
    if not 0 < value["table_lr_scale"] <= 1 or not 0 < value["router_bias_rate"] <= 0.01:
        raise ValueError("invalid table learning-rate scale or router rate")
    if value["expert_dispatcher"] not in ("torch", "deepep"):
        raise ValueError("unsupported upstream expert dispatcher")
    if value["expert_dispatcher"] == "deepep" and not value["grouped_experts"]:
        raise ValueError("DeepEP requires grouped experts")
    if type(value["microbatch"]) is not int or value["microbatch"] not in (4, 8, 16, 32):
        raise ValueError("invalid controlled microbatch")
    if value.get("simplicial_backend", "deterministic") not in ("deterministic", "triton"):
        raise ValueError("unsupported simplicial training backend")
    return value


def configure_performance(model, indexers, config, context):
    if config is None:
        return
    model.lm_head._archlab_loss_chunk = config["head_loss_chunk"]
    for indexer in indexers:
        indexer._archlab_sample_context = context
    if config["compile_hc_backward"]:
        compile_hc_backward(model)


def compile_hc_backward(model):
    import torch

    from archlab.architectures.deepseek_v41_math import hc_split_sinkhorn

    reference = torch.compile(hc_split_sinkhorn, fullgraph=True, dynamic=True)
    count = 0
    for module in model.modules():
        if hasattr(module, "_archlab_native_hc"):
            module._archlab_hc_reference = reference
            count += 1
    if not count:
        raise ValueError("no trainable HC boundaries found")
    return count
