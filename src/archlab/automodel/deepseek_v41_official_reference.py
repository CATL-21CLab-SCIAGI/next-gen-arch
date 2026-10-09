"""Released V4.1 inference structure with the native trainer's BF16 GEMMs.

This is a numerical reference, never a training backend. The pinned inference
module owns the complete forward, sparse attention, mHC, and cache/index
rounding. Project-owned transport shards experts and Engram independently.
Only frozen weight decoding and the dense/expert GEMM precision are changed;
the existing stable FP8 quantizer preserves the released KV rounding equations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from archlab.automodel.deepseek_v41_execution import ReplicatedAttentionContext
from archlab.automodel.deepseek_v41_loading import load_native_ep_checkpoint
from archlab.automodel.deepseek_v41_native_quantization import install_native_row_padding
from archlab.automodel.deepseek_v41_parallel import expert_parallel_classes
from archlab.automodel.deepseek_v41_pytorch import bf16_memory_plan, dequantize_base_once
from archlab.automodel.deepseek_v41_runtime import REFERENCE_DIGESTS, load_native_reference


def _co_resident_reserve(model, *, other_allocated_bytes: int, workspace_gib: float) -> tuple[float, dict]:
    """Include other live PyTorch tensors in the existing standalone-base gate."""
    if not math.isfinite(workspace_gib) or workspace_gib < 0 or other_allocated_bytes < 0:
        raise ValueError("reference capacity reserves must be finite and nonnegative")
    tensors = (*model.parameters(), *model.buffers())
    own_bytes = sum(t.numel() * t.element_size() for t in tensors if t.device.type == "cuda")
    # Include tensors the reference constructor may retain outside its module
    # tree. The initial allocation measurement also protects co-resident FSDP
    # weights even if the allocator's bookkeeping differs during construction.
    other_bytes = max(other_allocated_bytes, torch.cuda.memory_allocated() - own_bytes, 0)
    combined = workspace_gib + other_bytes / 2**30
    return combined, {"co_resident_tensor_gib": other_bytes / 2**30,
                      "reference_workspace_gib": workspace_gib,
                      "combined_reserve_gib": combined}


def build_matching_precision_reference(
    *, assets: Path, weights: Path, expert_group, context: int,
    engram_group=None, reserve_gib: float = 24,
    module_name: str = "_archlab_v41_official_precision_reference",
):
    """Load an independent frozen reference without altering runtime packages.

    ``expert_group`` selects native expert owners (normally node-local EP8).
    ``engram_group`` defaults to WORLD, independently selecting the contiguous
    memory row owners. Every participating rank must call this and subsequent
    forwards in the same order, while each may supply its own text window.

    The returned ``(reference_module, model, report)`` uses BF16 dense/expert
    GEMMs, native FP8 KV and FP4 index rounding, and an FP32 vocabulary head.
    The reference is kept resident for parity; callers own its lifetime.
    """
    if not dist.is_initialized() or context < 1:
        raise ValueError("the distributed reference requires an initialized group and positive context")
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("reference workspace reserve must be finite and nonnegative")
    if engram_group is None:
        engram_group = dist.group.WORLD
    expert_rank, expert_size = dist.get_rank(expert_group), dist.get_world_size(expert_group)
    engram_rank, engram_size = dist.get_rank(engram_group), dist.get_world_size(engram_group)
    if min(expert_rank, engram_rank) < 0:
        raise ValueError("this rank must belong to both reference ownership groups")
    device = torch.device("cuda", torch.cuda.current_device())
    co_resident_bytes = torch.cuda.memory_allocated(device)
    reference = load_native_reference(Path(assets), module_name=module_name)
    quantization = dict(install_native_row_padding(reference))
    reference.dist = ReplicatedAttentionContext()
    reference.MoE, _ = expert_parallel_classes(reference, expert_group)
    _, reference.ParallelEngramEmbedding = expert_parallel_classes(reference, engram_group)
    args = reference.ModelArgs(**json.loads((Path(assets) / "inference/config.json").read_text()))
    args.max_batch_size, args.max_seq_len = 1, context
    args.vision_n_layers = 0
    args.n_mtp_layers, args.dspark_block_size, args.dspark_target_layer_ids = 0, 0, ()
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(assets, local_files_only=True)
    # The released module uses ambient factories for caches, rotary frequencies,
    # and norms. Scope those defaults rather than changing the training process.
    with reference.set_dtype(torch.bfloat16), torch.device(device):
        model = reference.Transformer(args, tokenizer=tokenizer).requires_grad_(False).eval()
        combined_reserve, co_resident = _co_resident_reserve(
            model, other_allocated_bytes=co_resident_bytes, workspace_gib=reserve_gib)
        capacity = bf16_memory_plan(model, reserve_gib=combined_reserve)
        print(json.dumps({"event": "official_reference_capacity_before_checkpoint_io",
                          "rank": dist.get_rank(), **capacity, **co_resident}), flush=True)
        if not capacity["fits"]:
            raise MemoryError(f"co-resident BF16 reference capacity gate failed: {capacity}, {co_resident}")
        report = load_native_ep_checkpoint(
            model, Path(weights), ep_rank=expert_rank, ep_size=expert_size,
            engram_rank=engram_rank, engram_size=engram_size)
        decoding = dequantize_base_once(model, reserve_gib=combined_reserve)
    # No linear input quantization: match AutoModel's torch/BF16 projections.
    # The quantizers called directly by attention/indexers remain installed.
    reference.linear = F.linear
    if model.head.weight.dtype != torch.float32:
        raise RuntimeError("released reference vocabulary head lost FP32 precision")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("the matching-precision reference must remain frozen")
    quantization["native_gemms_retained"] = False
    quantization["linear_backend"] = "torch-bf16-without-linear-input-quantization"
    report.update({
        "reference_source_sha256": dict(REFERENCE_DIGESTS),
        "native_quantization": quantization,
        "expert_owner_ranks": dist.get_process_group_ranks(expert_group),
        "engram_owner_ranks": dist.get_process_group_ranks(engram_group),
        "world_size": dist.get_world_size(), "attention_tp_size": 1,
        "decoding": decoding, "capacity": capacity, **co_resident,
        "architecture": {"width": args.dim, "layers": args.n_layers, "streams": args.hc_mult,
                         "experts": args.n_routed_experts, "active_experts": args.n_activated_experts,
                         "engram_layers_0based": args.engram_layer_ids},
        "forward_owner": "pinned-released-Transformer.forward",
        "sparse_attention": "pinned-released-native-kernel", "mhc": "pinned-released-native-kernel",
        "lm_head_dtype": str(model.head.weight.dtype),
    })
    return reference, model, report


@torch.inference_mode()
def reference_hidden(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Capture final normalized hidden states through the unchanged forward.

    The released head computes only the final position, so the temporary hook
    captures all normalized positions for bounded full-vocabulary comparisons.
    Its generation sampling is ignored and does not advance training RNG state.
    """
    if (input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1
            or input_ids.shape[1] > model.max_seq_len):
        raise ValueError("reference expects one independent nonempty text window within configured context")
    device = next(model.parameters()).device
    if input_ids.device != device:
        raise ValueError("reference tokens and parameters must share a device")
    captured = []
    handle = model.norm.register_forward_hook(lambda _module, _args, output: captured.append(output))
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        devices = [device.index] if device.type == "cuda" else []
        with torch.device(device), torch.random.fork_rng(devices=devices):
            model(input_ids, start_pos=0)
    finally:
        torch.set_default_dtype(old_dtype)
        handle.remove()
    if len(captured) != 1 or captured[0].shape[:2] != input_ids.shape:
        raise RuntimeError("reference did not produce one final normalized state per input token")
    return captured[0]
