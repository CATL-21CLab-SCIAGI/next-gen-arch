"""Compose the native V4.1 backbone with explicit node-local EP, no TP/PP/CP.

Frozen weights do not have optimizer state or weight gradients. EP shards the
experts and Engram tables; remaining base weights and adapters are replicated.
This avoids requiring FSDP to gather quantized weights for every sublayer.
Qualification belongs to the launch contract, not to this constructor.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter
from archlab.architectures.deepseek_v41_normal_adapter import (
    V41NormalAttentionAdapter,
    normal_adapter_parameter_count,
)
from archlab.automodel.deepseek_v41_autograd import attach_adapters, install_frozen_backward
from archlab.automodel.deepseek_v41_loading import load_native_ep_checkpoint
from archlab.automodel.deepseek_v41_native_quantization import install_native_row_padding
from archlab.automodel.deepseek_v41_parallel import expert_parallel_classes
from archlab.automodel.deepseek_v41_runtime import load_native_reference
from archlab.optimizers.headwise_muon import HeadwiseMuon


class ReplicatedAttentionContext:
    """The reference's TP-only global sees a local attention replica, not EP."""

    @staticmethod
    def is_initialized():
        return False


def node_local_expert_group(ep_size):
    world = dist.get_world_size()
    if world % ep_size:
        raise ValueError("EP size must divide the world")
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    selected = None
    for first in range(0, world, ep_size):
        ranks = list(range(first, first + ep_size))
        if len({hosts[i] for i in ranks}) != 1:
            raise ValueError("expert groups must be node-local")
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            selected = group
    return selected


def build_replica(*, assets: Path, weights: Path, group, context: int):
    from transformers import PreTrainedTokenizerFast

    reference = load_native_reference(assets, module_name="_archlab_v41_pretrained")
    native_quantization = install_native_row_padding(reference)
    reference.dist = ReplicatedAttentionContext()
    reference.MoE, reference.ParallelEngramEmbedding = expert_parallel_classes(reference, group)
    args = reference.ModelArgs(**json.loads((assets / "inference/config.json").read_text()))
    args.max_batch_size, args.max_seq_len = 1, context
    args.vision_n_layers = 0
    args.n_mtp_layers, args.dspark_block_size, args.dspark_target_layer_ids = 0, 0, ()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(assets, local_files_only=True)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    model = reference.Transformer(args, tokenizer=tokenizer)
    model.requires_grad_(False)
    from archlab.automodel.deepseek_v41_pytorch import bf16_memory_plan

    capacity = bf16_memory_plan(model)
    print(
        json.dumps(
            {"event": "bf16_capacity_before_checkpoint_io", "rank": dist.get_rank(), **capacity}
        ),
        flush=True,
    )
    if not capacity["fits"]:
        raise MemoryError(f"BF16 capacity gate failed before reading weights: {capacity}")
    report = load_native_ep_checkpoint(
        model, weights, ep_rank=dist.get_rank(group), ep_size=dist.get_world_size(group)
    )
    report["native_quantization"] = native_quantization
    report["architecture"] = {
        "width": args.dim,
        "layers": args.n_layers,
        "streams": args.hc_mult,
        "experts": args.n_routed_experts,
        "active_experts": args.n_activated_experts,
        "engram_layers_0based": args.engram_layer_ids,
    }
    return reference, model, report


def install_training_branches(reference, model, *, seed=42):
    if not getattr(reference, "_archlab_backward_installed", False):
        install_frozen_backward(reference)
    config = V41AdapterConfig()
    old_dtype, old_device = torch.get_default_dtype(), torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cpu")
        adapters = {
            i: V41SimplicialAdapter(config, seed=seed + i).cuda()
            for i in (4, 9, 14, 19, 24, 29, 34, 39)
        }
    finally:
        torch.set_default_dtype(old_dtype)
        torch.set_default_device(old_device)
    attach_adapters(model, adapters)
    return adapters


def enable_activation_checkpointing(model):
    """Only pure mHC math and stateless MoE; never recompute shared-KV caches."""
    for layer in model.layers:
        original_mixes, original_moe = layer.hc_mixes, layer.ffn.forward

        def mixes(x, *coefficients, _original=original_mixes):
            if not torch.is_grad_enabled() or not x.requires_grad:
                return _original(x, *coefficients)
            return checkpoint(_original, x, *coefficients, use_reentrant=False)

        def moe(x, image_mask=None, _original=original_moe):
            if not torch.is_grad_enabled() or not x.requires_grad:
                return _original(x, image_mask)
            # Different expert assignments must not cause different ranks to
            # skip reverse/recomputation collectives.
            with set_checkpoint_early_stop(False):
                return checkpoint(_original, x, image_mask, use_reentrant=False)

        layer.hc_mixes, layer.ffn.forward = mixes, moe


def adapter_optimizers(model, adapters, *, lr=1e-7):
    heads, matrices, norms, scalars = [], [], [], []
    kinds = {type(adapter) for adapter in adapters.values()}
    if len(adapters) != 8 or kinds not in ({V41SimplicialAdapter}, {V41NormalAttentionAdapter}):
        raise ValueError("optimizer requires eight adapters of one reviewed variant")
    normal = kinds == {V41NormalAttentionAdapter}
    for adapter in adapters.values():
        if adapter.config != V41AdapterConfig():
            raise ValueError("optimizer partition expects the reviewed adapter geometry")
        keys = [adapter.k] if normal else [adapter.k1, adapter.k2]
        values = [adapter.v] if normal else [adapter.v1, adapter.v2]
        key_norms = [adapter.k_norm] if normal else [adapter.k1_norm, adapter.k2_norm]
        heads.extend([adapter.q.weight, *(key.weight for key in keys)])
        matrices.extend(
            [*(value.weight for value in values), adapter.output.weight, adapter.output_gate.weight]
        )
        norms.extend(
            [adapter.input_norm.weight, adapter.q_norm.weight, *(norm.weight for norm in key_norms)]
        )
        scalars.extend([adapter.read_logits, adapter.write_logits])
    combined = heads + matrices + norms + scalars
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if len({id(p) for p in combined}) != len(combined) or {id(p) for p in combined} != trainable:
        raise ValueError("optimizer partition is not exhaustive/disjoint or includes the base")
    expected = 8 * (
        normal_adapter_parameter_count(V41AdapterConfig())
        if normal
        else V41AdapterConfig().parameter_count()
    )
    if sum(p.numel() for p in combined) != expected or any(
        p.dtype != torch.float32 for p in combined
    ):
        raise ValueError("wrong adapter parameter budget/master precision")
    muon = HeadwiseMuon([{"params": heads, "head_dim": 128}, {"params": matrices}], lr=lr)
    adam = torch.optim.AdamW(
        [{"params": norms, "weight_decay": 0.1}, {"params": scalars, "weight_decay": 0}],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-20,
        foreach=False,
    )
    return [muon, adam]
