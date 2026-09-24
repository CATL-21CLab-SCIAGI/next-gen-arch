"""Bounded, checksum-verified import of the original EP8/FSDP2 parents."""

import hashlib
import json
import os
import re
from pathlib import Path

import torch
from miles.backends.megatron_utils.megatron_to_hf.deepseekv41 import convert_deepseekv41_to_hf
from miles.backends.megatron_utils.named_weights import named_params_and_buffers
from miles.backends.training_utils.parallel import get_parallel_state

from archlab.serving.v41_checkpoint_inventory import V41CheckpointInventory
from archlab.serving.v41_direct_checkpoint import owned_expert_weights
from archlab.serving.v41_export import released_name, verified_chunks


def source_names(args, name, param):
    match = re.fullmatch(r"module.module.decoder.layers.(\d+).self_attention_hyper_connection.archlab_adapter.(.+)", name)
    if match:
        return [f"layers.{match[1]}.attn_hc.simplicial_adapter.{match[2]}"]
    match = re.fullmatch(r"module.module.decoder.layers.(\d+).v41_engram.table", name)
    if match:
        return [f"layers.{match[1]}.engram.embed.weight"]
    match = re.fullmatch(r"module.module.decoder.layers.(\d+).(self_attention|mlp)_hyper_connection.alpha_(pre|post|res)", name)
    if match:
        sub = "attn" if match[2] == "self_attention" else "ffn"
        return [f"layers.{match[1]}.hc_{sub}_scale"]
    return [key for key, _ in convert_deepseekv41_to_hf(args, name, param)]


def load_row_range(inventory, name, target, first):
    columns = target.shape[1]
    end_row = min(first + target.shape[0], inventory.entries[name][0]["global_shape"][0])
    target.zero_()
    covered = 0
    for rank, entry in enumerate(inventory.entries[name]):
        offset = inventory.layouts[name][1][rank][0] * columns
        for chunk in entry["chunks"]:
            end = offset + chunk["elements"]
            lo, hi = max(first * columns, offset), min(end_row * columns, end)
            if hi > lo:
                value = next(verified_chunks(inventory, rank, {**entry, "chunks": [chunk]}))
                target.view(-1)[lo - first * columns:hi - first * columns].copy_(value[lo - offset:hi - offset])
                covered += hi - lo
            offset = end
    if covered != (end_row - first) * columns:
        raise ValueError(f"incomplete table import: {name}")


def partition_like(value, parameter, name):
    if value.shape == parameter.shape:
        return value
    parallel = get_parallel_state()
    axis = getattr(parameter, "partition_dim", -1)
    stride = getattr(parameter, "partition_stride", 1)
    if axis < 0:
        raise ValueError(f"unmarked shape mismatch: {name}: {value.shape} != {parameter.shape}")
    rank, size = parallel.tp.rank, parallel.tp.size
    if ".experts." in name:
        rank, size = parallel.etp.rank, parallel.etp.size
    if "linear_fc1" in name:
        stride = 2
    if value.shape[axis] % (size * stride):
        raise ValueError(f"nondivisible TP import: {name}")
    return torch.cat([part.chunk(size, dim=axis)[rank]
                      for part in value.chunk(stride, dim=axis)], dim=axis)


@torch.no_grad()
def load_parent(ddp_model, args, load_path):
    metadata = json.loads((Path(load_path) / "config.json").read_text())["archlab"]
    inventory = V41CheckpointInventory(metadata["full_checkpoint"])
    actual = hashlib.sha256((inventory.path / "COMPLETE.json").read_bytes()).hexdigest()
    if actual != metadata["complete_sha256"]:
        raise ValueError("parent checkpoint identity changed")
    if os.environ.get("EVERGREENTREE_WEIGHT_CACHE_DIR"):
        from archlab.megatron.miles_v41_storage import LocalWeightCache, cache_root

        rank = torch.distributed.get_rank()
        cache = LocalWeightCache(cache_root(rank))
        named = list(named_params_and_buffers(args, ddp_model))
        cache.validate(named, metadata["complete_sha256"])
        cache.restore(named)
        receipt = Path(args.save).parent / f"parent-load-rank-{rank:02d}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"checkpoint": metadata["full_checkpoint"],
                                      "complete_sha256": metadata["complete_sha256"],
                                      "loaded": [name for name, _ in named], "optimizer": "fresh",
                                      "source": "EvergreenTree validated local weight cache"}, indent=2))
        print(f"EvergreenTree rank {rank}: validated local SFT cache loaded", flush=True)
        return
    inverse = {released_name(name): name for name in inventory.entries}
    cache, loaded = {}, []
    last_expert_source = None
    parallel = get_parallel_state()
    receipt_root = Path(args.save).parent
    receipt_root.mkdir(parents=True, exist_ok=True)
    inventory_receipt = {name: {"shape": list(p.shape), "dtype": str(p.dtype),
                                "tp": getattr(p, "tensor_model_parallel", False),
                                "axis": getattr(p, "partition_dim", None),
                                "stride": getattr(p, "partition_stride", None)}
                         for name, p in named_params_and_buffers(args, ddp_model)}
    (receipt_root / f"model-layout-rank-{torch.distributed.get_rank():02d}.json").write_text(
        json.dumps(inventory_receipt, indent=2))
    for name, p in named_params_and_buffers(args, ddp_model):
        sources = source_names(args, name, p)
        if not sources:
            raise ValueError(f"parameter has no parent: {name}")
        if name.endswith("v41_engram.table"):
            load_row_range(inventory, inverse[sources[0]], p, parallel.tp.rank * p.shape[0])
        else:
            values = []
            for key in sources:
                expert = re.fullmatch(r"layers.(\d+).ffn.experts.(\d+).(w[123]).weight", key)
                if expert:
                    suffix = "down_projs" if expert[3] == "w2" else "gate_and_up_projs"
                    source = inverse[f"layers.{expert[1]}.ffn.experts.{suffix}"]
                    if source != last_expert_source:
                        cache = dict(owned_expert_weights(inventory, source, parallel.ep.rank))
                        last_expert_source = source
                    values.append(cache[key])
                else:
                    values.append(inventory.read_tensor(inverse[key], max_bytes=4 * 2**30))
            value = torch.cat(values, dim=0) if len(values) > 1 else values[0]
            if ".alpha_" in name:
                index = {"pre": 0, "post": 1, "res": 2}[name.rsplit("_", 1)[-1]]
                value = value[index].reshape_as(p)
            value = partition_like(value, p, name)
            if value.shape != p.shape:
                raise ValueError(f"parent shape mismatch: {name}: {tuple(value.shape)} != {tuple(p.shape)}; "
                                 f"axis={getattr(p, 'partition_dim', None)}, TP={parallel.tp.size}")
            p.copy_(value)
        loaded.append(name)
        if parallel.tp.rank == 0:
            print(json.dumps({"event": "miles_parent_tensor_loaded", "name": name}), flush=True)
    torch.cuda.synchronize()
    receipt = Path(args.save).parent / f"parent-load-rank-{torch.distributed.get_rank():02d}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"checkpoint": metadata["full_checkpoint"],
                                  "complete_sha256": metadata["complete_sha256"],
                                  "loaded": loaded, "optimizer": "fresh"}, indent=2))
