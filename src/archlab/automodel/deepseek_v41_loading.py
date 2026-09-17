"""Strict, streamed loading of native V4.1 with explicit expert/Engram owners.

No source shard is modified. Only the official wo_a FP8-to-BF16 conversion and
exact BF16-to-FP32 head/pooling-compressor promotions are allowed. Inactive vision/MTP tensors remain in
the original checkpoint; remote experts are owned by another EP rank.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from archlab.architectures.deepseek_v41_math import dequantize_frozen_weight


@torch.no_grad()
def load_native_ep_checkpoint(model, checkpoint: Path, *, ep_rank: int, ep_size: int,
                              engram_rank: int | None = None, engram_size: int | None = None,
                              chunk_bytes: int = 16 * 1024**2, require_verified_cache=True):
    """Load owned experts and contiguous Engram rows; row owners default to EP."""
    engram_rank = ep_rank if engram_rank is None else engram_rank
    engram_size = ep_size if engram_size is None else engram_size
    if (not 0 <= ep_rank < ep_size or not 0 <= engram_rank < engram_size
            or chunk_bytes < 1024):
        raise ValueError("invalid checkpoint loading geometry")
    index_path = checkpoint / "model.safetensors.index.json"
    raw_index = index_path.read_bytes()
    index_sha = hashlib.sha256(raw_index).hexdigest()
    if require_verified_cache:
        cache = json.loads((checkpoint / "ARCHLAB_VERIFIED_COPY.json").read_text())
        if cache["source_index_sha256"] != index_sha:
            raise ValueError("cache index no longer matches the verified copy")
    mapping = json.loads(raw_index)["weight_map"]
    if any(Path(p).name != p for p in mapping.values()):
        raise ValueError("weight shards must be direct children of the checkpoint")
    parameters = dict(model.named_parameters())
    if any(p.requires_grad for p in parameters.values()):
        raise ValueError("load the frozen base before attaching trainable modules")
    missing = parameters.keys() - mapping.keys()
    if missing:
        raise ValueError(f"checkpoint is missing native parameters: {sorted(missing)[:20]}")
    converted_scales = {k.removesuffix("weight") + "scale"
                        for k in parameters if k.endswith(".attn.wo_a.weight")}
    if not converted_scales <= mapping.keys():
        raise ValueError("native grouped output-projection scales are missing")
    unused = Counter()
    for name in mapping.keys() - parameters.keys() - converted_scales:
        if name.startswith(("vision.", "aligner.", "mtp.")) or name in ("image_start", "image_end", "image_newline"):
            unused["inactive-vision-or-dspark"] += 1
        elif name.endswith(".ffn.gate.bias_vl"):
            unused["inactive-vision-router-bias"] += 1
        elif match := re.fullmatch(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)", name):
            layer_id, expert = map(int, match.group(1, 2))
            local = model.layers[layer_id].ffn.local_experts
            if expert >= local * ep_size:
                raise ValueError(f"expert has no owner in the EP mesh: {name}")
            if ep_rank * local <= expert < (ep_rank + 1) * local:
                raise ValueError(f"unexpected unloaded local expert: {name}")
            unused["other-EP-rank-expert"] += 1
        else:
            raise ValueError(f"unaccounted pretrained tensor: {name}")
    by_shard = defaultdict(list)
    for name in parameters:
        by_shard[mapping[name]].append(name)
    loaded, conversions = [], []
    for filename, names in sorted(by_shard.items()):
        # Keep safetensors' internal tensor constructors on CPU even when the
        # reference has set a process-wide CUDA default. Only bounded chunks
        # below may move to the destination GPU.
        with torch.device("cpu"), safe_open(checkpoint / filename, framework="pt", device="cpu") as reader:
            for name in names:
                target = parameters[name]
                source = reader.get_slice(name)
                source_shape = source.get_shape()
                first, count = 0, source_shape[0]
                sharded = bool(re.fullmatch(r"layers\.\d+\.engram\.embed\.(weight|scale)", name))
                if sharded:
                    rows = (source_shape[0] + engram_size - 1) // engram_size
                    first = engram_rank * rows
                    count = max(0, min(rows, source_shape[0] - first))
                    if list(target.shape) != [rows, *source_shape[1:]]:
                        raise ValueError(f"Engram shard shape mismatch: {name}")
                    if count < rows:
                        padding = target[count:]
                        padding.copy_(torch.full(padding.shape, 1 if name.endswith(".scale") else 0,
                                                 device=padding.device, dtype=torch.float32).to(padding.dtype))
                elif list(target.shape) != source_shape:
                    raise ValueError(f"native tensor shape mismatch: {name}: {source_shape} vs {target.shape}")
                is_grouped_output = name.endswith(".attn.wo_a.weight")
                scale = None
                if is_grouped_output:
                    if target.dtype != torch.bfloat16:
                        raise ValueError("native grouped output projection must be BF16")
                    scale_name = name.removesuffix("weight") + "scale"
                    with safe_open(checkpoint / mapping[scale_name], framework="pt", device="cpu") as scales:
                        scale = scales.get_tensor(scale_name)
                    conversions.append({"name": name, "conversion": "official-wo_a-FP8-block32-to-BF16"})
                row_size = max(1, target[0].numel() * target.element_size())
                rows_per_chunk = max(32, (chunk_bytes // row_size // 32) * 32)
                for start in range(0, count, rows_per_chunk):
                    end = min(start + rows_per_chunk, count)
                    values = source[first + start:first + end]
                    if is_grouped_output:
                        # safetensors slices can follow the process default
                        # device while get_tensor scales remain on CPU. Decode
                        # only this bounded chunk on the explicit destination.
                        values = dequantize_frozen_weight(
                            values.to(target.device),
                            scale[start // 32:(end + 31) // 32].to(target.device),
                        )
                    elif target.dtype == getattr(torch, "float4_e2m1fn_x2", None):
                        if values.dtype != torch.int8:
                            raise ValueError(f"expected on-disk packed I8 expert: {name}")
                        values = values.view(torch.float4_e2m1fn_x2)
                    elif (values.dtype == torch.bfloat16 and target.dtype == torch.float32
                          and (name == "head.weight" or re.fullmatch(
                              r"layers\.\d+\.attn\.compressor\.(wkv|wgate)\.weight", name))):
                        # The reference explicitly promotes pooling weights;
                        # BF16 values are exactly representable in FP32. The
                        # ratio-1 compressor stays BF16 in its constructor.
                        if start == 0:
                            conversions.append({"name": name, "conversion": "exact-BF16-to-FP32"})
                    elif values.dtype != target.dtype:
                        raise ValueError(f"unapproved dtype conversion: {name}: {values.dtype} -> {target.dtype}")
                    target[start:end].copy_(values)
                loaded.append(name)
        print(json.dumps({"event": "native_checkpoint_shard_loaded", "ep_rank": ep_rank,
                          "shard": filename, "tensors": len(names)}), flush=True)
    if set(loaded) != parameters.keys():
        raise RuntimeError("not all native parameters were loaded")
    return {"index_sha256": index_sha, "loaded_tensors": len(loaded), "unused_tensors": dict(unused),
            "source_tensors": len(mapping), "ep_rank": ep_rank, "ep_size": ep_size,
            "engram_ep_rank": engram_rank, "engram_rank": engram_rank, "engram_size": engram_size,
            "conversions": conversions,
            "resident_parameter_bytes": sum(p.numel() * p.element_size() for p in parameters.values()),
            "source_unchanged": True, "verified_cache_required": require_verified_cache}
