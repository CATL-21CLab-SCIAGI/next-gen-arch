"""Read-only export preflight for the qualified 16-rank V4.1 checkpoints.

The old format records shapes, not DTensor placements. This reader therefore
admits only its explicitly reviewed EP8/FSDP2/Engram16 layout. It does not infer
an arbitrary checkpoint layout or claim to export an engine-ready model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

_BYTES = {"torch.bfloat16": 2, "torch.float32": 4, "torch.int64": 8}


def shard_extent(size, pieces, index):
    width = (size + pieces - 1) // pieces
    start = min(index * width, size)
    return start, min(width, size - start)


def tensor_layout(entries):
    first = entries[0]
    name, shape = first["name"], first["global_shape"]
    if len(entries) != 16 or not shape:
        raise ValueError("expected 16 rank entries and a nonscalar tensor")
    for entry in entries:
        if any(entry[k] != first[k] for k in ("name", "global_shape", "dtype")):
            raise ValueError(f"rank tensor identity differs: {name}")
        if sum(chunk["elements"] for chunk in entry["chunks"]) != math.prod(entry["shape"]):
            raise ValueError(f"chunk elements do not cover local shape: {name}")
    if all(entry["shape"] == shape for entry in entries):
        signatures = [[(c["elements"], c["sha256"]) for c in e["chunks"]] for e in entries]
        if any(value != signatures[0] for value in signatures[1:]):
            raise ValueError(f"replicated tensor checksums disagree: {name}")
        return "replicated", [tuple(0 for _ in shape)] * 16
    expert = name.endswith((".ffn.experts.gate_and_up_projs", ".ffn.experts.down_projs"))
    if expert and len(shape) != 3:
        raise ValueError("expert tensor must have three axes")
    offsets = []
    for rank, entry in enumerate(entries):
        expected, offset = list(shape), [0] * len(shape)
        axes = [(0, 8, rank % 8), (1, 2, rank // 8)] if expert else [(0, 16, rank)]
        for axis, pieces, index in axes:
            offset[axis], expected[axis] = shard_extent(shape[axis], pieces, index)
        if entry["shape"] != expected:
            raise ValueError(f"unsupported placement for {name} rank {rank}: {entry['shape']}")
        offsets.append(tuple(offset))
    return ("expert_ep8_fsdp2" if expert else "row16"), offsets


class V41CheckpointInventory:
    def __init__(self, checkpoint):
        self.path = Path(checkpoint)
        self.marker = json.loads((self.path / "COMPLETE.json").read_text())
        m, c = self.marker, self.marker["contract"]
        if (m["format"] != "archlab-v41-full-sharded-v1" or m["world_size"] != 16
                or tuple(c[k] for k in ("world_size", "ep_size", "expert_fsdp_size", "engram_owners"))
                != (16, 8, 2, 16) or not c["all_parameters_unfrozen"]):
            raise ValueError("unsupported full-training checkpoint contract")
        self.manifests = []
        for rank in range(16):
            manifest = json.loads((self.path / f"rank-{rank:02d}/MANIFEST.json").read_text())
            if (manifest["rank"] != rank or manifest["world_size"] != 16
                    or manifest["cursor"] != m["cursor"] or manifest["contract"] != c):
                raise ValueError("rank manifest differs from COMPLETE")
            self.manifests.append(manifest)
        names = [entry["name"] for entry in self.manifests[0]["tensors"]]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tensor names")
        if any([t["name"] for t in r["tensors"]] != names for r in self.manifests):
            raise ValueError("rank tensor lists differ")
        self.entries = {
            name: [manifest["tensors"][i] for manifest in self.manifests]
            for i, name in enumerate(names)
        }
        self.layouts = {name: tensor_layout(entries) for name, entries in self.entries.items()}

    def report(self):
        counts = Counter(layout for layout, _ in self.layouts.values())
        sizes = Counter()
        for entries in self.entries.values():
            t = entries[0]
            sizes[t["dtype"]] += math.prod(t["global_shape"]) * _BYTES[t["dtype"]]
        return {
            "checkpoint": str(self.path.resolve()), "cursor": self.marker["cursor"],
            "variant": self.marker["contract"]["variant"], "tensor_count": len(self.entries),
            "layouts": dict(counts), "unique_tensor_bytes_by_dtype": dict(sizes),
            "unique_tensor_bytes": sum(sizes.values()),
            "metadata_validated": True, "all_payloads_verified": False,
            "engine_ready": False, "optimizer_included": False,
        }

    def read_tensor(self, name, *, max_bytes=512 * 2**20):
        """Verify and reconstruct one bounded tensor, useful for adapter probes.

        Large backbone/Engram tensors need a separate streaming exporter. The
        limit is on output tensor bytes; source chunks also consume memory.
        """
        import torch

        entries = self.entries[name]
        first = entries[0]
        shape, dtype = first["global_shape"], first["dtype"]
        if math.prod(shape) * _BYTES[dtype] > max_bytes:
            raise ValueError("tensor exceeds reconstruction limit; streaming export required")
        output = torch.empty(shape, dtype=getattr(torch, dtype.removeprefix("torch.")), device="cpu")
        kind, offsets = self.layouts[name]
        ranks = range(1) if kind == "replicated" else range(16)
        for rank in ranks:
            entry = entries[rank]
            local = torch.empty(entry["shape"], dtype=output.dtype, device="cpu")
            flat, position = local.view(-1), 0
            for chunk in entry["chunks"]:
                filename = chunk["file"]
                if Path(filename).name != filename:
                    raise ValueError("chunk filename must be a basename")
                value = torch.load(self.path / f"rank-{rank:02d}" / filename,
                                   map_location="cpu", weights_only=True)
                digest = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                if (value.ndim != 1 or value.dtype != output.dtype
                        or value.numel() != chunk["elements"] or digest != chunk["sha256"]):
                    raise ValueError(f"checkpoint payload mismatch: {name}")
                flat[position:position + value.numel()].copy_(value)
                position += value.numel()
            slices = tuple(slice(start, start + size)
                           for start, size in zip(offsets[rank], local.shape, strict=True))
            output[slices].copy_(local)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(V41CheckpointInventory(args.checkpoint).report(), indent=2))


if __name__ == "__main__":
    main()
