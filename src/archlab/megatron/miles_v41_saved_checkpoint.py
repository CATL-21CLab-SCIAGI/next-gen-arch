"""Bounded native readback of an explicitly completed Miles V4.1 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from archlab.artifacts import atomic_write_json


def verify(root, iteration):
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint = root / "checkpoints" / f"iter_{iteration:07d}"
    if (root / "checkpoints/latest_checkpointed_iteration.txt").read_text().strip() != str(
        iteration
    ):
        raise ValueError("native checkpoint completion pointer differs")
    if not (checkpoint / "metadata.json").is_file():
        raise ValueError("native checkpoint metadata is missing")
    metadata = FileSystemReader(checkpoint).read_metadata()
    keys = list(metadata.state_dict_metadata)
    components = {
        name: [key for key in keys if token in key]
        for name, token in (
            ("attention_adapters", "archlab_adapter"),
            ("engram_tables", "v41_engram.table"),
            ("experts", "experts"),
            ("embedding", "word_embeddings"),
            ("head", "output_layer"),
        )
    }
    if not all(components.values()) or len(components["engram_tables"]) != 2:
        raise ValueError("checkpoint model components are incomplete")
    expected_layers = set(
        json.loads((root / "model/config.json").read_text())["archlab"]["adapter_layers"]
    )
    if {int(key.split(".")[2]) for key in components["attention_adapters"]} != expected_layers:
        raise ValueError("saved adapter layer inventory differs")
    sizes = {}
    for info in metadata.storage_data.values():
        path = checkpoint / info.relative_path
        if path not in sizes:
            sizes[path] = path.stat().st_size
        if info.offset + info.length > sizes[path]:
            raise ValueError(f"truncated model payload: {path}")
    samples = [
        key
        for key in components["attention_adapters"]
        if hasattr(metadata.state_dict_metadata[key], "size")
        and math.prod(metadata.state_dict_metadata[key].size) <= 1_000_000
    ][:4]
    if len(samples) != 4:
        raise ValueError("missing bounded adapter samples")
    loaded = {
        key: torch.empty(
            metadata.state_dict_metadata[key].size,
            dtype=metadata.state_dict_metadata[key].properties.dtype,
        )
        for key in samples
    }
    dcp.load(
        loaded,
        storage_reader=FileSystemReader(checkpoint),
        planner=dcp.DefaultLoadPlanner(flatten_state_dict=False),
        no_dist=True,
    )
    if not all(torch.isfinite(value).all() for value in loaded.values()):
        raise ValueError("nonfinite saved adapter sample")
    manifests = sorted((checkpoint / "nvme_opt_state").glob("rank*/opt*/manifest.json"))
    if len(manifests) != 64:
        raise ValueError("expected all 64 optimizer manifests")
    total_bytes = bucket_files = nonzero = sampled = 0
    for path in manifests:
        manifest = json.loads(path.read_text())
        if manifest["dtypes"] != {
            "main": "torch.float32",
            "exp_avg": "torch.bfloat16",
            "exp_avg_sq": "torch.bfloat16",
        }:
            raise ValueError("saved optimizer precision differs")
        for index, bucket in enumerate(manifest["buckets"]):
            if not all(step == iteration + 1 for step in bucket["steps"]):
                raise ValueError("saved optimizer steps differ from completed updates")
            lengths = [
                sum((n * size + 4095) // 4096 * 4096 for n in bucket["entry_numels"])
                for size in (4, 2, 2)
            ]
            payload = path.parent / bucket["file"]
            size = payload.stat().st_size
            if size != sum(lengths):
                raise ValueError(f"saved optimizer payload length differs: {payload}")
            total_bytes += size
            bucket_files += 1
            if index == 0:
                with payload.open("rb") as stream:
                    for offset, dtype, element_bytes in (
                        (0, torch.float32, 4),
                        (lengths[0], torch.bfloat16, 2),
                        (lengths[0] + lengths[1], torch.bfloat16, 2),
                    ):
                        stream.seek(offset)
                        tensor = torch.frombuffer(
                            bytearray(
                                stream.read(min(bucket["entry_numels"][0], 1024) * element_bytes)
                            ),
                            dtype=dtype,
                        ).float()
                        if not torch.isfinite(tensor).all():
                            raise ValueError("nonfinite saved optimizer sample")
                        if offset == lengths[0]:
                            nonzero += int(torch.count_nonzero(tensor))
                            sampled += tensor.numel()
    if nonzero == 0:
        raise ValueError("saved first-moment samples are all zero")
    residents = sorted(
        (checkpoint / "nvme_opt_state").glob("rank*/opt*/fp32_resident_optimizer.pt")
    )
    if len(residents) != 32:
        raise ValueError("expected all 32 resident optimizer files")
    for path in residents:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not state["state"]:
            raise ValueError("empty resident optimizer state")
        for entry in state["state"].values():
            for value in entry.values():
                if (
                    isinstance(value, torch.Tensor)
                    and not torch.isfinite(value.reshape(-1)[:1024]).all()
                ):
                    raise ValueError("nonfinite resident optimizer sample")
    return {
        "status": "passed",
        "checkpoint": str(checkpoint),
        "iteration": iteration,
        "completed_optimizer_updates": iteration + 1,
        "model_files": len(sizes),
        "model_bytes": sum(sizes.values()),
        "model_component_counts": {key: len(value) for key, value in components.items()},
        "native_adapter_readback": {key: value.numel() for key, value in loaded.items()},
        "optimizer_manifests": len(manifests),
        "optimizer_bucket_files": bucket_files,
        "optimizer_bytes": total_bytes,
        "resident_optimizer_files": len(residents),
        "sampled_first_moments": sampled,
        "nonzero_sampled_first_moments": nonzero,
        "full_training_resume_demonstrated": False,
        "verification": "all native model extents and optimizer lengths/steps; bounded finite tensor readback",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_root.resolve(), args.iteration)
    atomic_write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
