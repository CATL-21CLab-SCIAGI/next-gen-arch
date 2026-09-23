"""Stream full trained checkpoint values directly into SGLang's loader layout.

This avoids writing a second 1.5-TB model just to change its serialization.
Only startup uses bounded CPU staging. Experts are reconstructed for the local
EP owner; Engram reads only rows owned by this TP rank. Dense tensors retain
released names so the engine's native TP weight loaders place them normally.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

from archlab.serving.v41_checkpoint_inventory import V41CheckpointInventory
from archlab.serving.v41_export import released_name, verified_chunks


def local_tensor(inventory, name, rank):
    import torch

    entry = inventory.entries[name][rank]
    dtype = getattr(torch, entry["dtype"].removeprefix("torch."))
    if math.prod(entry["shape"]) * torch.empty((), dtype=dtype, device="cpu").element_size() > 2 * 2**30:
        raise ValueError("local expert shard exceeds the bounded staging limit")
    value = torch.empty(entry["shape"], dtype=dtype, device="cpu")
    position = 0
    for part in verified_chunks(inventory, rank, entry):
        value.view(-1)[position:position + part.numel()].copy_(part)
        position += part.numel()
    if position != value.numel():
        raise ValueError("local checkpoint shard is incomplete")
    return value


def owned_expert_weights(inventory, name, ep_rank):
    import torch

    if not 0 <= ep_rank < 8 or inventory.layouts[name][0] != "expert_ep8_fsdp2":
        raise ValueError("expected a qualified EP8 expert tensor")
    global_shape = inventory.entries[name][0]["global_shape"]
    first = local_tensor(inventory, name, ep_rank)
    shape = (first.shape[0], global_shape[1], global_shape[2])
    result = torch.empty(shape, dtype=first.dtype, device="cpu")
    split = first.shape[1]
    result[:, :split].copy_(first)
    del first
    second = local_tensor(inventory, name, ep_rank + 8)
    result[:, split:].copy_(second)
    del second
    base = released_name(name).rsplit(".", 1)[0]
    for local in range(shape[0]):
        expert = ep_rank * shape[0] + local
        if name.endswith("gate_and_up_projs"):
            middle = shape[2] // 2
            if middle * 2 != shape[2]:
                raise ValueError("gate/up dimension must split evenly")
            yield f"{base}.{expert}.w1.weight", result[local, :, :middle].T.contiguous()
            yield f"{base}.{expert}.w3.weight", result[local, :, middle:].T.contiguous()
        else:
            yield f"{base}.{expert}.w2.weight", result[local].T.contiguous()


def owned_engram_rows(inventory, name, *, logical_rows, tp_rank):
    entry = inventory.entries[name][0]
    if (not 0 <= tp_rank < 8 or len(entry["global_shape"]) != 2
            or not 0 <= entry["global_shape"][0] - logical_rows < 16
            or inventory.layouts[name][0] != "row16"):
        raise ValueError("unsupported Engram row placement")
    columns = entry["global_shape"][1]
    first = logical_rows * tp_rank // 8
    last = logical_rows * (tp_rank + 1) // 8
    covered = 0
    for rank, source in enumerate(inventory.entries[name]):
        start = inventory.layouts[name][1][rank][0] * columns
        for chunk in source["chunks"]:
            end = start + chunk["elements"]
            lo, hi = max(start, first * columns), min(end, last * columns)
            if lo < hi:
                if lo % columns or hi % columns:
                    raise ValueError("Engram payload chunks do not align to rows")
                packet = {**source, "chunks": [chunk]}
                value = next(verified_chunks(inventory, rank, packet))
                selected = value[lo - start:hi - start].view(-1, columns)
                yield lo // columns, selected
                covered += selected.shape[0]
            start = end
    if covered != last - first:
        raise ValueError("Engram row reader lost or duplicated rows")


def iter_weights(inventory, *, tp_rank, engram_rows):
    for name in inventory.entries:
        key = released_name(name)
        if key.startswith("engram_hash."):
            continue  # These derived buffers are separately checked against the engine.
        match = re.fullmatch(r"layers\.(\d+)\.engram\.embed\.weight", key)
        if match:
            for start, value in owned_engram_rows(inventory, name,
                                                  logical_rows=engram_rows[int(match[1])],
                                                  tp_rank=tp_rank):
                yield f"{key}.rows.{start}", value
        elif inventory.layouts[name][0] == "expert_ep8_fsdp2":
            yield from owned_expert_weights(inventory, name, tp_rank)
        else:
            yield key, inventory.read_tensor(name, max_bytes=4 * 2**30)
        if tp_rank == 0:
            print(json.dumps(dict(event="direct_checkpoint_tensor_loaded", name=name)), flush=True)


def prepare_model(checkpoint, assets, output):
    import torch
    from safetensors.torch import save_file

    inventory = V41CheckpointInventory(checkpoint)
    assets, output = Path(assets), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((assets / "config.json").read_text())
    config["architectures"] = ["ArchlabDeepseekV41ForCausalLM"]
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    config["text_config"].pop("quantization_config", None)
    config["text_config"]["num_nextn_predict_layers"] = 0
    config["vision_config"]["num_hidden_layers"] = 0
    buffers = {}
    (output / "archlab-buffers").mkdir()
    for name in inventory.entries:
        key = released_name(name)
        if key.startswith("engram_hash."):
            filename = f"archlab-buffers/{key}.safetensors"
            save_file({key: inventory.read_tensor(name)}, output / filename)
            buffers[key] = filename
    cursor = inventory.marker["cursor"]
    identity = hashlib.sha256((inventory.path / "COMPLETE.json").read_bytes()).hexdigest()
    config["archlab"] = dict(variant=inventory.marker["contract"]["variant"],
                              checkpoint_cursor=cursor, adapter_layers=[4, 9, 14, 19, 24, 29, 34, 39],
                              derived_buffer_files=buffers, engram_row_shards=True,
                              full_checkpoint=str(inventory.path.resolve()),
                              complete_sha256=identity, cpu_offload=False)
    marker = torch.tensor([cursor["step"], cursor["supervised_tokens"]], dtype=torch.int64, device="cpu")
    save_file({"archlab_full_checkpoint_pointer": marker}, output / "loader-pointer.safetensors")
    (output / "model.safetensors.index.json").write_text(json.dumps(dict(
        metadata=dict(total_size=marker.numel() * marker.element_size()),
        weight_map={"archlab_full_checkpoint_pointer": "loader-pointer.safetensors"})) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for filename in ("tokenizer.json", "tokenizer_config.json", "LICENSE"):
        if (assets / filename).exists():
            shutil.copyfile(assets / filename, output / filename)
    if (assets / "encoding").exists():
        shutil.copytree(assets / "encoding", output / "encoding")
    report = dict(format="archlab-v41-direct-load-v1", source=inventory.report(),
                  complete_sha256=identity, weights_copied=False, engine_qualified=False)
    (output / "DIRECT_CHECKPOINT.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "README.txt").write_text(
        "Metadata-only model directory. The pointer file is NOT model weights.\n"
        "Use archlab.serving.sglang_v41_launch; its external model implementation\n"
        "loads every trained parameter from the referenced full checkpoint.\n")
    return report
