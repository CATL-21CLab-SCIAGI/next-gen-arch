"""Stream a full 16-rank fine-tune into unquantized released-name safetensors.

Every trained backbone tensor is read from the full checkpoint. The released
model supplies configuration/tokenizer assets only, never replacement weights.
Export is staged in a new directory and admitted by EXPORT_COMPLETE.json last.
Engine integration must load custom adapter keys and verify derived buffers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import tempfile
import time
from pathlib import Path

from archlab.serving.v41_checkpoint_inventory import V41CheckpointInventory

_DTYPE = {"torch.bfloat16": ("BF16", 2), "torch.float32": ("F32", 4), "torch.int64": ("I64", 8)}


def released_name(name):
    name = name.replace("._checkpoint_wrapped_module", "")
    if name.startswith("model.embed_tokens."):
        return "embed." + name.removeprefix("model.embed_tokens.")
    if name.startswith("lm_head."):
        return "head." + name.removeprefix("lm_head.")
    name = name.removeprefix("model.")
    name = name.replace(".attn.sinks_param.weight", ".attn.attn_sink")
    match = re.fullmatch(r"layers\.(\d+)\.(attn|ffn)_hc\.(fn|base|scale)", name)
    if match:
        return f"layers.{match[1]}.hc_{match[2]}_{match[3]}"
    for source, target in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
        name = re.sub(rf"(\.ffn\.(?:experts\.\d+|shared_experts))\.{source}\.",
                      rf"\1.{target}.", name)
    if name.endswith(".ffn.gate.e_score_correction_bias"):
        name = name.removesuffix(".e_score_correction_bias") + ".bias"
    return name


def verified_chunks(inventory, rank, entry):
    import torch

    dtype = getattr(torch, entry["dtype"].removeprefix("torch."))
    for chunk in entry["chunks"]:
        filename = chunk["file"]
        if Path(filename).name != filename:
            raise ValueError("chunk filename must be a basename")
        tensor = torch.load(inventory.path / f"rank-{rank:02d}" / filename,
                            map_location="cpu", weights_only=True)
        if tensor.dtype != dtype or tensor.ndim != 1 or tensor.numel() != chunk["elements"]:
            raise ValueError(f"chunk tensor contract mismatch: {entry['name']}")
        if hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest() != chunk["sha256"]:
            raise ValueError(f"chunk checksum mismatch: {entry['name']}")
        yield tensor


def write_tensor(path, name, shape, dtype, chunks):
    """Write one standard safetensors file sequentially (also suitable for OSS)."""
    import torch

    code, width = _DTYPE[dtype]
    expected = math.prod(shape) * width
    header = json.dumps({name: dict(dtype=code, shape=shape, data_offsets=[0, expected])},
                        separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    digest, count = hashlib.sha256(), 0
    with Path(path).open("xb") as stream:
        prefix = struct.pack("<Q", len(header)) + header
        stream.write(prefix)
        digest.update(prefix)
        for tensor in chunks:
            if str(tensor.dtype) != dtype:
                raise ValueError("export cannot change tensor dtype")
            data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            if count + len(data) > expected:
                raise ValueError("export payload exceeds declared shape")
            stream.write(data)
            digest.update(data)
            count += len(data)
        if count != expected:
            raise ValueError("export payload does not cover declared shape")
    return dict(sha256=digest.hexdigest(), tensor_bytes=count, file_bytes=len(prefix) + count)


def row_chunks(inventory, name, elements):
    kind, _ = inventory.layouts[name]
    if kind not in ("row16", "replicated"):
        raise ValueError("not a row/replicated tensor")
    remaining = elements
    ranks = range(1) if kind == "replicated" else range(16)
    for rank in ranks:
        for tensor in verified_chunks(inventory, rank, inventory.entries[name][rank]):
            count = min(remaining, tensor.numel())
            if count:
                yield tensor[:count]
                remaining -= count
            # Still read and verify padding payloads after logical rows end.
    if remaining:
        raise ValueError("checkpoint has fewer elements than the logical tensor")


def bounded_parts(chunks, limit):
    """Split a tensor stream without concatenating a table into host RAM."""
    offset, buffered, count = 0, [], 0
    for chunk in chunks:
        first = 0
        while first < chunk.numel():
            take = min(limit - count, chunk.numel() - first)
            buffered.append(chunk[first:first + take])
            first += take
            count += take
            if count == limit:
                yield offset, count, buffered
                offset += count
                buffered, count = [], 0
    if count:
        yield offset, count, buffered


def expert_tensors(inventory, name, scratch):
    """One disk-backed stacked tensor; at most one local rank shard in RAM."""
    import torch

    entries = inventory.entries[name]
    shape = entries[0]["global_shape"]
    _, offsets = inventory.layouts[name]
    dtype = getattr(torch, entries[0]["dtype"].removeprefix("torch."))
    with tempfile.TemporaryDirectory(prefix="v41-expert-", dir=scratch) as directory:
        mapped = torch.from_file(str(Path(directory) / "tensor.bin"), shared=True,
                                 size=math.prod(shape), dtype=dtype).reshape(shape)
        for rank, entry in enumerate(entries):
            if math.prod(entry["shape"]) * _DTYPE[entry["dtype"]][1] > 2 * 2**30:
                raise ValueError("expert local shard exceeds the 2 GiB staging limit")
            local = torch.empty(entry["shape"], dtype=dtype)
            position = 0
            for chunk in verified_chunks(inventory, rank, entry):
                local.view(-1)[position:position + chunk.numel()].copy_(chunk)
                position += chunk.numel()
            slices = tuple(slice(first, first + size)
                           for first, size in zip(offsets[rank], local.shape, strict=True))
            mapped[slices].copy_(local)
            del local
        base = released_name(name).rsplit(".", 1)[0]
        for expert in range(shape[0]):
            if name.endswith("gate_and_up_projs"):
                if shape[2] % 2:
                    raise ValueError("gate/up dimension must split evenly")
                middle = shape[2] // 2
                yield f"{base}.{expert}.w1.weight", mapped[expert, :, :middle].T.contiguous()
                yield f"{base}.{expert}.w3.weight", mapped[expert, :, middle:].T.contiguous()
            else:
                yield f"{base}.{expert}.w2.weight", mapped[expert].T.contiguous()
        del mapped


def write_experts(path, inventory, name, scratch):
    """Bundle a stacked expert parameter into one sequential OSS object."""
    import torch

    entry = inventory.entries[name][0]
    experts, middle, width = entry["global_shape"]
    dtype = entry["dtype"]
    code, itemsize = _DTYPE[dtype]
    base = released_name(name).rsplit(".", 1)[0]
    specifications = []
    for expert in range(experts):
        if name.endswith("gate_and_up_projs"):
            if width % 2:
                raise ValueError("gate/up dimension must split evenly")
            specifications.extend((f"{base}.{expert}.w{projection}.weight", [width // 2, middle])
                                  for projection in (1, 3))
        else:
            specifications.append((f"{base}.{expert}.w2.weight", [width, middle]))
    header, offset = {}, 0
    for key, shape in specifications:
        end = offset + math.prod(shape) * itemsize
        header[key] = dict(dtype=code, shape=shape, data_offsets=[offset, end])
        offset = end
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    digest = hashlib.sha256()
    prefix = struct.pack("<Q", len(raw)) + raw
    count = 0
    with Path(path).open("xb") as stream:
        stream.write(prefix)
        digest.update(prefix)
        for (key, shape), (actual_key, tensor) in zip(
            specifications, expert_tensors(inventory, name, scratch), strict=True
        ):
            if actual_key != key or list(tensor.shape) != shape or str(tensor.dtype) != dtype:
                raise ValueError("expert export order/shape/dtype mismatch")
            data = tensor.view(torch.uint8).numpy().tobytes()
            stream.write(data)
            digest.update(data)
            count += len(data)
    if count != offset:
        raise ValueError("expert payload does not cover declared shapes")
    return specifications, dict(sha256=digest.hexdigest(), file_bytes=len(prefix) + count)


def export_checkpoint(checkpoint, assets, output, scratch, *, engram_part_bytes=1024**3):
    inventory = V41CheckpointInventory(checkpoint)
    assets, output, scratch = Path(assets), Path(output), Path(scratch)
    config = json.loads((assets / "config.json").read_text())
    text = config["text_config"]
    engram_rows = dict(zip(text["engram_layer_ids"], text["engram_num_embeddings"], strict=True))
    output.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=True)
    weight_map, receipts, buffers = {}, [], {}
    started = time.monotonic()

    def save(name, shape, dtype, chunks, source):
        if name in weight_map or name in buffers:
            raise ValueError(f"duplicate export tensor: {name}")
        filename = f"tensor-{len(receipts):06d}.safetensors"
        if name.startswith("engram_hash."):
            (output / "archlab-buffers").mkdir(exist_ok=True)
            filename = "archlab-buffers/" + filename
        receipt = write_tensor(output / filename, name, shape, dtype, chunks)
        receipt.update(name=name, source=source, file=filename, shape=shape, dtype=dtype)
        receipts.append(receipt)
        if name.startswith("engram_hash."):
            buffers[name] = filename
        else:
            weight_map[name] = filename

    for name, entries in inventory.entries.items():
        entry = entries[0]
        if inventory.layouts[name][0] == "expert_ep8_fsdp2":
            filename = f"tensor-{len(receipts):06d}.safetensors"
            specs, file_receipt = write_experts(output / filename, inventory, name, scratch)
            for key, shape in specs:
                if key in weight_map:
                    raise ValueError(f"duplicate expert key: {key}")
                weight_map[key] = filename
                receipts.append(dict(**file_receipt, name=key, shape=shape, dtype=entry["dtype"],
                                     tensor_bytes=math.prod(shape) * _DTYPE[entry["dtype"]][1],
                                     file=filename, source=name))
        else:
            key, shape = released_name(name), list(entry["global_shape"])
            match = re.fullmatch(r"layers\.(\d+)\.engram\.embed\.weight", key)
            if match:
                logical = engram_rows[int(match[1])]
                if not 0 <= shape[0] - logical < 16:
                    raise ValueError("unexpected Engram owner padding")
                shape[0] = logical
            if match and math.prod(shape) * _DTYPE[entry["dtype"]][1] > engram_part_bytes:
                row_elements = shape[1]
                rows_per_part = max(1, engram_part_bytes // (_DTYPE[entry["dtype"]][1] * row_elements))
                chunks = row_chunks(inventory, name, math.prod(shape))
                for offset, count, pieces in bounded_parts(chunks, rows_per_part * row_elements):
                    if offset % row_elements or count % row_elements:
                        raise ValueError("Engram shard does not align with complete rows")
                    save(f"{key}.rows.{offset // row_elements}", [count // row_elements, row_elements],
                         entry["dtype"], pieces, name)
            else:
                save(key, shape, entry["dtype"], row_chunks(inventory, name, math.prod(shape)), name)
        print(json.dumps(dict(event="tensor_exported", source=name, tensors=len(receipts),
                              seconds=time.monotonic() - started)), flush=True)
    # The external engine extension must explicitly register this architecture.
    config["architectures"] = ["ArchlabDeepseekV41ForCausalLM"]
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    text.pop("quantization_config", None)
    text["num_nextn_predict_layers"] = 0  # Fine-tune contains no trained draft model.
    config["vision_config"]["num_hidden_layers"] = 0
    config["archlab"] = dict(variant=inventory.marker["contract"]["variant"],
                              checkpoint_cursor=inventory.marker["cursor"],
                              adapter_layers=[4, 9, 14, 19, 24, 29, 34, 39],
                              derived_buffer_files=buffers, engram_row_shards=True, cpu_offload=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for filename in ("tokenizer.json", "tokenizer_config.json", "LICENSE"):
        if (assets / filename).exists():
            shutil.copyfile(assets / filename, output / filename)
    if (assets / "encoding").exists():
        shutil.copytree(assets / "encoding", output / "encoding")
    index = dict(metadata=dict(total_size=sum(r["tensor_bytes"] for r in receipts
                                             if r["name"] in weight_map)), weight_map=weight_map)
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    with (output / "export-manifest.jsonl").open("x") as stream:
        for receipt in receipts:
            stream.write(json.dumps(receipt) + "\n")
    result = dict(source=inventory.report(), tensors=len(receipts),
                  exported_bytes=sum(r["tensor_bytes"] for r in receipts),
                  all_source_tensor_names_exported=True, payload_checksums_verified=True,
                  requantized=False, optimizer_exported=False, engine_qualified=False,
                  seconds=time.monotonic() - started)
    (output / "EXPORT_COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "assets", "output", "scratch"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.assets, args.output, args.scratch)))


if __name__ == "__main__":
    main()
