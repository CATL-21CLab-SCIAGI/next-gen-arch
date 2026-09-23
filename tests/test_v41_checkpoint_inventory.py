import hashlib
import json
import math

import pytest
import torch

from archlab.serving.v41_checkpoint_inventory import V41CheckpointInventory, shard_extent


def checkpoint(tmp_path):
    values = {
        "model.small": torch.arange(3, dtype=torch.float32),
        "model.layers.0.ffn.experts.down_projs": torch.arange(8 * 6 * 2, dtype=torch.bfloat16).reshape(8, 6, 2),
        "model.replicated": torch.arange(4, dtype=torch.int64),
    }
    contract = dict(world_size=16, ep_size=8, expert_fsdp_size=2, engram_owners=16,
                    all_parameters_unfrozen=True, variant="normal")
    cursor = dict(step=4537, supervised_tokens=756364650)
    marker = dict(format="archlab-v41-full-sharded-v1", world_size=16,
                  contract=contract, cursor=cursor)
    (tmp_path / "COMPLETE.json").write_text(json.dumps(marker))
    for rank in range(16):
        directory = tmp_path / f"rank-{rank:02d}"
        directory.mkdir()
        tensors = []
        for number, (name, value) in enumerate(values.items()):
            if "experts" in name:
                local = value[rank % 8:rank % 8 + 1, rank // 8 * 3:rank // 8 * 3 + 3]
            elif "replicated" in name:
                local = value
            else:
                start, size = shard_extent(3, 16, rank)
                local = value[start:start + size]
            chunks = []
            for part, tensor in enumerate(local.contiguous().reshape(-1).split(5)):
                if not tensor.numel():
                    continue
                filename = f"{number}-{part}.pt"
                torch.save(tensor.clone(), directory / filename)
                digest = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
                chunks.append(dict(file=filename, elements=tensor.numel(), sha256=digest))
            tensors.append(dict(name=name, shape=list(local.shape), global_shape=list(value.shape),
                                dtype=str(value.dtype), chunks=chunks))
        (directory / "MANIFEST.json").write_text(json.dumps(dict(
            rank=rank, world_size=16, cursor=cursor, contract=contract, tensors=tensors)))
    return values


def test_reconstruction_handles_expert_axes_empty_shards_and_replicas(tmp_path):
    values = checkpoint(tmp_path)
    inventory = V41CheckpointInventory(tmp_path)
    for name, value in values.items():
        torch.testing.assert_close(inventory.read_tensor(name), value, rtol=0, atol=0)
    report = inventory.report()
    assert report["unique_tensor_bytes"] == sum(v.numel() * v.element_size() for v in values.values())
    assert report["layouts"] == dict(row16=1, expert_ep8_fsdp2=1, replicated=1)
    assert not report["engine_ready"] and not report["all_payloads_verified"]
    with pytest.raises(ValueError, match="limit"):
        inventory.read_tensor("model.small", max_bytes=1)


def test_corrupt_payload_is_rejected(tmp_path):
    checkpoint(tmp_path)
    torch.save(torch.tensor([99.0]), tmp_path / "rank-00/0-0.pt")
    inventory = V41CheckpointInventory(tmp_path)
    with pytest.raises(ValueError, match="payload mismatch"):
        inventory.read_tensor("model.small")


@pytest.mark.parametrize("corruption", ["replica", "shape", "cursor"])
def test_inconsistent_metadata_is_rejected(tmp_path, corruption):
    checkpoint(tmp_path)
    path = tmp_path / "rank-03/MANIFEST.json"
    manifest = json.loads(path.read_text())
    if corruption == "replica":
        manifest["tensors"][2]["chunks"][0]["sha256"] = "0" * 64
    elif corruption == "shape":
        manifest["tensors"][1]["shape"] = [3, 1, 2]
        assert math.prod(manifest["tensors"][1]["shape"]) == 6
    else:
        manifest["cursor"]["step"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        V41CheckpointInventory(tmp_path)
