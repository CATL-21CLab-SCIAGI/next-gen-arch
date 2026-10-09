import json

import torch
from safetensors.torch import load_file
from test_v41_checkpoint_inventory import checkpoint

from archlab.serving.v41_checkpoint_inventory import V41CheckpointInventory
from archlab.serving.v41_direct_checkpoint import (
    iter_weights,
    owned_engram_rows,
    owned_expert_weights,
    prepare_model,
)


def test_direct_ep_owners_reconstruct_every_expert_once(tmp_path):
    values = checkpoint(tmp_path)
    inventory = V41CheckpointInventory(tmp_path)
    name = "model.layers.0.ffn.experts.down_projs"
    found = {}
    for rank in range(8):
        items = dict(owned_expert_weights(inventory, name, rank))
        assert len(items) == 1 and not found.keys() & items.keys()
        found.update(items)
    for expert in range(8):
        torch.testing.assert_close(found[f"layers.0.ffn.experts.{expert}.w2.weight"],
                                   values[name][expert].T, rtol=0, atol=0)
    streamed = dict(iter_weights(inventory, tp_rank=0, engram_rows={}))
    torch.testing.assert_close(streamed["small"], values["model.small"], rtol=0, atol=0)
    assert len([key for key in streamed if ".experts." in key]) == 1


def test_direct_engram_repartitions_uneven_rows_without_padding(tmp_path):
    checkpoint(tmp_path)
    name = "model.layers.1.engram.embed.weight"
    for rank in range(16):
        path = tmp_path / f"rank-{rank:02d}/MANIFEST.json"
        manifest = json.loads(path.read_text())
        entry = manifest["tensors"][0]
        entry["name"] = name
        entry["shape"].append(1)
        entry["global_shape"].append(1)
        path.write_text(json.dumps(manifest))
    inventory = V41CheckpointInventory(tmp_path)
    collected = []
    for rank in range(8):
        pieces = list(owned_engram_rows(inventory, name, logical_rows=2, tp_rank=rank))
        for first, tensor in pieces:
            collected.append((first, tensor.clone()))
    assert [first for first, _ in collected] == [0, 1]
    torch.testing.assert_close(torch.cat([t for _, t in collected]), torch.tensor([[0.], [1.]]),
                               rtol=0, atol=0)


def test_metadata_pointer_cannot_be_mistaken_for_exported_weights(tmp_path):
    source, assets, output = (tmp_path / name for name in ("source", "assets", "output"))
    source.mkdir()
    assets.mkdir()
    checkpoint(source)
    (assets / "config.json").write_text(json.dumps(dict(
        quantization_config=dict(quant_method="fp8"), vision_config=dict(num_hidden_layers=24),
        text_config=dict(engram_layer_ids=[], engram_num_embeddings=[]))))
    report = prepare_model(source, assets, output)
    assert not report["weights_copied"] and not report["engine_qualified"]
    assert not (output / "EXPORT_COMPLETE.json").exists()
    pointer = load_file(output / "loader-pointer.safetensors")
    assert pointer["archlab_full_checkpoint_pointer"].tolist() == [4537, 756364650]
    config = json.loads((output / "config.json").read_text())
    assert config["archlab"]["full_checkpoint"] == str(source.resolve())
    assert "quantization_config" not in config


def test_loading_stays_on_cpu_under_an_ambient_meta_device(tmp_path):
    values = checkpoint(tmp_path)
    inventory = V41CheckpointInventory(tmp_path)
    with torch.device("meta"):
        value = inventory.read_tensor("model.small")
        expert = next(owned_expert_weights(inventory, "model.layers.0.ffn.experts.down_projs", 0))[1]
    assert value.device.type == expert.device.type == "cpu"
    torch.testing.assert_close(value, values["model.small"], rtol=0, atol=0)
