import json

import pytest
import torch
from safetensors.torch import load_file
from test_v41_checkpoint_inventory import checkpoint

from archlab.serving.v41_checkpoint_inventory import V41CheckpointInventory
from archlab.serving.v41_export import (
    bounded_parts,
    expert_tensors,
    export_checkpoint,
    released_name,
    write_experts,
    write_tensor,
)


def test_expert_unshard_and_transpose_are_exact(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    values = checkpoint(source)
    name = "model.layers.0.ffn.experts.down_projs"
    inventory = V41CheckpointInventory(source)
    outputs = dict(expert_tensors(inventory, name, tmp_path))
    for expert in range(8):
        torch.testing.assert_close(outputs[f"layers.0.ffn.experts.{expert}.w2.weight"],
                                   values[name][expert].T, rtol=0, atol=0)
    # Reuse the same fixture layout with a gate/up name to test both halves.
    for rank in range(16):
        path = source / f"rank-{rank:02d}/MANIFEST.json"
        manifest = json.loads(path.read_text())
        manifest["tensors"][1]["name"] = name.replace("down_projs", "gate_and_up_projs")
        path.write_text(json.dumps(manifest))
    outputs = dict(expert_tensors(V41CheckpointInventory(source),
                                 name.replace("down_projs", "gate_and_up_projs"), tmp_path))
    for expert in range(8):
        for part, projection in enumerate((1, 3)):
            torch.testing.assert_close(outputs[f"layers.0.ffn.experts.{expert}.w{projection}.weight"],
                                       values[name][expert, :, part:part + 1].T, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64])
def test_sequential_safetensors_uses_standard_reader(tmp_path, dtype):
    tensor = torch.arange(30).reshape(6, 5).to(dtype)
    path = tmp_path / "tensor.safetensors"
    receipt = write_tensor(path, "weight", [6, 5], str(dtype), tensor.reshape(-1).split(7))
    torch.testing.assert_close(load_file(path)["weight"], tensor, rtol=0, atol=0)
    assert receipt["tensor_bytes"] == tensor.numel() * tensor.element_size()
    with pytest.raises(FileExistsError):
        write_tensor(path, "weight", [6, 5], str(dtype), [tensor])


def test_export_preserves_all_trained_tensors_and_disables_base_quantization(tmp_path):
    source, assets, output = (tmp_path / name for name in ("source", "assets", "output"))
    source.mkdir()
    assets.mkdir()
    values = checkpoint(source)
    (assets / "config.json").write_text(json.dumps(dict(
        quantization_config=dict(quant_method="fp8"), vision_config=dict(num_hidden_layers=24),
        text_config=dict(engram_layer_ids=[], engram_num_embeddings=[], num_nextn_predict_layers=3))))
    report = export_checkpoint(source, assets, output, tmp_path / "scratch")
    config = json.loads((output / "config.json").read_text())
    assert "quantization_config" not in config
    assert config["text_config"]["num_nextn_predict_layers"] == 0
    assert config["architectures"] == ["ArchlabDeepseekV41ForCausalLM"]
    assert report["tensors"] == 10 and report["payload_checksums_verified"]
    assert not report["engine_qualified"]
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    for name in ("model.small", "model.replicated"):
        key = released_name(name)
        torch.testing.assert_close(load_file(output / index[key])[key], values[name], rtol=0, atol=0)


def test_key_translation_keeps_adapter_and_hc_boundaries():
    prefix = "model.layers.4._checkpoint_wrapped_module."
    assert released_name(prefix + "attn_hc.fn") == "layers.4.hc_attn_fn"
    assert released_name(prefix + "attn_hc.simplicial_adapter.k.weight") == (
        "layers.4.attn_hc.simplicial_adapter.k.weight")
    assert released_name(prefix + "ffn.shared_experts.gate_proj.weight") == (
        "layers.4.ffn.shared_experts.w1.weight")


@pytest.mark.parametrize("part_bytes", [1024**3, 4])
def test_export_removes_only_engram_padding(tmp_path, part_bytes):
    source, assets, output = (tmp_path / name for name in ("source", "assets", "output"))
    source.mkdir()
    assets.mkdir()
    checkpoint(source)
    for rank in range(16):
        path = source / f"rank-{rank:02d}/MANIFEST.json"
        manifest = json.loads(path.read_text())
        entry = manifest["tensors"][0]
        entry["name"] = "model.layers.1.engram.embed.weight"
        entry["global_shape"].append(1)
        entry["shape"].append(1)
        path.write_text(json.dumps(manifest))
    (assets / "config.json").write_text(json.dumps(dict(
        vision_config=dict(num_hidden_layers=24),
        text_config=dict(engram_layer_ids=[1], engram_num_embeddings=[2]))))
    export_checkpoint(source, assets, output, tmp_path / "scratch", engram_part_bytes=part_bytes)
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    key = "layers.1.engram.embed.weight"
    if key in index:
        actual = load_file(output / index[key])[key]
    else:
        shard_keys = sorted((name for name in index if name.startswith(key + ".rows.")),
                            key=lambda name: int(name.rsplit(".", 1)[1]))
        actual = torch.cat([load_file(output / index[name])[name] for name in shard_keys])
        assert len(shard_keys) == 2
    torch.testing.assert_close(actual, torch.tensor([[0.], [1.]]), rtol=0, atol=0)


def test_bundled_experts_are_standard_safetensors(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    values = checkpoint(source)
    name = "model.layers.0.ffn.experts.down_projs"
    path = tmp_path / "experts.safetensors"
    specs, receipt = write_experts(path, V41CheckpointInventory(source), name, tmp_path)
    loaded = load_file(path)
    assert len(loaded) == len(specs) == 8
    assert receipt["file_bytes"] == path.stat().st_size
    for expert in range(8):
        torch.testing.assert_close(loaded[f"layers.0.ffn.experts.{expert}.w2.weight"],
                                   values[name][expert].T, rtol=0, atol=0)


def test_row_stream_parts_are_bounded_and_lossless():
    value = torch.arange(103).bfloat16()
    parts = list(bounded_parts(value.split(7), 16))
    restored = []
    offset = 0
    for start, size, tensors in parts:
        assert start == offset and 0 < size <= 16
        restored.extend(tensors)
        offset += size
    assert offset == 103
    torch.testing.assert_close(torch.cat(restored), value, rtol=0, atol=0)
