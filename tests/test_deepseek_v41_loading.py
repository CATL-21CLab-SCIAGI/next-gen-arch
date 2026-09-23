import json

import pytest
import torch
from safetensors.torch import save_file

from archlab.automodel.deepseek_v41_loading import load_native_ep_checkpoint


@pytest.mark.parametrize("name", ["head", "layers.2.attn.compressor.wkv", "layers.2.attn.compressor.wgate"])
def test_only_reference_fp32_promotions_are_allowed(tmp_path, name):
    model = torch.nn.Module()
    parent = model
    for part in name.split("."):
        child = torch.nn.Module()
        parent.add_module(part, child)
        parent = child
    parent.register_parameter("weight", torch.nn.Parameter(torch.empty(32, 32), requires_grad=False))
    source = torch.randn(32, 32).bfloat16()
    save_file({f"{name}.weight": source}, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {f"{name}.weight": "weights.safetensors"}}))
    # A non-CPU ambient default must not redirect safetensors allocations.
    with torch.device("meta"):
        result = load_native_ep_checkpoint(model, tmp_path, ep_rank=0, ep_size=1, require_verified_cache=False)
        assert torch.empty(0).device.type == "meta"
    torch.testing.assert_close(parent.weight, source.float(), rtol=0, atol=0)
    assert result["conversions"] == [{"name": f"{name}.weight", "conversion": "exact-BF16-to-FP32"}]


def test_unlisted_dtype_conversion_still_fails_closed(tmp_path):
    model = torch.nn.Module()
    model.register_parameter("unknown", torch.nn.Parameter(torch.empty(32, 32), requires_grad=False))
    save_file({"unknown": torch.ones(32, 32, dtype=torch.bfloat16)}, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"unknown": "weights.safetensors"}}))
    with pytest.raises(ValueError, match="unapproved dtype conversion"):
        load_native_ep_checkpoint(model, tmp_path, ep_rank=0, ep_size=1, require_verified_cache=False)


@pytest.mark.parametrize("owner_rank,owner_size", [(2, 4), (3, 4), (15, 16), (None, None)])
def test_engram_owners_are_independent_of_expert_owners(tmp_path, owner_rank, owner_size):
    model = torch.nn.Module()
    layer = torch.nn.Module()
    model.layers = torch.nn.ModuleList([layer])
    layer.ffn = torch.nn.Module()
    layer.ffn.local_experts = 1
    expert = torch.nn.Module()
    expert.w1 = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    layer.ffn.experts = torch.nn.ModuleList([expert, None])
    layer.engram = torch.nn.Module()
    layer.engram.embed = torch.nn.Module()
    size = 2 if owner_size is None else owner_size
    rank = 0 if owner_rank is None else owner_rank
    rows = (10 + size - 1) // size
    layer.engram.embed.weight = torch.nn.Parameter(torch.empty(rows, 32, dtype=torch.bfloat16))
    layer.engram.embed.scale = torch.nn.Parameter(torch.empty(rows, 1, dtype=torch.bfloat16))
    model.requires_grad_(False)
    source_weight = torch.arange(320).reshape(10, 32).bfloat16()
    source_scale = torch.arange(10).reshape(10, 1).bfloat16()
    source = {
        "layers.0.ffn.experts.0.w1.weight": torch.full((2, 2), 3, dtype=torch.bfloat16),
        "layers.0.ffn.experts.1.w1.weight": torch.full((2, 2), 7, dtype=torch.bfloat16),
        "layers.0.engram.embed.weight": source_weight,
        "layers.0.engram.embed.scale": source_scale,
    }
    save_file(source, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": dict.fromkeys(source, "weights.safetensors")}))
    result = load_native_ep_checkpoint(
        model, tmp_path, ep_rank=0, ep_size=2, engram_rank=owner_rank,
        engram_size=owner_size, require_verified_cache=False)
    expected_weight = torch.zeros_like(layer.engram.embed.weight)
    expected_scale = torch.ones_like(layer.engram.embed.scale)
    selected_weight = source_weight[rank * rows:(rank + 1) * rows]
    selected_scale = source_scale[rank * rows:(rank + 1) * rows]
    expected_weight[:len(selected_weight)] = selected_weight
    expected_scale[:len(selected_scale)] = selected_scale
    torch.testing.assert_close(layer.engram.embed.weight, expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(layer.engram.embed.scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(expert.w1.weight, source["layers.0.ffn.experts.0.w1.weight"], rtol=0, atol=0)
    assert result["ep_rank"] == 0 and result["ep_size"] == 2
    assert result["engram_rank"] == rank and result["engram_size"] == size
    assert result["unused_tensors"] == {"other-EP-rank-expert": 1}
