import pytest
import torch

from archlab.serving.sglang_v41_live_weights import _stage_compressor, _verify_expert_slices


def test_compressor_pairs_can_cross_recycled_ipc_buckets():
    state = {"compressor_pairs": {}}
    prefix = "layers.0.attn.compressor"
    target = prefix + ".wkv_gate.weight"
    parameters = {target: torch.empty(4, 3)}
    first = torch.arange(6).reshape(2, 3)
    expected = first.clone()
    assert _stage_compressor(state, prefix + ".wkv.weight", first, parameters, lambda x: x) == []
    first.zero_()
    result = _stage_compressor(state, prefix + ".wgate.weight", torch.ones(2, 3), parameters, lambda x: x)
    assert result[0][0] == target
    torch.testing.assert_close(result[0][1], torch.cat([expected, torch.ones(2, 3)]))
    assert state["compressor_pairs"] == {}


def test_one_expert_cannot_establish_full_parameter_coverage():
    key = "model.layers.0.mlp.experts.w13_weight"
    parameters = {key: torch.empty(2, 4, 3)}
    loaded = {(key, 2, "w1"), (key, 2, "w3"), (key, 3, "w1")}
    with pytest.raises(ValueError, match="expert=3, part=w3"):
        _verify_expert_slices(parameters, loaded, rank=1)
    loaded.add((key, 3, "w3"))
    _verify_expert_slices(parameters, loaded, rank=1)
