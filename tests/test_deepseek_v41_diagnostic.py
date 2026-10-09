import json

import torch

from archlab.automodel.deepseek_v41_diagnostic import _compare_tensors, _json_safe, _snapshot


def test_unsampled_nonfinite_is_reported_and_json_serializable():
    value = torch.ones(1, 100, 4, 8)
    value[0, 50, 0, :3] = torch.tensor([torch.nan, torch.inf, -torch.inf])
    statistics, sample = _snapshot(value, sample_rows=2, chunk_rows=7)
    assert sample.isfinite().all()
    assert statistics["sample_rows"] == [0, 99]
    assert statistics["finite"] is False
    assert statistics["finite_count"] == value.numel() - 3
    assert statistics["nan_count"] == 1
    assert statistics["positive_inf_count"] == statistics["negative_inf_count"] == 1
    assert statistics["finite_l2"] == (value.numel() - 3) ** .5
    json.dumps(statistics, allow_nan=False)
    value.zero_()
    assert torch.equal(sample, torch.ones_like(sample))


def test_full_comparison_detects_difference_outside_sampled_rows():
    original = torch.ones(1, 100, 4)
    actual = original.clone()
    actual[0, 50, 2] = 3
    _, sampled_original = _snapshot(original, sample_rows=2)
    _, sampled_actual = _snapshot(actual, sample_rows=2)
    assert _compare_tensors(sampled_original, sampled_actual)["equal"]
    comparison = _compare_tensors(original, actual, chunk_elements=17)
    assert comparison["equal"] is False
    assert comparison["finite_pair_max_abs"] == 2
    assert comparison["finite_pair_relative_l2"] == .1
    assert comparison["finite_pair_count"] == 400


def test_nonfinite_patterns_and_zero_reference_are_explicit():
    original = torch.tensor([torch.nan, torch.inf, -torch.inf, 0.])
    actual = original.clone()
    equal_pattern = _compare_tensors(original, actual, chunk_elements=2)
    assert equal_pattern["equal"] is False
    assert equal_pattern["same_nonfinite_pattern"] is True
    actual[2] = torch.inf
    actual[3] = 1
    changed = _compare_tensors(original, actual, chunk_elements=2)
    assert changed["same_nonfinite_pattern"] is False
    assert changed["finite_pair_relative_l2"] == 1e20
    json.dumps(_json_safe({"comparison": changed, "overflow": float("inf")}), allow_nan=False)


def test_capture_localizes_first_module_and_block_without_retaining_hooks(monkeypatch):
    from archlab.automodel import deepseek_v41_diagnostic as diagnostic

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Identity()
            self.offset = 0.

        def forward(self, value):
            result = self.attn(value + self.offset)
            return result, result.mean(-1)

    model = torch.nn.Module()
    model.layer = Block()
    model.norm = torch.nn.Identity()
    monkeypatch.setattr(diagnostic, "_observed_modules", lambda _: [
        ("layers.0.attn", model.layer.attn), ("layers.0", model.layer), ("norm", model.norm)])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    value = torch.ones(1, 8, 4)
    def forward():
        model.norm(model.layer(value)[0])
    arguments = {"model": model, "forward": forward, "context": 8, "sample_rows": 2,
                 "emit": lambda *args, **kwargs: None}
    native, _, _ = diagnostic._capture(stage="native", **arguments)
    model.layer.offset = 1.
    _, _, changed = diagnostic._capture(stage="candidate", reference=native, **arguments)
    assert changed["first_sampled_divergence"] == "layers.0.attn"
    assert changed["first_block_sampled_divergence"] == "layers.0.0"
    model.layer.offset = torch.nan
    _, _, nonfinite = diagnostic._capture(stage="nonfinite", reference=native, **arguments)
    assert nonfinite["first_nonfinite"] == "layers.0.attn"
    assert nonfinite["first_block_nonfinite"] == "layers.0.0"
    assert all(not module._forward_hooks for module in model.modules())
