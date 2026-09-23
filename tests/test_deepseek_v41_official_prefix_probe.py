import hashlib
import json

import pytest

from archlab.automodel.deepseek_v41.qualification.official_prefix_probe import (
    prepare_prefix_fixture,
)


def _sources(tmp_path):
    assets, weights = tmp_path / "assets", tmp_path / "weights"
    assets.mkdir()
    weights.mkdir()
    (assets / "inference").mkdir()
    (assets / "inference/model.py").write_text("# unchanged released source\n")
    (assets / "tokenizer.json").write_text("{}")
    (assets / "inference/config.json").write_text(
        json.dumps({"n_layers": 40, "compress_ratios": [0, 0, 2, 2]})
    )
    (weights / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_hidden_layers": 40,
                    "compress_ratios": [0, 0, 2, 2],
                    "engram_layer_ids": [1, 14],
                },
                "vision_config": {"num_hidden_layers": 32},
            }
        )
    )
    source = {
        name: "original.safetensors"
        for name in (
            "embed.weight",
            "head.weight",
            "norm.weight",
            "layers.0.attn.wq_a.weight",
            "layers.1.engram.embed.weight",
            "layers.2.attn.wo_a.weight",
            "layers.2.attn.wo_a.scale",
            "layers.3.attn.wq_a.weight",
            "vision.weight",
            "mtp.0.weight",
        )
    }
    raw = json.dumps({"weight_map": source}).encode()
    (weights / "model.safetensors.index.json").write_bytes(raw)
    (weights / "original.safetensors").write_bytes(b"untouched fixture shard")
    (weights / "ARCHLAB_VERIFIED_COPY.json").write_text(
        json.dumps({"source_index_sha256": hashlib.sha256(raw).hexdigest()})
    )
    return assets, weights


def test_prefix_fixture_preserves_source_files_and_complete_schedules(tmp_path):
    assets, weights = _sources(tmp_path)
    before = {
        path: path.read_bytes()
        for directory in (assets, weights)
        for path in directory.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "prefix"
    report = prepare_prefix_fixture(assets=assets, weights=weights, output=output)
    selected = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    assert len(selected) == 7
    assert "layers.2.attn.wo_a.scale" in selected
    assert not any(name.startswith(("layers.3.", "vision.", "mtp.")) for name in selected)
    config = json.loads((output / "config.json").read_text())
    assert config["text_config"]["num_hidden_layers"] == 3
    assert config["text_config"]["compress_ratios"] == [0, 0, 2, 2]
    assert config["text_config"]["engram_layer_ids"] == [1, 14]
    assert json.loads((output / "inference/config.json").read_text())["n_layers"] == 3
    assert (output / "original.safetensors").is_symlink()
    assert report["partial_fixture"] and not report["production_qualification"]
    assert all(path.read_bytes() == value for path, value in before.items())
    assert prepare_prefix_fixture(assets=assets, weights=weights, output=output) == report


def test_prefix_fixture_detects_modified_index_on_reuse(tmp_path):
    assets, weights = _sources(tmp_path)
    output = tmp_path / "prefix"
    prepare_prefix_fixture(assets=assets, weights=weights, output=output)
    (output / "model.safetensors.index.json").write_text("{}")
    with pytest.raises(ValueError, match="derived prefix file changed"):
        prepare_prefix_fixture(assets=assets, weights=weights, output=output)


def test_prefix_rejects_changed_original_verification(tmp_path):
    assets, weights = _sources(tmp_path)
    (weights / "model.safetensors.index.json").write_text("{}")
    with pytest.raises(ValueError, match="verified cache"):
        prepare_prefix_fixture(assets=assets, weights=weights, output=tmp_path / "prefix")


def test_observers_preserve_native_forward_and_remove_method_wrappers():
    import torch
    from torch import nn

    from archlab.automodel.deepseek_v41.qualification.official_prefix_probe import capture_prefix

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.compressor = None
            for name in ("wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "wo_b"):
                self.add_module(name, nn.Identity())

        def forward(self, x):
            for module in self.children():
                x = module(x)
            return x

    class Gate(nn.Module):
        def forward(self, x):
            return torch.ones(*x.shape[:-1], 1), torch.zeros(*x.shape[:-1], 1, dtype=torch.long)

    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate, self.shared_experts = Gate(), nn.Identity()

        def forward(self, x):
            self.gate(x.flatten(0, 1))
            return self.shared_experts(x)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.engram = None
            self.attn_norm, self.ffn_norm = nn.Identity(), nn.Identity()
            self.attn, self.ffn = Attention(), FFN()

        def hc_mixes(self, x):
            coefficients = torch.ones(*x.shape[:-1])
            return (
                coefficients,
                coefficients,
                coefficients.unsqueeze(-1).expand(*coefficients.shape, 4),
            )

        def hc_pre(self, x, mix):
            return (x * mix.unsqueeze(-1)).mean(2)

        def hc_post(self, x, residual, post, comb):
            return residual + x.unsqueeze(2)

        def forward(self, x, previous):
            pre, post, comb = self.hc_mixes(x)
            x = self.hc_post(self.attn(self.attn_norm(self.hc_pre(x, previous))), x, post, comb)
            next_pre, post, comb = self.hc_mixes(x)
            x = self.hc_post(self.ffn(self.ffn_norm(self.hc_pre(x, pre))), x, post, comb)
            return x, next_pre

    model = nn.Module()
    model.embed, model.norm = nn.Embedding(12, 4), nn.Identity()
    model.engram_hash = None
    model.layers = nn.ModuleList([Block()])

    def forward():
        x = model.embed(torch.tensor([[1, 2, 3]])).unsqueeze(2).expand(-1, -1, 4, -1)
        x, mix = model.layers[0](x, torch.ones(*x.shape[:-1]))
        return model.norm(model.layers[0].hc_pre(x, mix))

    baseline = forward()
    with capture_prefix(model, official=False) as captured:
        result = forward()
    torch.testing.assert_close(result, baseline, rtol=0, atol=0)
    assert captured["layers.0.attn_mix.pre"].shape == (1, 3, 4)
    assert captured["layers.0.ffn_mix.comb"].shape == (1, 3, 4, 4)
    assert captured["layers.0.final_collapsed"].shape == (1, 3, 4)
    assert not any(name in model.layers[0].__dict__ for name in ("hc_mixes", "hc_pre", "hc_post"))
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
