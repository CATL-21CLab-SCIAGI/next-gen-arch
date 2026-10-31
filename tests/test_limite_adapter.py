import sys
from pathlib import Path

import pytest
import torch

from archlab.architectures.limite_adapter import (
    AttentionPrelude,
    DeltaPrelude,
    LimiteAdapterConfig,
    matched_compensation,
    prelude_parameters,
    set_normal_attention_kernel,
)
from archlab.architectures.limite_loader import upstream_classes

GDN = LimiteAdapterConfig(variant="gdn", heads=5, attention_backend="reference")
TRIADIC = LimiteAdapterConfig(variant="triadic", heads=5, feature_dim=4, attention_backend="reference",
                              compensation=16)


def _native():
    pytest.importorskip("transformers")
    snapshot = Path("/mnt/oss/models/limite-1b-base-cc612bafcd4a")
    if not snapshot.exists():
        pytest.skip("verified native snapshot is not installed")
    config_cls, model_cls = upstream_classes(snapshot)
    return (config_cls.from_pretrained(snapshot, local_files_only=True),
            sys.modules[model_cls.__module__].LimiteAttention)


def test_native_geometry_and_shared_initialization():
    pytest.importorskip("transformers")
    snapshot = Path("/mnt/oss/models/limite-1b-base-cc612bafcd4a")
    if not snapshot.exists():
        pytest.skip("verified native snapshot is not installed")
    config_cls, model_cls = upstream_classes(snapshot)
    import sys

    native_cls = sys.modules[model_cls.__module__].LimiteAttention
    cfg = config_cls.from_pretrained(snapshot, local_files_only=True)
    for index in (0, 3):
        original = native_cls(cfg, index)
        normal = AttentionPrelude(original, LimiteAdapterConfig(variant="normal"), index)
        simplicial = AttentionPrelude(original, LimiteAdapterConfig(variant="simplicial"), index)
        for name in [
            "head_dim",
            "num_heads",
            "num_kv_heads",
            "scaling",
            "is_global",
            "window_span",
            "applies_rope",
            "has_xsa",
            "has_ve",
        ]:
            assert (
                getattr(normal.native, name)
                == getattr(original, name)
                == getattr(simplicial.native, name)
            )
        for name, p in normal.native.named_parameters():
            torch.testing.assert_close(
                p, dict(simplicial.native.named_parameters())[name], rtol=0, atol=0
            )
        assert normal.native.o_proj.weight.count_nonzero() == 0
        assert simplicial.short_kv.out_features == 2 * original.kv_size
        assert type(normal.native).forward.__code__ is normal.native.forward.__func__.__code__
        assert original.config._attn_implementation != "archlab_simplicial"


def test_invalid_variant():
    with pytest.raises(ValueError):
        LimiteAdapterConfig(variant="other")


@pytest.mark.parametrize("invalid", [
    dict(variant="gdn", heads=5, feature_dim=4, attention_backend="reference"),
    dict(variant="triadic", heads=5, feature_dim=1, attention_backend="reference"),
    dict(variant="triadic", heads=5, feature_dim=3, attention_backend="reference"),
    dict(variant="gdn", heads=5),
    dict(variant="gdn", attention_backend="reference"),
    dict(variant="normal", heads=5),
    dict(variant="simplicial", attention_backend="official"),
    dict(variant="normal", compensation=-1),
])
def test_prelude_mechanism_contracts(invalid):
    with pytest.raises(ValueError):
        LimiteAdapterConfig(**invalid)


@pytest.mark.parametrize("config", [
    GDN, TRIADIC,
    LimiteAdapterConfig(variant="normal", compensation=16),
    LimiteAdapterConfig(variant="simplicial", short_window=64, compensation=8),
])
def test_prelude_parameter_count_matches_construction(config):
    cfg, native_cls = _native()
    for index in (0, 3):
        layer = native_cls(cfg, index)
        prelude = (DeltaPrelude if config.variant in ("gdn", "triadic") else AttentionPrelude)(
            layer, config, index)
        assert sum(p.numel() for p in prelude.parameters()) == prelude_parameters(layer, config)


@pytest.mark.parametrize("config", [GDN, TRIADIC])
def test_delta_prelude_starts_at_identity_keeps_rows_separate_and_learns(config):
    cfg, native_cls = _native()
    prelude = DeltaPrelude(native_cls(cfg, 1), config, 1)
    hidden = torch.randn(2, 7, cfg.hidden_size)
    assert torch.equal(prelude(hidden, dict(cache=None)), hidden)
    with torch.no_grad():
        prelude.core.output.weight.normal_(std=0.02)
    packed = prelude(hidden, dict(cache=None))
    rows = torch.cat([prelude(hidden[i : i + 1], dict(cache=None)) for i in range(2)])
    torch.testing.assert_close(packed, rows, rtol=1e-5, atol=1e-5)
    assert not torch.equal(packed, hidden)
    packed.square().sum().backward()
    assert prelude.core.q.weight.grad.count_nonzero()
    if prelude.compensation is not None:
        assert prelude.compensation.down.weight.grad.count_nonzero()
    with pytest.raises(ValueError, match="cached decode"):
        prelude(hidden, dict(cache=object()))


def test_matched_compensation_meets_the_recipe_bound_on_native_geometry():
    cfg, native_cls = _native()
    layers = [native_cls(cfg, index) for index in range(cfg.num_hidden_layers)]
    arms = dict(
        normal=LimiteAdapterConfig(variant="normal", attention_backend="tilelang"),
        simplicial_w16=LimiteAdapterConfig(variant="simplicial", attention_backend="tilelang"),
        simplicial_w4=LimiteAdapterConfig(variant="simplicial", short_window=4,
                                          attention_backend="tilelang"),
        gdn=GDN, triadic_e4=TRIADIC,
        triadic_e16=LimiteAdapterConfig(variant="triadic", heads=5, feature_dim=16,
                                        attention_backend="reference"),
    )
    matched = matched_compensation(layers, arms)
    assert matched["simplicial_w16"][0] == matched["simplicial_w4"][0] == 64
    assert len({total for _, total, _ in matched.values()}) > 1
    assert all(width > 0 and error <= 0.005 for width, _, error in matched.values())


def test_normal_kernel_selection_preserves_geometry_and_tensor_state():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.adapter_config = dict(variant="normal", attention_backend="tilelang")
    adapters = [torch.nn.Module() for _ in range(2)]
    for adapter in adapters:
        adapter.native = torch.nn.Linear(2, 2)
    model.model.adapters = torch.nn.ModuleList(adapters)
    geometry = dict(model.model.adapter_config)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    set_normal_attention_kernel(model, "gqa")
    assert model.model.normal_kernel == "gqa"
    assert all(adapter.native._archlab_gqa_kernel == "gqa" for adapter in adapters)
    assert model.model.adapter_config == geometry
    assert model.state_dict().keys() == state.keys()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, state[name])
    set_normal_attention_kernel(model, "shared")
    assert all(adapter.native._archlab_gqa_kernel == "shared" for adapter in adapters)


@pytest.mark.parametrize("variant,backend", [("normal", "native"), ("simplicial", "tilelang")])
def test_gqa_selection_rejects_other_execution_paths(variant, backend):
    from types import SimpleNamespace

    model = SimpleNamespace(
        model=SimpleNamespace(adapter_config=dict(variant=variant, attention_backend=backend))
    )
    with pytest.raises(ValueError, match="normal TileLang"):
        set_normal_attention_kernel(model, "gqa")
