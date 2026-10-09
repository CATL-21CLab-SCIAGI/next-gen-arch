from pathlib import Path

import pytest
import torch

from archlab.architectures.limite_adapter import (
    AttentionPrelude,
    LimiteAdapterConfig,
    set_normal_attention_kernel,
)
from archlab.architectures.limite_loader import upstream_classes


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
