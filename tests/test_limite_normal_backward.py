"""Native backward selection retains geometry and cached-decoding dispatch."""

import sys
from types import SimpleNamespace

import pytest
import torch

from archlab.architectures.limite_adapter import (
    _tilelang_reduce,
    set_normal_attention_backward,
    set_normal_attention_kernel,
)
from archlab.automodel.limite_adapter_common import attention_kernel_name


def policy(*, variant="normal", backend="tilelang", kernel="gqa"):
    return SimpleNamespace(
        model=SimpleNamespace(
            adapter_config={"variant": variant, "attention_backend": backend},
            normal_kernel=kernel,
            adapters=[SimpleNamespace(native=SimpleNamespace()) for _ in range(2)],
        )
    )


def test_selector_persists_execution_only_and_can_restore_tilelang(monkeypatch):
    model = policy()
    monkeypatch.setitem(
        sys.modules,
        "archlab.architectures.fa4_attention",
        SimpleNamespace(fa4_runtime_contract=lambda: {"backend": "fa4", "validated": True}),
    )
    geometry = model.model.adapter_config.copy()
    set_normal_attention_backward(model, "fa4")
    assert model.model.normal_backward == "fa4"
    assert all(a.native._archlab_normal_backward == "fa4" for a in model.model.adapters)
    assert model.archlab_normal_attention_backward_contract["validated"]
    set_normal_attention_backward(model, "tilelang")
    assert all(a.native._archlab_normal_backward == "tilelang" for a in model.model.adapters)
    assert model.model.adapter_config == geometry


def test_kernel_change_cannot_create_an_unreloadable_fa4_checkpoint(monkeypatch):
    model = policy()
    monkeypatch.setitem(
        sys.modules,
        "archlab.architectures.fa4_attention",
        SimpleNamespace(fa4_runtime_contract=lambda: {"validated": True}),
    )
    set_normal_attention_backward(model, "fa4")
    with pytest.raises(ValueError, match="before leaving the FA4 GQA"):
        set_normal_attention_kernel(model, "shared")
    assert model.model.normal_kernel == "gqa" and model.model.normal_backward == "fa4"
    set_normal_attention_backward(model, "tilelang")
    set_normal_attention_kernel(model, "shared")
    assert model.model.normal_kernel == "shared"


@pytest.mark.parametrize(
    "variant,backend,kernel",
    [
        ("simplicial", "tilelang", "gqa"),
        ("normal", "native", "gqa"),
        ("normal", "tilelang", "shared"),
    ],
)
def test_fa4_selector_rejects_other_architectures(variant, backend, kernel):
    with pytest.raises(ValueError, match="normal TileLang GQA"):
        set_normal_attention_backward(
            policy(variant=variant, backend=backend, kernel=kernel), "fa4"
        )


@pytest.mark.parametrize("bounded", [False, True])
def test_cached_decode_keeps_original_kernel_even_with_fa4_selected(monkeypatch, bounded):
    calls = []

    def decode(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, kwargs))
        return q + 1

    def reject(*args, **kwargs):
        raise AssertionError("Cached decoding entered a prefill reduction")

    monkeypatch.setitem(
        sys.modules,
        "archlab.architectures.tilelang_attention",
        SimpleNamespace(tilelang_attention=reject, tilelang_decode=decode),
    )
    monkeypatch.setitem(
        sys.modules,
        "archlab.architectures.fa4_attention",
        SimpleNamespace(native_bf16_gqa_attention=reject),
    )
    lengths = torch.tensor([5, 1, 0], dtype=torch.int32) if bounded else None
    module = SimpleNamespace(
        is_global=True, _archlab_normal_backward="fa4", _archlab_decode_lengths=lengths,
    )
    q = torch.zeros(1, 10, 1, 8, dtype=torch.bfloat16)
    kv = torch.zeros(1, 2, 5, 8, dtype=torch.bfloat16)
    out, probabilities = _tilelang_reduce(module, q, kv, kv, None, scaling=0.1)
    assert probabilities is None
    assert out.shape == (1, 1, 10, 8) and out.dtype == torch.bfloat16
    assert torch.equal(out, torch.ones_like(out))
    assert len(calls) == 1
    q_shape, k_shape, options = calls[0]
    assert q_shape == torch.Size([1, 1, 10, 8]) and k_shape == torch.Size([1, 5, 2, 8])
    assert options["scaling"] == 0.1 and options["short"] is None
    assert options["lengths"] is lengths


def test_kernel_metadata_names_native_fa4_boundary():
    assert attention_kernel_name(
        "normal", "tilelang", normal_kernel="gqa", normal_backward="fa4"
    ) == ("tilelang-forward-fa4-native-bf16-backward-v3")
