from types import SimpleNamespace

import pytest

from archlab.speedrun import precision


class _NVFP4BlockScaling:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_te(monkeypatch, *, available: bool, reason: str = "") -> None:
    te = SimpleNamespace(is_nvfp4_available=lambda **_: (available, reason))
    recipe = SimpleNamespace(NVFP4BlockScaling=_NVFP4BlockScaling)
    monkeypatch.setattr(precision, "_import_transformer_engine", lambda: (te, recipe))


def test_fp4_uses_runtime_capability_when_gpu_label_is_misleading(monkeypatch):
    _install_te(monkeypatch, available=True)

    backend = precision.resolve_precision_backend(
        "fp4_blackwell",
        device_type="cuda",
        gpu_name="NVIDIA L20D",
    )

    assert backend.runtime_backend == "te_fp4"
    assert backend.te_recipe_name == "NVFP4BlockScaling"


def test_fp4_rejects_named_blackwell_when_runtime_capability_is_absent(monkeypatch):
    _install_te(monkeypatch, available=False, reason="compute capability unsupported")

    with pytest.raises(RuntimeError, match="compute capability unsupported"):
        precision.resolve_precision_backend(
            "fp4_blackwell",
            device_type="cuda",
            gpu_name="NVIDIA B300",
        )


def test_fp4_refuses_runtime_that_cannot_report_capability(monkeypatch):
    te = SimpleNamespace()
    recipe = SimpleNamespace(NVFP4BlockScaling=_NVFP4BlockScaling)
    monkeypatch.setattr(precision, "_import_transformer_engine", lambda: (te, recipe))

    with pytest.raises(RuntimeError, match="cannot verify NVFP4 runtime support"):
        precision.resolve_precision_backend(
            "fp4_blackwell",
            device_type="cuda",
            gpu_name="NVIDIA L20D",
        )
