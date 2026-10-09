import sys
from types import ModuleType, SimpleNamespace

import pytest

from archlab.megatron import miles_v41_checkpoint_writer as writer


@pytest.fixture
def native(monkeypatch):
    package = ModuleType("miles.backends.megatron_utils")
    calls = []
    config = SimpleNamespace(async_save=False, ckpt_format="torch_dist", ckpt_assume_constant_structure=False)
    model = SimpleNamespace(get_args=lambda: config, save_checkpoint=lambda *a, **kw: calls.append((a, kw)))
    package.model = model
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils", package)
    strategy = object()
    monkeypatch.setattr(writer, "make_strategy", lambda: strategy)
    return model, config, calls, strategy


def test_preserves_native_save_and_context(native):
    model, _, calls, strategy = native
    context = {"load_strategy": object()}
    writer.install()
    installed = model.save_checkpoint
    writer.install()
    assert model.save_checkpoint is installed
    model.save_checkpoint(3, "model", "optimizer", checkpointing_context=context)
    assert calls[0][0] == (3, "model", "optimizer")
    assert calls[0][1]["checkpointing_context"] == context | {"save_strategy": strategy}
    assert "save_strategy" not in context


@pytest.mark.parametrize("name,value", [("async_save", True), ("ckpt_format", "torch"), ("ckpt_assume_constant_structure", True)])
def test_rejects_incompatible_save_contract(native, name, value):
    model, config, calls, _ = native
    setattr(config, name, value)
    writer.install()
    with pytest.raises(ValueError):
        model.save_checkpoint(3)
    assert not calls
