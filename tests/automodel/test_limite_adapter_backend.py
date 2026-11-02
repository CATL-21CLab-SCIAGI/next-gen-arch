"""Checkpoint backend inheritance and the resolved kernel runtime contract."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from archlab.automodel import limite_adapter_common as common


@pytest.fixture
def model_factory(monkeypatch):
    model = SimpleNamespace(
        model=SimpleNamespace(adapters=torch.nn.Linear(2, 2, bias=False), adapter_config=None)
    )

    def install(model, config):
        model.model.adapter_config = dict(
            variant=config.variant, attention_backend=config.attention_backend
        )
        return model

    monkeypatch.setattr(common, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(common, "install_adapters", install)
    monkeypatch.setattr(common, "LimiteAdapterConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(common, "frozen_fingerprint", lambda model: "frozen-base")
    monkeypatch.setattr(
        common,
        "set_normal_attention_backward",
        lambda model, backward: setattr(model.model, "normal_backward", backward),
    )
    return model


def checkpoint(root, model, backend, *, frozen_sha256="frozen-base"):
    root.mkdir()
    torch.save(model.model.adapters.state_dict(), root / "adapter.pt")
    adapter = dict(variant="normal")
    if backend is not None:
        adapter["attention_backend"] = backend
    receipt = dict(
        adapter=adapter,
        files={"adapter.pt": common.file_hash(root / "adapter.pt")},
        frozen_sha256=frozen_sha256,
    )
    (root / "COMPLETE.json").write_text(json.dumps(receipt))
    return receipt


@pytest.mark.parametrize("requested,expected", [(None, "native"), ("tilelang", "tilelang")])
def test_fresh_backend_selection(model_factory, requested, expected):
    model = common.build_model("snapshot", "normal", "cpu", attention_backend=requested)
    assert model.model.adapter_config["attention_backend"] == expected


@pytest.mark.parametrize("saved,expected", [(None, "native"), ("tilelang", "tilelang")])
def test_resume_inherits_backend_and_preserves_receipt(tmp_path, model_factory, saved, expected):
    root = tmp_path / "checkpoint"
    receipt = checkpoint(root, model_factory, saved)
    expected_state = model_factory.model.adapters.weight.detach().clone()
    with torch.no_grad():
        model_factory.model.adapters.weight.zero_()
    model = common.build_model("snapshot", "normal", "cpu", root)
    assert model.model.adapter_config["attention_backend"] == expected
    torch.testing.assert_close(model.model.adapters.weight, expected_state)
    assert json.loads((root / "COMPLETE.json").read_text()) == receipt


def test_resume_rejects_backend_change(tmp_path, model_factory):
    root = tmp_path / "checkpoint"
    checkpoint(root, model_factory, None)
    with pytest.raises(ValueError, match="checkpoint adapter geometry"):
        common.build_model("snapshot", "normal", "cpu", root, attention_backend="tilelang")


def test_backend_selection_retains_checksum_verification(tmp_path, model_factory):
    root = tmp_path / "checkpoint"
    checkpoint(root, model_factory, "tilelang")
    with (root / "adapter.pt").open("ab") as payload:
        payload.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint checksum mismatch"):
        common.build_model("snapshot", "normal", "cpu", root)


def test_backend_selection_retains_frozen_identity_verification(tmp_path, model_factory):
    root = tmp_path / "checkpoint"
    checkpoint(root, model_factory, "tilelang", frozen_sha256="other-base")
    with pytest.raises(ValueError, match="different frozen base"):
        common.build_model("snapshot", "normal", "cpu", root)


def test_runtime_records_lazy_overlay_versions_and_paths(monkeypatch):
    versions = {"tilelang": "0.1.15", "apache-tvm-ffi": "0.1.11", "torch-c-dlpack-ext": "0.1.5"}

    def version(package):
        if package not in versions:
            raise common.importlib.metadata.PackageNotFoundError(package)
        return versions[package]

    monkeypatch.setattr(common.importlib.metadata, "version", version)
    monkeypatch.setattr(
        common.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=f"/overlay/{name}/__init__.py"),
    )
    monkeypatch.delitem(sys.modules, "tilelang", raising=False)
    monkeypatch.delitem(sys.modules, "tvm_ffi", raising=False)
    monkeypatch.setenv("ARCHLAB_SOURCE_REVISION", "test-revision")
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: (2, 28, 3))
    contract = common.runtime_contract()
    assert {name: contract["packages"][name] for name in versions} == versions
    assert contract["resolved_modules"]["tilelang"] == dict(
        version="0.1.15", path="/overlay/tilelang/__init__.py"
    )
    assert contract["resolved_modules"]["tvm_ffi"] == dict(
        version="0.1.11", path="/overlay/tvm_ffi/__init__.py"
    )
    assert "tilelang" not in sys.modules and "tvm_ffi" not in sys.modules


@pytest.mark.parametrize(
    "variant,backend,expected",
    [
        ("normal", "native", "publisher-sdpa"),
        ("simplicial", "native", "packed-compensated-bf16-pipelined-v4"),
        ("normal", "tilelang", "tilelang-gqa-bf16-v1"),
        ("simplicial", "tilelang", "tilelang-joint-bf16-v2"),
    ],
)
def test_attention_kernel_metadata(variant, backend, expected):
    assert common.attention_kernel_name(variant, backend) == expected
