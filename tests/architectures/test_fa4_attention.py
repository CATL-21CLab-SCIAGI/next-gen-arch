"""Optional FA4 scope, explicit output boundary and exact dependency admission."""

import hashlib
import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from archlab.architectures import fa4_attention as leaf


@pytest.mark.parametrize("n,window", [(1, 1), (31, 1), (31, 17), (129, 65), (10240, 1025)])
def test_cpu_and_unsupported_scopes_keep_original_backward(monkeypatch, n, window):
    calls = []
    forward = object()

    def original(q, k, v, **kwargs):
        calls.append(kwargs)
        return q.float() + k.float().repeat_interleave(5, dim=2) + v.float().repeat_interleave(5, dim=2)

    def unexpected():
        pytest.fail("fallback must not load optional native dependencies")

    monkeypatch.setattr(leaf, "_qualified_gqa", original)
    monkeypatch.setattr(leaf, "_native_function", unexpected)
    values = [torch.randn(1, n, heads, 128, dtype=torch.bfloat16, requires_grad=True) for heads in (10, 2, 2)]
    do = torch.randn(values[0].shape, dtype=torch.float32)
    expected = original(*values, scaling=.1, long_window=window, forward=forward).to(torch.bfloat16)
    expected_grad = torch.autograd.grad(expected, values, do)
    actual = leaf.native_bf16_gqa_attention(*values, scaling=.1, long_window=window, forward=forward)
    actual_grad = torch.autograd.grad(actual, values, do)
    assert actual.dtype == torch.bfloat16 and torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad, strict=True))
    assert calls[-1] == {"scaling": .1, "long_window": window, "forward": forward}


class Distribution:
    def __init__(self, root, version):
        self.root, self.version = root, version

    def locate_file(self, relative):
        return self.root / relative


def dependencies(monkeypatch, tmp_path):
    roots = {}
    expected_hashes = {}
    cute = tmp_path / "flash-attn-4/flash_attn/cute"
    cute.mkdir(parents=True)
    for name in leaf._UPSTREAM_HASHES:
        path = cute / name
        path.write_text("official fixture " + name)
        expected_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, version in leaf._VERSIONS.items():
        roots[name] = Distribution(tmp_path / name, version)
    monkeypatch.setattr(leaf.metadata, "distribution", lambda name: roots[name])
    monkeypatch.setattr(leaf, "_UPSTREAM_HASHES", expected_hashes)
    paths = {
        "cutlass": tmp_path / "nvidia-cutlass-dsl/nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py",
        "tvm_ffi": tmp_path / "apache-tvm-ffi/tvm_ffi/__init__.py",
        "quack": tmp_path / "quack-kernels/quack/__init__.py",
        "tilelang": tmp_path / "tilelang/tilelang/__init__.py",
    }
    monkeypatch.setattr(leaf.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(paths[name])))
    namespace = SimpleNamespace(__file__="container/flash_attn/__init__.py", __path__=["container/flash_attn"])
    original_import = importlib.import_module
    monkeypatch.setattr(leaf.importlib, "import_module", lambda name: namespace if name == "flash_attn" else original_import(name))
    return roots, cute, namespace


def test_dependency_validation_selects_only_pinned_namespace(monkeypatch, tmp_path):
    _, cute, namespace = dependencies(monkeypatch, tmp_path)
    inspect = leaf.fa4_runtime_contract(validate=False)
    assert not inspect["validated"] and namespace.__path__ == ["container/flash_attn"]
    validated = leaf.fa4_runtime_contract()
    assert validated["validated"]
    assert namespace.__path__ == [str(cute.parent), "container/flash_attn"]
    assert validated["upstream_sha256"] == validated["required_upstream_sha256"]


@pytest.mark.parametrize("failure", ["version", "source", "loaded_module", "import_path"])
def test_incompatible_dependency_cannot_activate_namespace(monkeypatch, tmp_path, failure):
    roots, cute, namespace = dependencies(monkeypatch, tmp_path)
    if failure == "version":
        roots["flash-attn-4"].version = "4.0.0b32"
    elif failure == "source":
        (cute / "flash_bwd_sm100.py").write_text("modified source")
    elif failure == "loaded_module":
        monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", SimpleNamespace(__file__="/container/old/interface.py"))
    else:
        monkeypatch.setattr(leaf.importlib.util, "find_spec", lambda name: SimpleNamespace(origin="/container/old/" + name + ".py"))
    with pytest.raises(RuntimeError, match="FA4 runtime contract"):
        leaf.fa4_runtime_contract()
    assert namespace.__path__ == ["container/flash_attn"]


def test_missing_optional_dependencies_are_inspectable(monkeypatch):
    def missing(name):
        raise leaf.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(leaf.metadata, "distribution", missing)
    inspect = leaf.fa4_runtime_contract(validate=False)
    assert inspect["errors"] and all(value is None for value in inspect["versions"].values())
    with pytest.raises(RuntimeError, match="requires installed version"):
        leaf.fa4_runtime_contract()


def test_deterministic_algorithms_reject_atomic_backward():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        value = torch.ones(1, 1, 1, 128, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="FP32 atomics"):
            leaf.native_bf16_gqa_attention(value, value, value, scaling=.1, long_window=1, forward=None)
    finally:
        torch.use_deterministic_algorithms(previous)
