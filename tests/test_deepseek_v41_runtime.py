"""Verified native imports may share dependencies through fixture symlinks."""

import hashlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

from archlab.automodel import deepseek_v41_runtime as runtime


@pytest.fixture
def reference_paths(tmp_path, monkeypatch):
    original = tmp_path / "original/inference"
    fixture = tmp_path / "prefix/inference"
    original.mkdir(parents=True)
    fixture.mkdir(parents=True)
    contents = {
        "model.py": "from kernel import sentinel\nMARKER = sentinel\n",
        "kernel.py": "sentinel = 42\n",
        "engram.py": "# reviewed engram\n",
        "vision.py": "# reviewed vision\n",
        "image_processor.py": "# reviewed image processor\n",
    }
    digests = {}
    for name, source in contents.items():
        path = original / name
        path.write_text(source)
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (fixture / name).symlink_to(path)
        if name != "model.py":
            module = ModuleType(path.stem)
            module.__file__ = str(path)
            module.sentinel = 42
            monkeypatch.setitem(sys.modules, path.stem, module)
    monkeypatch.setattr(runtime, "REFERENCE_DIGESTS", digests)
    return original, fixture


def test_native_import_accepts_symlink_identical_loaded_dependencies(reference_paths):
    original, fixture = reference_paths
    assert Path(sys.modules["kernel"].__file__).resolve() == (fixture / "kernel.py").resolve()
    assert Path(sys.modules["kernel"].__file__) != fixture / "kernel.py"
    name = "_archlab_test_native_symlink_reference"
    before_path = sys.path[:]
    try:
        module = runtime.load_native_reference(fixture.parent, module_name=name)
        assert module.MARKER == 42
        assert Path(sys.modules["kernel"].__file__) == original / "kernel.py"
        assert sys.path == before_path
    finally:
        sys.modules.pop(name, None)


def test_native_import_rejects_distinct_loaded_dependency_path(
    reference_paths, tmp_path, monkeypatch
):
    original, fixture = reference_paths
    other = tmp_path / "other/kernel.py"
    other.parent.mkdir()
    # Even identical bytes at a different canonical path are a module collision.
    other.write_bytes((original / "kernel.py").read_bytes())
    module = ModuleType("kernel")
    module.__file__ = str(other)
    monkeypatch.setitem(sys.modules, "kernel", module)
    name = "_archlab_test_native_different_reference"
    before_path = sys.path[:]
    with pytest.raises(RuntimeError, match="collide with module kernel"):
        runtime.load_native_reference(fixture.parent, module_name=name)
    assert name not in sys.modules
    assert sys.path == before_path


@pytest.mark.parametrize("changed_name", ["kernel.py", "model.py"])
def test_native_symlink_import_still_verifies_every_reviewed_digest(reference_paths, changed_name):
    original, fixture = reference_paths
    (original / changed_name).write_text("# modified after the reviewed digest was recorded\n")
    name = "_archlab_test_native_modified_reference"
    before_path = sys.path[:]
    with pytest.raises(ValueError, match="unreviewed reference source"):
        runtime.load_native_reference(fixture.parent, module_name=name)
    assert name not in sys.modules
    assert sys.path == before_path
