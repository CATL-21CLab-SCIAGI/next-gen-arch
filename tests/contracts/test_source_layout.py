"""Dependency checks for the architecture/training package boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "archlab"
ARCH_ROOT = PACKAGE_ROOT / "architectures"


def test_architecture_modules_do_not_depend_on_training_runtime() -> None:
    forbidden_prefixes = (
        "megatron",
        "archlab.megatron",
        "archlab.training",
    )
    violations: list[tuple[str, str]] = []
    for path in sorted(ARCH_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            violations.extend(
                (path.name, module) for module in modules if module.startswith(forbidden_prefixes)
            )
    assert violations == []


@pytest.mark.parametrize("statement", [
    "import megatron",
    "import megatron.core.dist_checkpointing.mapping as mapping",
    "from megatron.core.dist_checkpointing.mapping import ShardedTensor",
    "from archlab.megatron.ple_checkpoint import DistributedPLE",
])
def test_architecture_guard_catches_method_local_runtime_imports(tmp_path, monkeypatch, statement):
    (tmp_path / "model.py").write_text("def checkpoint():\n    " + statement + "\n")
    monkeypatch.setitem(globals(), "ARCH_ROOT", tmp_path)
    with pytest.raises(AssertionError):
        test_architecture_modules_do_not_depend_on_training_runtime()
