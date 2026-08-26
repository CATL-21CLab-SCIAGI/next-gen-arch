"""Dependency checks for the architecture/training package boundary."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "archlab"
ARCH_ROOT = PACKAGE_ROOT / "architectures"


def test_architecture_modules_do_not_depend_on_training_runtime() -> None:
    forbidden_prefixes = (
        "archlab.speedrun",
        "archlab.megatron",
        "archlab.optimizers",
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
