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


def test_qwen_consumers_do_not_import_training_entrypoints() -> None:
    forbidden = {"archlab.megatron.qwen38_train", "archlab.megatron.qwen38_flash_next_full_train"}
    violations = []
    for path in sorted((PACKAGE_ROOT / "megatron").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module in forbidden:
                violations.append((path.name, node.module))
    assert violations == []


def test_neutral_helpers_do_not_import_execution_backends() -> None:
    for name in ("artifacts", "campaigns", "model_factory", "fineweb", "dataset_paths", "distributed"):
        source = ast.parse((PACKAGE_ROOT / f"{name}.py").read_text())
        for node in ast.walk(source):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            else:
                continue
            assert not any(m.startswith(("archlab.speedrun", "archlab.megatron", "archlab.automodel")) for m in modules), (name, modules)
