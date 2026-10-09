"""Exact extraction guards against pre-refactor f822c4b (formatting ignored).

These are structural checks, not claims of distributed numerical qualification.
Intentional future mechanism edits should replace the affected guard with a
behavioral regression test and an explicit experiment contract.
"""

import ast
import hashlib
from pathlib import Path

import pytest


@pytest.mark.parametrize("module,name,expected", [
    ("qwen38_flash_next_model", "_build_model_classes", "787e3190f5db00f39716a5d3ab8fb5be8a70a77b19ce1320d29e8b44cbf42a0d"),
    ("qwen38_flash_next_model", "build_model", "80bce51cd6f0cd2fbbfad7a91293decd9869e05cee6a6dc3cab472b0551c8c19"),
    ("qwen38_flash_next_model", "_tag_native_optimizer_fallbacks", "84746a895574adcd27aeae2abf2977a28b53328ea40b1bc6804f2d69114589d6"),
    ("checkpoint_staging", "_execute_checkpoint_request_by_local_rank", "038c62ea924cab7235ae9a3a91ebe4fb90f2be9eca8452b061f938a7a2968c42"),
])
def test_numerical_and_checkpoint_functions_are_exact_extractions(module, name, expected):
    path = Path(__file__).resolve().parents[1] / "src" / "archlab" / "megatron" / f"{module}.py"
    node, = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name]
    # Python 3.12 adds empty type_params fields to functions/classes. Ignore
    # those empty fields so the original 3.11 fingerprint is portable; retain
    # nonempty type parameters, which would be a real source change.
    for child in ast.walk(node):
        if getattr(child, "type_params", None) == []:
            child._fields = tuple(field for field in child._fields if field != "type_params")
    assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == expected
