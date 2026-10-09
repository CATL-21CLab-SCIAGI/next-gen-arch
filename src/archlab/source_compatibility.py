"""Verify explicitly recorded formatting-only migrations of sealed source files.

This does not authorize changed model behavior. Each record binds both byte
digests and an identical Python AST, including import order and string values.
Unlisted changes still fail the caller's original checkpoint admission rules.
"""

import ast
import hashlib
import json
from pathlib import Path


def ast_sha256(source):
    def canonical(value):
        if isinstance(value, ast.AST):
            fields = [(name, canonical(item)) for name, item in ast.iter_fields(value)
                      # Python 3.12 added this empty field to existing definitions.
                      if not (name == "type_params" and item == [])]
            return [type(value).__name__, fields]
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, bytes):
            return ["bytes", value.hex()]
        if isinstance(value, (float, complex)):
            return [type(value).__name__, repr(value)]
        if value is Ellipsis:
            return ["ellipsis"]
        return value

    tree = json.dumps(canonical(ast.parse(source)), ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(tree.encode()).hexdigest()


def formatting_predecessor(relative, path, actual_sha256):
    manifest = Path(__file__).with_name("data") / "source-formatting-20260923.json"
    records = json.loads(manifest.read_text())["files"]
    record = records.get(relative)
    if record is None or record["after_sha256"] != actual_sha256:
        return actual_sha256, None
    source = Path(path).read_bytes()
    if (hashlib.sha256(source).hexdigest() != record["after_sha256"]
            or ast_sha256(source) != record["ast_sha256"]):
        raise ValueError(f"formatting-only source proof differs: {relative}")
    return record["before_sha256"], record
