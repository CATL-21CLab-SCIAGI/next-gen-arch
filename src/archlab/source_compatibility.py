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
    tree = ast.dump(ast.parse(source), annotate_fields=True, include_attributes=False)
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
