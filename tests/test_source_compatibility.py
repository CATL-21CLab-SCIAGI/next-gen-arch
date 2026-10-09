import hashlib
import json

import pytest

from archlab import source_compatibility


def test_formatting_receipt_checks_bytes_and_ast(tmp_path, monkeypatch):
    before = b"value=1\n"
    after = b"value = 1\n"
    changed = b"value = 2\n"
    def digest(source):
        return hashlib.sha256(source).hexdigest()

    record = {
        "before_sha256": digest(before),
        "after_sha256": digest(after),
        "ast_sha256": source_compatibility.ast_sha256(before),
    }
    monkeypatch.setattr(source_compatibility, "__file__", str(tmp_path / "module.py"))
    manifest = tmp_path / "data" / "source-formatting-20260923.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"files": {"model.py": record}}))
    source = tmp_path / "model.py"
    source.write_bytes(after)
    check = source_compatibility.formatting_predecessor
    assert check("model.py", source, digest(after)) == (digest(before), record)
    assert check("unknown.py", source, digest(after)) == (digest(after), None)

    source.write_bytes(changed)
    assert check("model.py", source, digest(changed)) == (digest(changed), None)
    with pytest.raises(ValueError, match="formatting-only source proof differs"):
        check("model.py", source, digest(after))

    # Even a matching byte receipt cannot authorize a semantic change.
    record["after_sha256"] = digest(changed)
    manifest.write_text(json.dumps({"files": {"model.py": record}}))
    with pytest.raises(ValueError, match="formatting-only source proof differs"):
        check("model.py", source, digest(changed))
