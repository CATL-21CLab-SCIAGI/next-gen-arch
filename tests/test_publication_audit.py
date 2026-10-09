import subprocess

import pytest

from archlab.reporting.publication_audit import audit_tree, inspect_blob


def test_publication_boundary_rejects_staging_and_credentials_without_echoing_them():
    assert "workspace_artifact" in inspect_blob("results/run/state.json", "100644", b"{}")
    assert "data_or_secret_file" in inspect_blob("docs/weights.pt", "100644", b"weights")
    assert "non_regular_file" in inspect_blob("docs/link", "120000", b"../../private")
    token = b"ghp_" + b"A" * 36
    assert inspect_blob("src/config.py", "100644", token) == ["github_token"]
    assert inspect_blob("src/code.py", "100644", b"import os\ntoken = os.environ['TOKEN']\n") == []


def test_tree_audit_reads_committed_content_not_ignored_or_modified_working_files(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    (tmp_path / "README.md").write_text("public\n")
    git("add", "README.md")
    git("commit", "-qm", "fixture")
    (tmp_path / "README.md").write_text("ghp_" + "A" * 36)
    (tmp_path / "private.pt").write_bytes(b"not published")
    result = audit_tree(tmp_path)
    assert result["passed"]
    assert result["files"] == 1
    assert result["manifest"][0]["bytes"] == len(b"public\n")
    git("add", "README.md")
    git("commit", "-qm", "unsafe fixture")
    assert not audit_tree(tmp_path)["passed"]


def _submodule_tree(tmp_path, configuration, path="verl"):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True).stdout.decode().strip()

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    (tmp_path / "README.md").write_text("public\n")
    git("add", "README.md")
    git("commit", "-qm", "fixture")
    pin = git("rev-parse", "HEAD")
    if configuration is not None:
        (tmp_path / ".gitmodules").write_text(configuration)
        git("add", ".gitmodules")
    if path is not None:
        git("update-index", "--add", "--cacheinfo", f"160000,{pin},{path}")
    git("commit", "-qm", "submodule fixture")
    return pin


def test_registered_submodule_records_pin_without_reading_upstream_source(tmp_path):
    config = '[submodule "verl"]\n\tpath = verl\n\turl = https://github.com/XiaomiMiMo/verl.git\n'
    pin = _submodule_tree(tmp_path, config)
    (tmp_path / "verl").mkdir()
    (tmp_path / "verl" / "weights.pt").write_bytes(b"not a published project blob")
    (tmp_path / ".gitmodules").write_text("modified working copy\n")
    result = audit_tree(tmp_path)
    assert result["passed"]
    assert result["files"] == 2
    assert result["submodules"] == [{
        "name": "verl", "path": "verl", "url": "https://github.com/XiaomiMiMo/verl.git",
        "revision": pin,
    }]


@pytest.mark.parametrize(("configuration", "path"), [
    (None, "verl"),
    ('[submodule "verl"]\npath = verl\nurl = https://example.invalid/verl.git\n', "verl"),
    ('[submodule "verl"]\npath = verl\nurl = ../verl.git\n', "verl"),
    ('[submodule "other"]\npath = other\nurl = https://github.com/XiaomiMiMo/verl.git\n', "other"),
    ('[submodule "verl"]\npath = verl\nurl = https://github.com/XiaomiMiMo/verl.git\n'
     'update = !unsafe\n', "verl"),
    ('[submodule "verl"]\npath = verl\nurl = https://github.com/XiaomiMiMo/verl.git\n'
     'url = https://example.invalid/other.git\n', "verl"),
    ('[submodule "verl"]\npath = verl\nurl = https://github.com/XiaomiMiMo/verl.git\n', None),
    ("invalid configuration\n", "verl"),
])
def test_tree_audit_rejects_unregistered_or_misconfigured_submodules(tmp_path, configuration, path):
    _submodule_tree(tmp_path, configuration, path)
    result = audit_tree(tmp_path)
    assert not result["passed"]
    assert any(row["reasons"] for row in result["findings"])
