import subprocess

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
