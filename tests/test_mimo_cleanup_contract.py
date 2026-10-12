"""Exercise the patched upstream cleanup against a real disposable Git tree."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_cleanup_keeps_root_dependencies_without_retaining_nested_build_output(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "third_party/verl/third_party/mimoagent-osr/src"
    if not (source / "mimoagent/__init__.py").is_file():
        pytest.skip("requires the initialized pinned MiMo submodule")
    monkeypatch.syspath_prepend(str(source))
    pytest.importorskip("mimoagent")
    from mimoagent.environments.datasets.base import DatasetEnvironment

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    files = ("lib/dependency.txt", "node_modules/dependency.txt", "Manifest.toml",
             "build/lib/pyupgrade.py", "build/node_modules/generated.txt")
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("disposable fixture")

    def execute(command, *, cwd, timeout):
        completed = subprocess.run(command, shell=True, cwd=cwd, capture_output=True,
                                   text=True, timeout=timeout, check=True)
        return {"returncode": completed.returncode, "output": completed.stdout}

    context = SimpleNamespace(
        _language="", _CLEAN_KEEP=DatasetEnvironment._CLEAN_KEEP,
        _CLEAN_KEEP_UNKNOWN=DatasetEnvironment._CLEAN_KEEP_UNKNOWN,
        _CLEAN_KEEP_COMMON=DatasetEnvironment._CLEAN_KEEP_COMMON,
        repo_path=str(tmp_path), env=SimpleNamespace(execute=execute, copy_to=shutil.copyfile),
        logger=SimpleNamespace(warning=lambda message: pytest.fail(message)))
    DatasetEnvironment._purge_build_artifacts(context)
    for name in files[:3]:
        assert (tmp_path / name).is_file(), name
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("declaration", [
    '[tool.hatch.build.hooks.vcs]\nversion-file = "src/demo/_version.py"\n',
    '[tool.setuptools_scm]\nwrite_to = "src/demo/_version.py"\n',
    '[tool.setuptools_scm]\nversion_file = "src/demo/_version.py"\n',
    '[tool.hatch.build.hooks.vcs]\nversion-file = "src"\n',
])
def test_cleanup_preserves_declared_version_from_captured_base(tmp_path, monkeypatch, declaration):
    source = Path(__file__).resolve().parents[1] / "third_party/verl/third_party/mimoagent-osr/src"
    if not (source / "mimoagent/__init__.py").is_file():
        pytest.skip("requires the initialized pinned MiMo submodule")
    monkeypatch.syspath_prepend(str(source))
    from mimoagent.environments.datasets.base import DatasetEnvironment

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    (tmp_path / "pyproject.toml").write_text(declaration)
    git("add", "pyproject.toml")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "base")
    base_ref = git("rev-parse", "HEAD")
    # An agent-modified declaration must not protect arbitrary new files.
    (tmp_path / "pyproject.toml").write_text('[tool.hatch.build.hooks.vcs]\nversion-file = "rogue.py"\n')
    for name in ("src/demo/_version.py", "rogue.py", "build/lib/generated.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")

    def execute(command, *, cwd, timeout):
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True,
                                text=True, timeout=timeout)
        return {"returncode": result.returncode, "output": result.stdout}

    context = SimpleNamespace(_base_ref=base_ref, _language="", _generated_version_snapshots={},
        _CLEAN_KEEP=DatasetEnvironment._CLEAN_KEEP,
        _CLEAN_KEEP_UNKNOWN=DatasetEnvironment._CLEAN_KEEP_UNKNOWN,
        _CLEAN_KEEP_COMMON=DatasetEnvironment._CLEAN_KEEP_COMMON,
        repo_path=str(tmp_path), env=SimpleNamespace(execute=execute, copy_to=shutil.copyfile),
        logger=SimpleNamespace(warning=lambda message: pytest.fail(message)))
    DatasetEnvironment._purge_build_artifacts(context)
    assert (tmp_path / "src/demo/_version.py").is_file() is ("_version.py" in declaration)
    assert not (tmp_path / "rogue.py").exists()
    assert not (tmp_path / "build").exists()
    if "_version.py" in declaration:
        version = tmp_path / "src/demo/_version.py"
        version.write_text("policy-modified metadata")
        DatasetEnvironment._restore_generated_versions(context)
        assert version.read_text() == "fixture"
        version.unlink()
        DatasetEnvironment._restore_generated_versions(context)
        assert version.read_text() == "fixture"
        outside = tmp_path.parent / (tmp_path.name + "-outside.txt")
        outside.write_text("unchanged outside target")
        version.unlink()
        version.symlink_to(outside)
        with pytest.raises(RuntimeError, match="path changed"):
            DatasetEnvironment._restore_generated_versions(context)
        assert outside.read_text() == "unchanged outside target"
        version.unlink()
        version.parent.rmdir()
        outside_dir = tmp_path.parent / (tmp_path.name + "-outside-dir")
        outside_dir.mkdir()
        version.parent.symlink_to(outside_dir, target_is_directory=True)
        with pytest.raises(RuntimeError, match="path changed"):
            DatasetEnvironment._restore_generated_versions(context)
        assert not list(outside_dir.iterdir())


@pytest.mark.parametrize("path", ["../secret.py", "/root/secret.py", "src/*.py", "src/", ".", "src/../secret.py"])
def test_generated_version_exclusion_rejects_nonliteral_paths(monkeypatch, path):
    source = Path(__file__).resolve().parents[1] / "third_party/verl/third_party/mimoagent-osr/src"
    if not (source / "mimoagent/__init__.py").is_file():
        pytest.skip("requires the initialized pinned MiMo submodule")
    monkeypatch.syspath_prepend(str(source))
    from mimoagent.environments.datasets.base import _generated_version_paths

    assert not _generated_version_paths(f'[tool.hatch.build.hooks.vcs]\nversion-file = "{path}"\n')
