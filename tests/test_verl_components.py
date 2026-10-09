from pathlib import Path

import pytest

from archlab.rl.verl_components import repetition_component


def test_pinned_component_is_importable_without_verl_runtime():
    root = Path(__file__).resolve().parents[1] / "verl"
    if not (root / "recipes/design/repetition.py").exists():
        pytest.skip("upstream submodule not checked out")
    detector, receipt = repetition_component(root)
    assert detector("word " * 1000)
    assert not detector("finite valid mathematical reasoning")
    assert receipt["executor"] == "native Limite with upstream components"


def test_component_rejects_missing_unqualified_checkout(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        repetition_component(tmp_path)
