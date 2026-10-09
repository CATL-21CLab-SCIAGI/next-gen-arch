from pathlib import Path

import pytest

from archlab.recipe import read_recipe, recipe_sha256
from archlab.spec import SpecError, load_experiment


def test_catalog_requires_explicit_selection_and_binds_hash_to_selected_profile(tmp_path):
    path = tmp_path / "catalog.yaml"
    path.write_text("profiles:\n  normal: {variant: normal}\n  simplicial: {variant: simplicial}\n")
    with pytest.raises(ValueError, match="explicit"):
        read_recipe(path)
    with pytest.raises(ValueError, match="unknown"):
        read_recipe(str(path) + "#absent")
    normal, simplicial = str(path) + "#normal", str(path) + "#simplicial"
    assert read_recipe(normal) == {"variant": "normal"}
    assert recipe_sha256(normal) != recipe_sha256(simplicial)
    selected = read_recipe(normal)
    selected["variant"] = "changed"
    assert read_recipe(normal)["variant"] == "normal"


def test_same_catalog_profiles_compose_without_false_cycles_and_real_cycle_is_rejected(tmp_path):
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "profiles:\n"
        "  base: {schema_version: 1, backend: speedrun, seed: 42}\n"
        "  model: {selection: {size: 100m}}\n"
        "  arm:\n"
        "    extends: ['catalog.yaml#base', 'catalog.yaml#model']\n"
        "    name: paired-normal\n"
        "    variant: baseline\n"
        "  cycle: {extends: 'catalog.yaml#cycle'}\n"
    )
    spec = load_experiment(str(path) + "#arm")
    assert spec.variant == "baseline" and spec.config["selection"]["size"] == "100m"
    assert len(spec.source_files) == 3
    assert all(source.is_file() for source in spec.source_files)
    with pytest.raises(SpecError, match="cyclic"):
        load_experiment(str(path) + "#cycle")
    assert Path(str(path) + "#arm") == spec.source
