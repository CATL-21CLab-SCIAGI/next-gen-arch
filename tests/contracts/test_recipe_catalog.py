import pytest

from archlab.recipe import read_recipe, recipe_sha256


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

