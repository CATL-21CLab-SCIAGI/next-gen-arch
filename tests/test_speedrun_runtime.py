from archlab.speedrun.runtime import resolve_climbmix_data_dir


def test_climbmix_data_dir_prefers_current_layout(tmp_path):
    current = tmp_path / "base_data_climbmix"
    current.mkdir()
    (tmp_path / "base_data").mkdir()

    assert resolve_climbmix_data_dir(tmp_path) == current.resolve()


def test_climbmix_data_dir_falls_back_to_legacy_layout(tmp_path):
    legacy = tmp_path / "base_data"
    legacy.mkdir()

    assert resolve_climbmix_data_dir(tmp_path) == legacy.resolve()
