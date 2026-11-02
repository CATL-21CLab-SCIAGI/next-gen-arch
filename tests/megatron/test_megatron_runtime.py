from importlib.machinery import ModuleSpec

from archlab.megatron import runtime as megatron_backend


def test_megatron_runtime_resolves_system_package(monkeypatch, tmp_path):
    package_path = tmp_path / "site-packages" / "megatron"
    package_path.mkdir(parents=True)
    pretrain_script = package_path.parent / "pretrain_gpt.py"
    pretrain_script.write_text("pass\n", encoding="utf-8")

    def fake_find_spec(name):
        spec = ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(package_path)]
        return spec

    monkeypatch.setattr(megatron_backend, "find_spec", fake_find_spec)
    versions = {
        "megatron-core": "0.13.0",
        "torch": "2.9.1",
        "transformer-engine": "2.11.0",
        "nemo-toolkit": "2.5.0",
    }
    monkeypatch.setattr(megatron_backend, "version", versions.__getitem__)
    monkeypatch.setattr(megatron_backend.platform, "python_version", lambda: "3.12.0")

    runtime = megatron_backend.validate_runtime()
    assert runtime == {
        "provider": "system",
        "distribution": "megatron-core",
        "version": "0.13.0",
        "package_path": str(package_path),
        "validated_container_profile": "nemo-26.06",
        "python": "3.12.0",
        "torch": "2.9.1",
        "transformer_engine": "2.11.0",
        "nemo_toolkit": "2.5.0",
        "pretrain_script": str(pretrain_script),
    }
