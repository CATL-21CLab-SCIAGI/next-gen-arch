"""Private settings are loaded from shared files on workers, never printed in Ray config."""

import json
import os

from omegaconf import OmegaConf

from archlab.training.runtime import launch_config, worker_setup


def test_external_ray_receives_source_and_private_config_path_without_credentials(tmp_path, monkeypatch):
    path = tmp_path / "tracking.json"
    path.write_text(json.dumps({"tracking_uri": "https://tracking.example.invalid", "token": "private-test-token"}))
    monkeypatch.setenv("ARCHLAB_MLFLOW_CONFIG", str(path))
    monkeypatch.setenv("PYTHONPATH", "/immutable/src:/immutable/third_party/verl")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "old-private-token")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "old-uri")
    config = OmegaConf.create({"ray_kwargs": {"ray_init": {"address": "auto", "runtime_env": {"env_vars": {}}}}})
    new = launch_config(config)
    variables = new.ray_kwargs.ray_init.runtime_env.env_vars
    assert variables.PYTHONPATH == os.environ["PYTHONPATH"]
    assert variables.ARCHLAB_MLFLOW_CONFIG == str(path)
    assert new.ray_kwargs.ray_init.runtime_env.worker_process_setup_hook == "archlab.training.runtime.worker_setup"
    assert "private-test-token" not in OmegaConf.to_yaml(new)
    assert "MLFLOW_TRACKING_TOKEN" not in variables
    assert config.ray_kwargs.ray_init.runtime_env.env_vars == {}
    assert os.environ["MLFLOW_TRACKING_TOKEN"] == "private-test-token"
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "worker-old-token")
    monkeypatch.delenv("ARCHLAB_WORKER_SETUP_HOOK", raising=False)
    worker_setup()
    assert os.environ["MLFLOW_TRACKING_TOKEN"] == "private-test-token"


def test_prior_worker_setup_is_preserved(tmp_path, monkeypatch):
    path = tmp_path / "tracking.json"
    path.write_text('{"tracking_uri":"uri","token":"test-token"}')
    monkeypatch.setenv("ARCHLAB_MLFLOW_CONFIG", str(path))
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "old-token")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "old-uri")
    config = OmegaConf.create({"ray_kwargs": {"ray_init": {"runtime_env": {
        "env_vars": {}, "worker_process_setup_hook": "archlab.architectures.fa4_attention.fa4_runtime_contract"}}}})
    new = launch_config(config)
    assert new.ray_kwargs.ray_init.runtime_env.env_vars.ARCHLAB_WORKER_SETUP_HOOK == (
        "archlab.architectures.fa4_attention.fa4_runtime_contract")
