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


def test_actors_are_pinned_to_the_creating_node(monkeypatch):
    import sys
    from types import SimpleNamespace

    from archlab.training.runtime import pinned_to_driver_node

    class Strategy:
        def __init__(self, node, soft):
            self.node, self.soft = node, soft

    ray = SimpleNamespace(
        get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: "gpu-head"),
        util=SimpleNamespace(scheduling_strategies=SimpleNamespace(NodeAffinitySchedulingStrategy=Strategy)))
    monkeypatch.setitem(sys.modules, "ray", ray)

    class Actor:
        @staticmethod
        def options(scheduling_strategy):
            return SimpleNamespace(remote=lambda *args, **kwargs: (scheduling_strategy, args, kwargs))

    strategy, args, kwargs = pinned_to_driver_node(Actor).remote(1, key=2)
    assert (strategy.node, strategy.soft, args, kwargs) == ("gpu-head", False, (1,), {"key": 2})


def test_transfer_queue_storage_is_packed_onto_the_creating_node(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from archlab.training import runtime

    bootstrap = ModuleType("simple_storage_bootstrap")
    bootstrap.get_placement_group = lambda n, num_cpus_per_actor=1: "spread"
    package = ModuleType("transfer_queue.storage.bootstrap")
    package.simple_storage_bootstrap = bootstrap
    created = {}

    def placement_group(bundles, strategy):
        created.update(bundles=bundles, strategy=strategy)
        return SimpleNamespace(ready=lambda: None)

    ray = SimpleNamespace(get=lambda ref: None, util=SimpleNamespace(get_node_ip_address=lambda: "10.0.0.1",
                                                                     placement_group=placement_group))
    for name, module in (("transfer_queue", ModuleType("transfer_queue")), ("transfer_queue.storage", ModuleType("s")),
                         ("transfer_queue.storage.bootstrap", package),
                         ("transfer_queue.storage.bootstrap.simple_storage_bootstrap", bootstrap), ("ray", ray)):
        monkeypatch.setitem(sys.modules, name, module)
    runtime.pin_transfer_queue_storage()
    bootstrap.get_placement_group(2)
    assert created == {"bundles": [{"CPU": 1, "node:10.0.0.1": 0.001}] * 2, "strategy": "STRICT_PACK"}


def test_only_external_heads_pin_transfer_queue(tmp_path, monkeypatch):
    path = tmp_path / "tracking.json"
    path.write_text('{"tracking_uri":"uri","token":"t"}')
    monkeypatch.setenv("ARCHLAB_MLFLOW_CONFIG", str(path))
    for address, pinned in (("auto", True), ("local", False)):
        config = OmegaConf.create({"ray_kwargs": {"ray_init": {"address": address, "runtime_env": {"env_vars": {}}}}})
        variables = launch_config(config).ray_kwargs.ray_init.runtime_env.env_vars
        assert (variables.get("ARCHLAB_PIN_TRANSFER_QUEUE") == "1") == pinned
