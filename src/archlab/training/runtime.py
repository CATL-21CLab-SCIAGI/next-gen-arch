"""Launch upstream verl with the same environment on local and external Ray heads."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


def worker_setup():
    """Load private settings on each Ray worker without placing values in config."""
    if path := os.environ.get("ARCHLAB_MLFLOW_CONFIG"):
        tracking = json.loads(Path(path).read_text())
        for variable, key in (("MLFLOW_TRACKING_URI", "tracking_uri"), ("MLFLOW_TRACKING_TOKEN", "token")):
            os.environ[variable] = tracking[key]
    if hook := os.environ.get("ARCHLAB_WORKER_SETUP_HOOK"):
        module, _, name = hook.rpartition(".")
        getattr(importlib.import_module(module), name)()


def launch_config(config):
    """Inject private tracking settings only into an in-memory launch copy.

    Source selection belongs to the qualified snapshot's launch environment.
    Ray workers must inherit that PYTHONPATH when joining an existing head.
    No credentials are written into the resolved configuration or receipts.
    """
    runtime = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    variables = runtime.ray_kwargs.ray_init.runtime_env.env_vars
    for key in ("PYTHONPATH", "NO_PROXY", "no_proxy"):
        if key in os.environ:
            variables[key] = os.environ[key]
    if path := os.environ.get("ARCHLAB_MLFLOW_CONFIG"):
        environment = runtime.ray_kwargs.ray_init.runtime_env
        variables["ARCHLAB_MLFLOW_CONFIG"] = str(Path(path).resolve())
        if hook := environment.get("worker_process_setup_hook"):
            variables["ARCHLAB_WORKER_SETUP_HOOK"] = hook
        environment.worker_process_setup_hook = "archlab.training.runtime.worker_setup"
        # Upstream prints Ray kwargs; only a path, never a credential, is sent.
        tracking = json.loads(Path(path).read_text())
        os.environ["MLFLOW_TRACKING_URI"] = tracking["tracking_uri"]
        os.environ["MLFLOW_TRACKING_TOKEN"] = tracking["token"]
    return runtime


def pinned_to_driver_node(actor_class):
    """Place an actor on the node that creates it.

    On an external Ray head, CPU agent pods join the cluster and are evictable. Losing the TaskRunner there ends
    the run; on the GPU head it survives pod loss (the rollout controller is pinned the same way, uni-agent 0007).
    """

    class Pinned:
        @staticmethod
        def remote(*args, **kwargs):
            import ray

            strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                ray.get_runtime_context().get_node_id(), soft=False)
            return actor_class.options(scheduling_strategy=strategy).remote(*args, **kwargs)

    return Pinned


def run(config):
    from verl.trainer import main_ppo

    config = launch_config(config)
    if config.ray_kwargs.ray_init.get("address") not in (None, "local"):
        main_ppo.TaskRunnerV1 = pinned_to_driver_node(main_ppo.TaskRunnerV1)
    return main_ppo.main(config)


if __name__ == "__main__":  # python -m archlab.training.runtime <resolved.yaml>
    import sys

    raise SystemExit(run(OmegaConf.load(sys.argv[1])))
