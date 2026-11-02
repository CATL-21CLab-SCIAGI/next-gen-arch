"""Launch upstream verl with the same environment on local and external Ray heads."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import sys
import time
import uuid
from pathlib import Path

from filelock import FileLock
from omegaconf import OmegaConf

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.failures import classify_failure
from archlab.training.admission import verify_admission
from archlab.training.resume import claim_run, resumed_config


def flash_attn_namespace():
    """Resolve ``flash_attn.cute`` to the container's FA4 wheel before engines import it.

    The wheel installs ``flash_attn/cute`` beside the container's FA2 ``flash_attn``
    package, and Transformer Engine imports it whenever the wheel is installed.
    """
    try:
        distribution = importlib.metadata.distribution("flash-attn-4")
    except importlib.metadata.PackageNotFoundError:
        return
    root = Path(distribution.locate_file("flash_attn")).resolve()
    imported = getattr(sys.modules.get("flash_attn.cute"), "__file__", None)
    if imported is not None and not Path(imported).resolve().is_relative_to(root):
        raise RuntimeError(f"flash_attn.cute was already imported from {imported}")
    namespace = importlib.import_module("flash_attn")
    paths = [str(path) for path in namespace.__path__]
    if paths[:1] != [str(root)]:
        namespace.__path__ = [str(root), *(path for path in paths if path != str(root))]


def worker_setup():
    """Load private settings on each Ray worker without placing values in config."""
    if path := os.environ.get("ARCHLAB_MLFLOW_CONFIG"):
        tracking = json.loads(Path(path).read_text())
        for variable, key in (("MLFLOW_TRACKING_URI", "tracking_uri"), ("MLFLOW_TRACKING_TOKEN", "token")):
            os.environ[variable] = tracking[key]
    if hook := os.environ.get("ARCHLAB_WORKER_SETUP_HOOK"):
        module, _, name = hook.rpartition(".")
        getattr(importlib.import_module(module), name)()
    if os.environ.get("ARCHLAB_PIN_TRANSFER_QUEUE") == "1":
        pin_transfer_queue_storage()


def pin_transfer_queue_storage():
    """Create TransferQueue storage units on the node of the process that creates them.

    Upstream places them in a SPREAD placement group, so on an external head they land on evictable,
    memory-limited CPU agent pods. A starved pod then stalls every batch fetch (27B 27b-rev, 2026-10-28:
    "ZMQ recv timeout (200s) from storage unit"). The creator is the TaskRunner, which is pinned to the GPU head.
    """
    try:
        from transfer_queue.storage.bootstrap import simple_storage_bootstrap
    except ImportError:
        return

    def get_placement_group(num_ray_actors: int, num_cpus_per_actor: int = 1):
        import ray

        here = f"node:{ray.util.get_node_ip_address()}"
        group = ray.util.placement_group([{"CPU": num_cpus_per_actor, here: 0.001} for _ in range(num_ray_actors)],
                                         strategy="STRICT_PACK")
        ray.get(group.ready())
        return group

    simple_storage_bootstrap.get_placement_group = get_placement_group


def launch_config(config):
    """Inject private tracking settings only into an in-memory launch copy.

    Source selection belongs to the qualified snapshot's launch environment.
    Ray workers must inherit that PYTHONPATH when joining an existing head.
    No credentials are written into the resolved configuration or receipts.
    """
    runtime = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    variables = runtime.ray_kwargs.ray_init.runtime_env.env_vars
    for key in ("PYTHONPATH", "NO_PROXY", "no_proxy", "ARCHLAB_PROGRESS_FILE"):
        if key in os.environ:
            variables[key] = os.environ[key]
    if path := os.environ.get("ARCHLAB_MLFLOW_CONFIG"):
        environment = runtime.ray_kwargs.ray_init.runtime_env
        variables["ARCHLAB_MLFLOW_CONFIG"] = str(Path(path).resolve())
        if hook := environment.get("worker_process_setup_hook"):
            variables["ARCHLAB_WORKER_SETUP_HOOK"] = hook
        environment.worker_process_setup_hook = "archlab.training.runtime.worker_setup"
        if runtime.ray_kwargs.ray_init.get("address") not in (None, "local"):
            variables["ARCHLAB_PIN_TRANSFER_QUEUE"] = "1"
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


def run(config, *, qualification: Path, attempt: Path | None = None, resume: Path | None = None):
    """The CLI and supervisor share this gate and the same per-run process lock."""
    qualification = Path(qualification).resolve()
    attempt = Path(attempt) if attempt else qualification / "attempts" / uuid.uuid4().hex
    attempt.mkdir(parents=True, exist_ok=True)
    status = attempt / "result.json"
    if status.exists():
        raise FileExistsError("attempt results are immutable; use a new attempt directory")
    # O_EXCL prevents two invocations from publishing into the same attempt.
    with (attempt / ".started").open("x"):
        pass
    atomic_write_json(status, dict(status="admitting", at=time.time()))
    try:
        report = verify_admission(config, qualification)
        root = Path(config.trainer.default_local_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(root / ".archlab-trainer.lock", timeout=0):
            identity = dict(config_sha256=report["config_sha256"],
                            qualification_sha256=sha256_file(qualification / "qualification.json"))
            claim_run(root, identity, resume=resume)
            if resume is not None:
                config = resumed_config(config, resume)
            OmegaConf.save(config, attempt / "resolved.yaml", resolve=True)
            identity["attempt_config_sha256"] = sha256_file(attempt / "resolved.yaml")
            from verl.trainer import main_ppo

            previous = main_ppo.TaskRunnerV1
            old_progress = os.environ.get("ARCHLAB_PROGRESS_FILE")
            os.environ["ARCHLAB_PROGRESS_FILE"] = str((attempt / "progress.json").resolve())
            try:
                config = launch_config(config)
                if config.ray_kwargs.ray_init.get("address") not in (None, "local"):
                    main_ppo.TaskRunnerV1 = pinned_to_driver_node(previous)
                atomic_write_json(status, dict(status="running", at=time.time(), **identity))
                result = main_ppo.main(config)
                atomic_write_json(status, dict(status="succeeded", at=time.time(), **identity))
                return result
            finally:
                main_ppo.TaskRunnerV1 = previous
                if old_progress is None:
                    os.environ.pop("ARCHLAB_PROGRESS_FILE", None)
                else:
                    os.environ["ARCHLAB_PROGRESS_FILE"] = old_progress
    except Exception as error:
        atomic_write_json(status, dict(status="failed", at=time.time(),
                                       failure=classify_failure(error).to_dict()))
        raise


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    return run(OmegaConf.load(args.config), qualification=args.qualification,
               attempt=args.attempt, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
