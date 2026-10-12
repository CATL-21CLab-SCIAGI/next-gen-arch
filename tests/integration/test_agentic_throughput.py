"""Fixed-workload, stock Megatron train_batch timing, outside production state."""
import importlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.slow


def test_package_learner_throughput(tmp_path):
    if os.getenv("ARCHLAB_RUN_THROUGHPUT") != "1":
        pytest.skip("requires the deployment's free GPU world under torchrun")
    from importlib.metadata import version

    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from test_megatron_agentic_training import deployment_engine_arguments

    importlib.import_module("archlab.training.megatron")
    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    from archlab.artifacts import atomic_write_json, sha256_file

    config = OmegaConf.load(os.environ["ARCHLAB_QUALIFICATION_CONFIG"])
    experiment = json.loads(Path(os.environ["ARCHLAB_QUALIFICATION_EXPERIMENT"]).read_text())
    # Keep the harness usable against the frozen pre-restructure package too.
    specification = SimpleNamespace(**experiment["throughput"])
    assert specification.sequence_length >= 2 and specification.warmup >= 1 and specification.samples >= 3
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == config.trainer.nnodes * config.trainer.n_gpus_per_node
    engine = MegatronEngineWithLMHead(*deployment_engine_arguments(config))
    engine.initialize()
    ids = 1 + (torch.arange(specification.sequence_length, device=device) + 17 * rank) % 511
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[-1] = False
    batch = TensorDict({
        "input_ids": torch.nested.nested_tensor([ids], layout=torch.jagged),
        "loss_mask": torch.nested.nested_tensor([mask], layout=torch.jagged),
        "temperature": torch.tensor([config.actor_rollout_ref.rollout.temperature], device=device),
    }, batch_size=[1])
    tu.assign_non_tensor(batch, use_dynamic_bsz=False, micro_batch_size_per_gpu=1,
                         calculate_entropy=config.actor_rollout_ref.actor.calculate_entropy)

    def loss_function(model_output, data, dp_group):
        selected = data["loss_mask"].values().float()
        loss = -(model_output["log_probs"].values() * selected).sum() / selected.sum()
        return loss, {}

    seconds = []
    for index in range(specification.warmup + specification.samples):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with engine.train_mode():
            result = engine.train_batch(batch, loss_function)
            engine.lr_scheduler_step()
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter() - start, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        # Exclude JIT/cache/allocator warmup; record the slowest actual rank.
        if index >= specification.warmup:
            seconds.append(float(elapsed))
        assert result is not None
    if rank == 0:
        model = Path(config.actor_rollout_ref.model.path)
        native = json.loads((model / "EXPORT.json").read_text())
        settings = vars(specification).copy()
        settings.pop("baseline")
        settings.pop("max_slowdown")
        contract = dict(
            scope="stock distributed learner update including offload; excludes generation/tools",
            workload="arange-mod511-rank17-v1", settings=settings, world_size=world,
            export_files=native["files"],
            actor=OmegaConf.to_container(config.actor_rollout_ref.actor, resolve=True),
            temperature=config.actor_rollout_ref.rollout.temperature,
            runtime={name: version(name) for name in ("torch", "megatron-core", "transformer-engine", "tilelang")},
            cuda=torch.version.cuda, device_capability=list(torch.cuda.get_device_capability()),
            effective_hardware="NVIDIA B300", reported_hardware=torch.cuda.get_device_name(),
            container_identity=os.environ["ARCHLAB_CONTAINER_IDENTITY"],
            benchmark_sha256=sha256_file(Path(__file__)),
        )
        atomic_write_json(tmp_path / "throughput.json", dict(
            contract=contract, seconds=seconds,
            model_tokens_per_update=world * specification.sequence_length,
            source_module=str(importlib.import_module("archlab.training.megatron").__file__),
        ))
    dist.barrier()
    dist.destroy_process_group()
