"""Bounded torchrun qualification of skewed graph work before NCCL admission."""

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from archlab.rl.grpo_log_collection import bind_host_object_gather
from archlab.rl.limite_checkpoint import capture_rng, checkpoint_payloads
from archlab.rl.rollout_rendezvous import RolloutRendezvous


def qualify_trl_binding(rendezvous):
    """Check the installed method's actual globals/decorators without a model."""
    import trl
    from trl import GRPOTrainer

    original = GRPOTrainer._generate_and_score_completions
    bound = bind_host_object_gather(original, rendezvous.gather_object)
    wrappers = 0
    while hasattr(original, "__wrapped__"):
        assert bound.__code__ is original.__code__
        original, bound = original.__wrapped__, bound.__wrapped__
        wrappers += 1
    assert bound.__code__ is original.__code__
    assert bound.__globals__ is not original.__globals__
    collector = bound.__globals__["gather_object"]
    assert collector.__self__ is rendezvous
    assert collector.__func__ is RolloutRendezvous.gather_object
    assert original.__globals__["gather_object"] is not collector
    return dict(
        passed=True, trl=trl.__version__, method=original.__qualname__,
        original_transport_module=original.__globals__["gather_object"].__module__,
        bytecode_unchanged=True, vendor_globals_unchanged=True, profiling_wrappers=wrappers,
    )


def _log_payload(rank, iteration):
    # Every nonempty text exceeds the failed 48,680-byte NCCL log payload;
    # varying lengths and empty ranks exercise serialization and flattening.
    text = f"rank={rank}, iteration={iteration}\n" + "数学 🧮 café\n" * (10000 + rank * 31)
    return dict(
        prompt=[f"prompt {rank}"],
        completion=[] if rank % 3 == 2 else [text],
        extra=[] if rank % 2 == 0 else [dict(rank=rank, nested=[iteration])],
    )


def qualify_object_logs(rendezvous, rank, world, iteration):
    local = _log_payload(rank, iteration)
    for key in sorted(local):
        expected = [item for peer in range(world) for item in _log_payload(peer, iteration)[key]]
        assert rendezvous.gather_object(local[key]) == expected
    return max(len(item.encode("utf-8")) for item in _log_payload(0, iteration)["completion"])


@dataclass
class _CheckpointState:
    global_step: int = 190


def qualify_checkpoint_objects(rendezvous, rank, world):
    """Exercise the production checkpoint gather with a large pending batch."""
    before = capture_rng()
    local_rollout = dict(
        generator=torch.get_rng_state(),
        pending=dict(rank=rank, prompts=[f"rank {rank}"],
                     batch=dict(completion_ids=[list(range(16384 + rank * 13))])),
    )
    trainer = SimpleNamespace(
        state=_CheckpointState(),
        lr_scheduler=SimpleNamespace(state_dict=lambda: dict(last_epoch=190)),
        archlab_rollout_rendezvous=rendezvous,
        archlab_async_rollout=SimpleNamespace(checkpoint_state=lambda: local_rollout),
    )
    payloads = checkpoint_payloads(trainer, dict(applied_updates=157))
    saved = payloads["rl_state.pt"]
    assert saved["world_size"] == world
    for peer, state in enumerate(saved["rank_rng"]):
        assert set(state).issuperset({"python", "numpy", "cpu", "rollout"})
        assert state["rollout"]["pending"]["rank"] == peer
        assert state["rollout"]["pending"]["batch"]["completion_ids"] == [
            list(range(16384 + peer * 13))
        ]
    assert torch.equal(saved["rank_rng"][rank]["rollout"]["generator"], local_rollout["generator"])
    assert saved["scheduler"] == dict(last_epoch=190)
    assert saved["evidence"] == dict(applied_updates=157)
    after = capture_rng()
    assert before["python"] == after["python"] and before["numpy"] == after["numpy"]
    for name in ("cpu", "cuda"):
        if name in before:
            assert torch.equal(before[name], after[name])
    return dict(passed=True, world=world, intact_rank_objects=True,
                pending_batches_retained=True, rng_unchanged_by_transport=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, local, world = (int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    rendezvous = RolloutRendezvous(world)
    trl_binding = qualify_trl_binding(rendezvous)
    from archlab.automodel.limite_adapter_common import runtime_contract

    runtime = runtime_contract()
    stream = torch.cuda.Stream()
    source = torch.ones(128, device=f"cuda:{local}")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for _ in range(3):
            result = source.square() + 1
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            result = source.square() + 1
    stream.synchronize()
    args.output.mkdir(parents=True, exist_ok=True)
    elapsed = []
    object_bytes = []
    for iteration in range(3):
        started = time.monotonic()
        with torch.cuda.stream(stream):
            for _ in range(128 if rank == iteration % world else 4):
                graph.replay()
                assert result.sum().item() == 256
                if rank == iteration % world:
                    time.sleep(0.002)
        stream.synchronize()
        (args.output / f"drained-{iteration}-{rank}").write_text("done\n")
        rendezvous()
        assert all((args.output / f"drained-{iteration}-{peer}").exists() for peer in range(world))
        object_bytes.append(qualify_object_logs(rendezvous, rank, world, iteration))
        checkpoint_objects = qualify_checkpoint_objects(rendezvous, rank, world)
        value = torch.tensor([rank + 1.0], device=f"cuda:{local}")
        dist.all_reduce(value)
        assert value.item() == world * (world + 1) / 2
        elapsed.append(time.monotonic() - started)
    (args.output / f"rank-{rank}.json").write_text(json.dumps(dict(
        passed=True, rank=rank, world=world, iterations=3, elapsed_seconds=elapsed,
        torch=torch.__version__, cuda=torch.version.cuda, hardware="NVIDIA B300",
        nccl=torch.cuda.nccl.version(), container=os.environ.get("ARCHLAB_CONTAINER_IMAGE"),
        source_revision=os.environ.get("ARCHLAB_SOURCE_REVISION"),
        runtime=runtime,
        trl_binding=trl_binding, object_log_bytes=object_bytes,
        checkpoint_objects=checkpoint_objects, object_transport="gloo",
        scope="skewed graph drain, host admission, installed TRL binding, repeated large object logs and checkpoint RNG, then NCCL; not a full-model numeric test",
    ), indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
