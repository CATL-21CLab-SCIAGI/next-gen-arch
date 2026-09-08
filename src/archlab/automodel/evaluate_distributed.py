"""Concurrent DLC capability evaluation using the existing FSDP/EP backend.

Run this file with torchrun and the checkpoint's immutable training source on
PYTHONPATH. A separate process group loads an immutable adapter checkpoint; it
never reads live training parameters or restores an optimizer. All ranks make
matching forward calls, including exhausted lanes, without padding scored text.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp


def _evaluation_module():
    path = Path(__file__).with_name("evaluate.py")
    spec = importlib.util.spec_from_file_location("archlab_distributed_evaluation_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluation = _evaluation_module()


def evaluation_rounds(selected: list[dict], world_size: int) -> list[list[dict]]:
    """Keep MC tasks separate; batch all compatible math tasks into shared lanes."""
    if world_size < 1 or len({row["id"] for row in selected}) != len(selected):
        raise ValueError("positive world size and unique example IDs required")
    supported = {"mmlu", "arc_challenge", "gsm8k", "aime24", "aime25"}
    if any(row["task"] not in supported for row in selected):
        raise ValueError("unsupported benchmark task")
    rounds = []
    for tasks in ({"mmlu"}, {"arc_challenge"}, {"gsm8k", "aime24", "aime25"}):
        rows = [row for row in selected if row["task"] in tasks]
        rounds.extend(rows[start:start + world_size] for start in range(0, len(rows), world_size))
    return rounds


@torch.no_grad()
def restore_distributed_adapters(adapters: dict, checkpoint: Path) -> str:
    """Reshard only FP32 adapter weights through DCP with strict key coverage."""
    from archlab.automodel.checkpointing import state_digest
    from archlab.automodel.loading import assert_loaded_weights_finite, poison_weights_before_load

    container = torch.nn.ModuleDict(adapters)
    poison_weights_before_load(container)
    payload = {"adapters": {name: module.state_dict() for name, module in adapters.items()}}
    expected = {f"adapters.{layer}.{name}": tensor
                for layer, state in payload["adapters"].items() for name, tensor in state.items()}
    saved = dcp.FileSystemReader(checkpoint / "state").read_metadata().state_dict_metadata
    if {name for name in saved if name.startswith("adapters.")} != expected.keys():
        raise ValueError("adapter checkpoint key coverage mismatch")
    for name, tensor in expected.items():
        if saved[name].size != tensor.shape or saved[name].properties.dtype != tensor.dtype:
            raise ValueError(f"adapter checkpoint shape/dtype mismatch: {name}")
    dcp.load(payload, checkpoint_id=checkpoint / "state")
    for name, module in adapters.items():
        module.load_state_dict(payload["adapters"][name], strict=True)
    assert_loaded_weights_finite(container)
    return state_digest(payload["adapters"])


def build_evaluation_model(base: Path, checkpoint: Path, ep_size: int):
    """Reuse the production sharded loader, native PLE ownership and EP groups."""
    from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextConfig
    from torch.distributed.fsdp import fully_shard

    from archlab.architectures.simplicial_adapter import SimplicialAdapterConfig
    from archlab.automodel.checkpointing import read_training_checkpoint
    from archlab.automodel.execution import build_frozen_base, emit
    from archlab.automodel.simplicial import install_simplicial_modules

    metadata = json.loads((checkpoint / "COMPLETE.json").read_text())
    read_training_checkpoint(checkpoint, metadata["contract"])
    for name, expected in metadata["contract"]["pretrained_metadata_sha256"].items():
        if evaluation._capability.file_sha256(base / name) != expected:
            raise ValueError(f"pretrained metadata changed: {name}")
    config = Qwen3_8_FlashNextConfig.from_pretrained(base, local_files_only=True, language_model_only=True)
    model, mesh, precision = build_frozen_base(config, tiny=False, checkpoint=base, ep_size=ep_size,
                                              activation_checkpointing=False)
    adapter_config = SimplicialAdapterConfig(**metadata["contract"]["adapter"])
    adapters = install_simplicial_modules(model, adapter_config, device="cuda", dtype=torch.float32)
    for adapter in adapters.values():
        fully_shard(adapter, mesh=mesh.device_mesh["dp_shard_cp"], mp_policy=precision, reshard_after_forward=True)
    digest = restore_distributed_adapters(adapters, checkpoint)
    model.requires_grad_(False)
    model.eval()
    emit("evaluation_model_ready", checkpoint_step=metadata["cursor"], local_adapter_sha256=digest,
         allocated_bytes=torch.cuda.memory_allocated(), free_bytes=torch.cuda.mem_get_info()[0])
    return model, adapters, digest


def main() -> None:
    from nemo_automodel.components.moe.megatron.fused_a2a import free_buffer

    from archlab.automodel.checkpointing import state_digest, write_json
    from archlab.automodel.execution import emit
    from archlab.automodel.runtime import configure_frozen_gdn_runtime, runtime_provenance

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "checkpoint", "data", "recipe", "prompts", "harness", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.gpu_memory_fraction <= 0.35 or args.ep_size < 1:
        parser.error("concurrent evaluation requires a positive memory fraction <= .35 and positive EP size")
    prepared = evaluation.prepare_benchmark(args)
    recipe, config, data_manifest, selected, tokenizer, rendered, grader, eos_ids = prepared
    if args.preflight_only:
        evaluation._sampling.emit("distributed_benchmark_preflight_passed", selected=len(selected),
                                  planned_rounds=len(evaluation_rounds(selected, int(os.environ.get("WORLD_SIZE", 32)))))
        return
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    lock = None
    try:
        if world % args.ep_size:
            raise ValueError("world size must be divisible by EP size")
        runtime = runtime_provenance(ep_size=args.ep_size)
        metadata = json.loads((args.checkpoint / "COMPLETE.json").read_text())
        expected_runtime = metadata["contract"]["runtime"]
        for key in ("packages", "cuda", "container_image"):
            if runtime[key] != expected_runtime[key]:
                raise RuntimeError(f"use the checkpoint's frozen DLC runtime: {key}")
        gdn_runtime = configure_frozen_gdn_runtime()
        provenance = evaluation._provenance(args.checkpoint, args.harness,
            environment="existing frozen DLC NeMo container; separate concurrent FSDP/EP evaluator; no installs")
        free = torch.tensor(torch.cuda.mem_get_info()[0], device="cuda", dtype=torch.int64)
        dist.all_reduce(free, op=dist.ReduceOp.MIN)
        if free.item() < 96 * 2**30:
            raise MemoryError("concurrent evaluation requires at least 96 GiB free on every GPU")
        rounds = evaluation_rounds(selected, world)
        contract = {"config": asdict(config), "recipe": recipe, "provenance": provenance,
                    "data_manifest": data_manifest, "selected_ids": [row["id"] for row in selected],
                    "prompt_file_sha256": evaluation._capability.file_sha256(args.prompts),
                    "grader_source_sha256": grader.provenance,
                    "tokenizer_chat_template_sha256": evaluation._capability.file_sha256(args.base / "chat_template.jinja"),
                    "eos_ids": sorted(eos_ids), "execution": {"world_size": world, "ep_size": args.ep_size,
                        "tensor_parallel": 1, "pipeline_parallel": 1, "context_parallel": 1,
                        "gpu_memory_fraction": args.gpu_memory_fraction, "runtime": runtime,
                        "gdn_runtime": gdn_runtime, "rounds": [[r["id"] for r in group] for group in rounds],
                        "mode_order": "alternates by global round, identical on all ranks",
                        "dummy_lanes": "repeat a real prompt for collectives; never scored or persisted",
                        "source_sha256": evaluation._capability.file_sha256(Path(__file__))}}
        # All ranks validate identical inputs before any large model allocation.
        protocol_hash = evaluation._sampling.sha256(args.recipe)
        hashes = [None] * world
        dist.all_gather_object(hashes, (protocol_hash, data_manifest["cases_sha256"], contract["selected_ids"]))
        if any(value != hashes[0] for value in hashes):
            raise ValueError("evaluation rank input contracts differ")
        records = []
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=args.resume)
            lock = (args.output / "writer.lock").open("a")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.resume:
                if json.loads((args.output / "contract.json").read_text()) != contract:
                    raise ValueError("resume contract changed")
                with (args.output / "pairs.jsonl").open() as stream:
                    records = [json.loads(line) for line in stream]
            else:
                write_json(args.output / "contract.json", contract)
            if len({r["id"] for r in records}) != len(records):
                raise ValueError("duplicate persisted pairs")
            write_json(args.output / "summary.json", evaluation._summary(records, selected, complete=False))
        done = [{r["id"] for r in records} if rank == 0 else None]
        dist.broadcast_object_list(done)
        if not done[0] <= set(contract["selected_ids"]):
            raise ValueError("persisted pairs are outside the registered subset")
        model, adapters, before_digest = build_evaluation_model(args.base, args.checkpoint, args.ep_size)
        digests = [None] * world
        dist.all_gather_object(digests, before_digest)
        if rank == 0:
            write_json(args.output / "model.json", {"checkpoint_step": metadata["cursor"],
                       "local_adapter_sha256": digests, "world_size": world, "ep_size": args.ep_size,
                       "backbone_and_adapters_frozen": True, "optimizer_created": False})
        for round_index, group in enumerate(rounds):
            pending = [row for row in group if row["id"] not in done[0]]
            if not pending:
                continue
            active = rank < len(pending)
            row = pending[rank % len(pending)]
            context, choices = rendered[row["id"]]
            result = {key: row[key] for key in ("id", "task", "subject")}
            result["target"] = row["answer"]
            modes = ("pretrained", "adapted") if round_index % 2 == 0 else ("adapted", "pretrained")
            emit("evaluation_round_begin", round=round_index, example=row["id"], active=active,
                 actual_pairs=len(pending), modes=modes)
            for mode in modes:
                started = time.monotonic()
                with evaluation.added_modules(model, enabled=mode == "adapted"):
                    if choices is not None:
                        output = evaluation.continuation_scores(model, tokenizer, context, choices,
                            max_context=config.max_context, synchronize=True)
                        output["metrics"] = {"acc": int(output["prediction"] == row["answer"])}
                        if row["task"] == "arc_challenge":
                            output["metrics"]["acc_norm"] = int(output["normalized_prediction"] == row["answer"])
                    else:
                        output = evaluation.math_completion(model, tokenizer, context, config=config,
                                                             eos_ids=eos_ids, synchronize=True)
                        output.update(grader.score(row, output["completion"]))
                output["seconds"] = time.monotonic() - started
                result[mode] = output
                emit("evaluation_mode_complete", example=row["id"], mode=mode, active=active,
                     seconds=output["seconds"], metrics=output["metrics"])
            gathered = [None] * world if rank == 0 else None
            dist.gather_object(result if active else None, gathered, dst=0)
            if rank == 0:
                with (args.output / "pairs.jsonl").open("a") as stream:
                    for record in gathered:
                        if record is not None:
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                            records.append(record)
                    stream.flush()
                    os.fsync(stream.fileno())
                write_json(args.output / "summary.json", evaluation._summary(records, selected, complete=False))
                emit("evaluation_round_complete", completed=len(records), total=len(selected))
            dist.barrier()
        after = state_digest({name: module.state_dict() for name, module in adapters.items()})
        unchanged = torch.tensor(int(after == before_digest), device="cuda")
        dist.all_reduce(unchanged, op=dist.ReduceOp.MIN)
        if not unchanged.item() or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("evaluation modified adapter weights or enabled gradients")
        if rank == 0:
            if {r["id"] for r in records} != set(contract["selected_ids"]):
                raise RuntimeError("benchmark did not produce every selected pair exactly once")
            write_json(args.output / "summary.json", evaluation._summary(records, selected, complete=True))
            emit("benchmark_complete", output=str(args.output), pairs=len(records), adapters_unchanged=True)
    finally:
        if lock is not None:
            lock.close()
        free_buffer()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
