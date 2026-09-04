"""Retained CUDA probes for the repository-owned Flash-Next mechanisms."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from archlab.architectures.qwen38_flash_next_full import (
    DistributedPLE,
    FourStreamGatedResidual,
    GatedDeltaNet,
    OwnerShardedPLEEmbedding,
    Qwen38FlashNextFullConfig,
    ple_partition_ownership,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _require_finite_gradients(module: torch.nn.Module) -> int:
    count = 0
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError("a probe parameter has a missing or non-finite gradient")
        count += parameter.numel()
    return count


def mechanism_probe(output: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the mechanism probe requires one CUDA GPU")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    device = torch.device("cuda", 0)
    config = Qwen38FlashNextFullConfig.tiny(sequence_len=64)
    attention_residual = FourStreamGatedResidual(config).to(device=device, dtype=torch.bfloat16)
    gdn = GatedDeltaNet(config).to(device=device, dtype=torch.bfloat16)
    mlp_residual = FourStreamGatedResidual(config).to(device=device, dtype=torch.bfloat16)
    ple = DistributedPLE(config, owner_rank=0, owner_world_size=1).to(
        device=device, dtype=torch.bfloat16
    )
    ple.embedding.reset_parameters()
    modules = (attention_residual, gdn, mlp_residual, ple)

    packed = torch.randn(
        config.sequence_len,
        1,
        config.residual_streams * config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    tokens = torch.randint(
        0, config.vocab_size, (1, config.sequence_len), device=device
    )
    packed = packed + ple(tokens, packed)
    mixed, residual, injection = attention_residual(packed)
    packed = FourStreamGatedResidual.inject(residual, gdn(mixed), injection)
    mixed, residual, injection = mlp_residual(packed)
    packed = FourStreamGatedResidual.inject(residual, mixed, injection)
    loss = packed.float().square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("the mechanism probe produced a non-finite loss")
    loss.backward()
    gradient_parameters = sum(_require_finite_gradients(module) for module in modules)
    if gdn.in_proj_qkv.weight.grad is None or not gdn.in_proj_qkv.weight.grad.any():
        raise RuntimeError("the early GDN backbone gradient is zero")
    _atomic_json(
        output,
        {
            "status": "passed",
            "probe": "single_gpu_gdn_gr_ple",
            "loss": loss.item(),
            "gradient_parameters": gradient_parameters,
            "device": torch.cuda.get_device_name(0),
            "completed_at_unix": time.time(),
        },
    )


def _set_partition_values(embedding: OwnerShardedPLEEmbedding) -> None:
    width = embedding.embedding_dim
    rows = embedding.rows_per_partition
    for partition, table in zip(embedding.global_partitions, embedding.tables, strict=True):
        values = torch.arange(table.numel(), device=table.device, dtype=table.dtype)
        table.data.copy_(values + partition * rows * width)


def ple_probe(output: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the PLE probe requires CUDA GPUs")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if world != 8:
        raise RuntimeError(f"the PLE probe requires exactly 8 ranks, got {world}")
    device = torch.device("cuda", local_rank)
    config = Qwen38FlashNextFullConfig.tiny(ngram_partitions=128)
    embedding = OwnerShardedPLEEmbedding(
        config,
        owner_rank=rank,
        owner_world_size=world,
        process_group=dist.group.WORLD,
    ).to(device=device, dtype=torch.bfloat16)
    _set_partition_values(embedding)
    row = rank % config.ngram_rows_per_partition
    partitions = torch.arange(config.ngram_partitions, device=device)
    global_ids = partitions * config.ngram_rows_per_partition + row
    output_values = embedding(global_ids)
    expected = global_ids.unsqueeze(-1) * config.ngram_branch_dim + torch.arange(
        config.ngram_branch_dim, device=device
    )
    torch.testing.assert_close(output_values.float(), expected.to(torch.bfloat16).float())
    output_values.float().sum().backward()
    expected_gradient_sum = world * config.ngram_branch_dim
    for table in embedding.tables:
        if table.grad is None or not torch.isfinite(table.grad).all():
            raise RuntimeError("an owner PLE shard has a missing or non-finite gradient")
        if int(table.grad.sum().item()) != expected_gradient_sum:
            raise RuntimeError("an owner PLE shard received the wrong remote gradient count")
    optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-3, weight_decay=0.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ownership = ple_partition_ownership(config.ngram_partitions, world)
    local_result = {
        "rank": rank,
        "partitions": list(embedding.global_partitions),
        "parameters": sum(parameter.numel() for parameter in embedding.parameters()),
    }
    gathered: list[dict[str, Any] | None] = [None] * world
    dist.all_gather_object(gathered, local_result)
    if rank == 0:
        if tuple(tuple(item["partitions"]) for item in gathered if item is not None) != ownership:
            raise RuntimeError("the gathered PLE ownership map is incorrect")
        _atomic_json(
            output,
            {
                "status": "passed",
                "probe": "eight_gpu_ple_all_to_all_adam",
                "world_size": world,
                "partitions": config.ngram_partitions,
                "ownership": [list(value) for value in ownership],
                "ranks": gathered,
                "completed_at_unix": time.time(),
            },
        )
    dist.barrier()
    dist.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("mechanism", "ple"))
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite probe evidence: {args.output}")
    if args.mode == "mechanism":
        mechanism_probe(args.output)
    else:
        ple_probe(args.output)


if __name__ == "__main__":
    main()
