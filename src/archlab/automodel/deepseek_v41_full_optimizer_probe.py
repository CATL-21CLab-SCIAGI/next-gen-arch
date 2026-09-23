# ruff: noqa: I001  # Preserve the checkpoint-qualified import order.
"""Distributed Adafactor oracle for dense/Engram rows and EP/FSDP expert banks."""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from archlab.optimizers.sharded_adafactor import ShardedAdafactor


def qualify_distributed_optimizer():
    world = dist.get_world_size()
    rows = init_device_mesh("cuda", (world,), mesh_dim_names=("optimizer_rows",))
    experts = init_device_mesh(
        "cuda", (world // 8, 8), mesh_dim_names=("optimizer_fsdp", "optimizer_ep")
    )
    results = []
    for shape, mesh, placements in [
        ((19,), rows, (Shard(0),)),
        ((33, 7), rows, (Shard(0),)),
        ((16, 18, 7), experts, (Shard(1), Shard(0))),
    ]:
        generator = torch.Generator(device="cpu").manual_seed(97)
        initial = torch.randn(shape, generator=generator).cuda()
        actual = torch.nn.Parameter(distribute_tensor(initial, mesh, placements))
        expected = torch.nn.Parameter(initial.clone())
        opt = ShardedAdafactor([actual], lr=0.01, stochastic_rounding=False, chunk_elements=21)
        # This ordinary tensor path has independent storage and no collectives;
        # its equations are also checked against the scalar CPU oracle tests.
        reference = ShardedAdafactor(
            [expected], lr=0.01, stochastic_rounding=False, chunk_elements=10**9
        )
        errors = []
        for step in range(3):  # noqa: B007 — preserve checkpoint-qualified executable AST
            grad = torch.randn(shape, generator=generator).cuda()
            actual.grad = distribute_tensor(grad, mesh, placements)
            expected.grad = grad.clone()
            opt.step()
            reference.step()
            observed = actual.full_tensor()
            torch.testing.assert_close(observed, expected, rtol=3e-6, atol=3e-7)
            errors.append(float((observed - expected).abs().max()))
        results.append({"shape": shape, "placements": str(placements), "max_abs_errors": errors})
    return {"passed": True, "cases": results}
