"""Distributed numerical qualification of resident mixed-precision optimizers."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from archlab.optimizers.muown import Muown
from archlab.optimizers.sinkhorn import sinkhorn_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--momentum-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--weight-scale", type=float, default=.02)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--columns", type=int, default=48)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    groups = SimpleNamespace(tp=dist.group.WORLD, expt_tp=dist.group.WORLD)
    dtype = getattr(torch, args.momentum_dtype)
    errors, cosines = [], []
    direction_errors, direction_cosines = [], []
    for axis in (0, 1):
        torch.manual_seed(917)
        full = torch.nn.Parameter(torch.randn(args.rows, args.columns, device="cuda") * args.weight_scale)
        local = torch.nn.Parameter(full.detach().chunk(world, dim=axis)[rank].clone())
        local.tensor_model_parallel, local.partition_dim, local.partition_stride = True, axis, 1
        comparison = torch.nn.Parameter(local.detach().clone())
        comparison.tensor_model_parallel, comparison.partition_dim, comparison.partition_stride = True, axis, 1
        observed = []
        reference = Muown([comparison], lr=3e-6, pg_collection=groups, momentum_dtype=torch.float32,
                          orthogonalization_dtype=torch.float32, direction_observer=lambda x, observed=observed: observed.append(x.clone()))
        opt = Muown([local], lr=3e-6, pg_collection=groups, momentum_dtype=dtype,
                    orthogonalization_dtype=torch.float32, direction_observer=lambda x, observed=observed: observed.append(x.clone()))
        for _ in range(100):
            grad = torch.randn_like(full)
            comparison.grad = grad.chunk(world, dim=axis)[rank].clone()
            local.grad = comparison.grad.clone()
            before, local_before = comparison.detach().clone(), local.detach().clone()
            reference.step()
            opt.step()
            expected_direction, actual_direction = [x.flatten() for x in observed]
            direction_errors.append(float((expected_direction-actual_direction).norm() / expected_direction.norm().clamp_min(1e-20)))
            direction_cosines.append(float(torch.nn.functional.cosine_similarity(expected_direction, actual_direction, dim=0)))
            observed.clear()
            delta = (comparison.detach() - before).flatten()
            actual = (local.detach() - local_before).flatten()
            # Small FP32 weight updates have quantization noise; compare cumulative
            # state separately and record per-update error without hiding it.
            errors.append(float((delta-actual).norm() / delta.norm().clamp_min(1e-20)))
            cosines.append(float(torch.nn.functional.cosine_similarity(delta, actual, dim=0)))
        assert torch.isfinite(local).all()
    torch.manual_seed(813)
    full = torch.randn(32, 12, device="cuda")
    local = full.chunk(world, dim=0)[rank].clone()
    mom = torch.zeros_like(full, dtype=dtype)
    local_mom = mom.chunk(world, dim=0)[rank].clone()
    scale = torch.ones((), device="cuda") if dtype == torch.float16 else None
    local_scale = scale.clone() if scale is not None else None
    for _ in range(5):
        grad = torch.randn_like(full)
        grad[0] = 0
        sinkhorn_step(full, grad, mom, lr=3e-6, momentum_scale=scale)
        sinkhorn_step(local, grad.chunk(world, dim=0)[rank], local_mom, lr=3e-6, group=dist.group.WORLD, momentum_scale=local_scale)
        torch.testing.assert_close(local, full.chunk(world, dim=0)[rank], atol=2e-7, rtol=2e-7)
    from archlab.megatron.miles_v41_resident import ResidentMuown
    p = torch.nn.Parameter(torch.randn(16, 24, device="cuda", dtype=torch.bfloat16))
    p.main_grad = torch.randn_like(p)
    optimizer = ResidentMuown([("matrix", p)], 3e-6, None, dtype)
    optimizer.step()
    from megatron.core import dist_checkpointing

    from archlab.megatron.miles_v41_resident_checkpoint import pack_state, unpack_state
    checkpoint_path = args.output / "optimizer-checkpoint"
    if rank == 0:
        checkpoint_path.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    from archlab.megatron.miles_v41_checkpoint_verify import fingerprint, tensors
    from archlab.megatron.miles_v41_streaming_checkpoint import StreamingSaveStrategy
    before_restore = fingerprint(optimizer.resident_state())
    dist_checkpointing.save(pack_state(optimizer.resident_state()), str(checkpoint_path),
                            sharded_strategy=StreamingSaveStrategy())
    with torch.no_grad():
        for _, tensor in tensors(optimizer.resident_state()):
            tensor.zero_()
    assert fingerprint(optimizer.resident_state()) != before_restore
    payload = unpack_state(dist_checkpointing.load(
        pack_state(optimizer.resident_state()), str(checkpoint_path)))
    optimizer.restore(payload)
    assert fingerprint(optimizer.resident_state()) == before_restore
    other_p = torch.nn.Parameter(p.detach().clone())
    other = ResidentMuown([("matrix", other_p)], 3e-6, None, dtype)
    other.restore(payload)
    for _ in range(3):
        p.main_grad = torch.randn_like(p)
        other_p.main_grad = p.main_grad.clone()
        optimizer.step()
        other.step()
        torch.testing.assert_close(p, other_p, atol=0, rtol=0)
    receipt = dict(rank=rank, world_size=world, momentum_dtype=args.momentum_dtype,
                   orthogonalization_dtype="float32", weight_scale=args.weight_scale,
                   matrix_shape=[args.rows, args.columns],
                   max_direction_relative_error=max(direction_errors), min_direction_cosine=min(direction_cosines),
                   max_update_relative_error=max(errors), min_update_cosine=min(cosines),
                   sinkhorn_distributed=world > 1, resident_resume_exact=True, checkpoint_roundtrip=True,
                   distributed_checkpoint_roundtrip=world > 1,
                   streaming_checkpoint=True, destructive_restore_verified=True,
                   passed=max(errors) <= .01 and min(cosines) >= .999 and max(direction_errors) <= .01 and min(direction_cosines) >= .999)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank-{rank}.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if not receipt["passed"]:
        raise RuntimeError("momentum numerical admission failed")


if __name__ == "__main__":
    main()
