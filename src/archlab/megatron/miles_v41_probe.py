"""Two-GPU numerical admission for TP Muown and immutable disk snapshots."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from archlab.optimizers.muown import Muown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    pg = SimpleNamespace(tp=dist.group.WORLD, expt_tp=dist.group.WORLD)
    errors = []
    for axis in (0, 1):
        torch.manual_seed(37)
        full = torch.nn.Parameter(torch.randn(8, 12, device="cuda"))
        local = torch.nn.Parameter(full.detach().chunk(world, dim=axis)[rank].clone())
        local.tensor_model_parallel, local.partition_dim, local.partition_stride = True, axis, 1
        reference, optimizer = Muown([full], lr=1e-3), Muown([local], lr=1e-3, pg_collection=pg)
        for _ in range(3):
            gradient = torch.randn_like(full)
            full.grad = gradient
            local.grad = gradient.chunk(world, dim=axis)[rank].clone()
            optimizer.step()
            reference.step()
            expected = full.detach().chunk(world, dim=axis)[rank]
            errors.append(float((local - expected).abs().max()))
            torch.testing.assert_close(local, expected, atol=2e-5, rtol=2e-5)
    from archlab.megatron.miles_muown_streaming import DiskMuown
    from archlab.megatron.miles_v41_model import TrainableEngram

    torch.manual_seed(19)
    parameter = torch.nn.Parameter(torch.randn(7, 11, device="cuda", dtype=torch.bfloat16))
    oracle = torch.nn.Parameter(parameter.float().detach().clone())
    reference = Muown([oracle], lr=1e-3)
    root = args.output / f"rank-{rank}"
    stream = DiskMuown([("matrix", parameter)], root / "live", 1e-3, None)
    for step in range(2):
        parameter.main_grad = torch.randn_like(parameter)
        oracle.grad = parameter.main_grad.float()
        reference.step()
        stream.step()
        torch.testing.assert_close(parameter, oracle.to(parameter.dtype), rtol=0, atol=0)
        if step == 0:
            receipt = stream.snapshot(root / "snapshot")
            filename = root / "snapshot" / receipt["files"][0]
            digest = hashlib.sha256(filename.read_bytes()).hexdigest()
    assert hashlib.sha256(filename.read_bytes()).hexdigest() == digest
    # A row-sharded lookup must preserve both the summed value and owner gradients.
    table = torch.randn(18, 4, device="cuda", requires_grad=True)
    dist.broadcast(table.detach(), src=0)
    layer = TrainableEngram.__new__(TrainableEngram)
    torch.nn.Module.__init__(layer)
    layer.row_start, layer.rows_avail, layer.tp_group = rank * 9, 9, dist.group.WORLD
    layer.table = torch.nn.Parameter(table.detach()[rank * 9:(rank + 1) * 9].clone())
    ids = torch.tensor([0, 9, 2, 17, 9], device="cuda")
    actual = layer.lookup(ids)
    expected = torch.nn.functional.embedding(ids, table)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(layer.table.grad, table.grad[rank * 9:(rank + 1) * 9])
    root.mkdir(parents=True, exist_ok=True)
    (root / "PASSED.json").write_text(json.dumps({"tp_muown_max_error": max(errors),
                                                "streaming_master_exact": True,
                                                "snapshot_immutable": True,
                                                "engram_value_and_gradient": True}))
    dist.barrier()
    if rank == 0:
        print("DISTRIBUTED_MUOWN_AND_ENGRAM_PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
