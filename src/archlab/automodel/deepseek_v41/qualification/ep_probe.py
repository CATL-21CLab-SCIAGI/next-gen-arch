"""Distributed numerical probe for EP routing and row-sharded Engram lookups.

Synthetic weights only. Run with torchrun; this is not a finetuning entrypoint.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--container-kernel-packages", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

    from archlab.automodel.deepseek_v41_autograd import install_frozen_backward
    from archlab.automodel.deepseek_v41_parallel import expert_parallel_classes
    from archlab.automodel.deepseek_v41_runtime import (
        load_native_reference,
        select_container_kernel_packages,
    )

    packages = select_container_kernel_packages(args.container_kernel_packages)
    reference = load_native_reference(args.checkpoint, module_name="_archlab_v41_ep_reference")
    install_frozen_backward(reference)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    reference.default_dtype = torch.bfloat16
    ep_moe, ep_engram = expert_parallel_classes(reference, dist.group.WORLD)
    config = reference.ModelArgs(dim=64, moe_inter_dim=64, n_routed_experts=size * 2,
                                 n_activated_experts=2, expert_dtype=None, dtype="bf16",
                                 swiglu_limit=10.)
    oracle = reference.MoE(0, config)
    torch.manual_seed(998)
    with torch.no_grad():
        for p in oracle.parameters():
            p.copy_(torch.randn_like(p) * .1)
    oracle.requires_grad_(False)
    candidate = ep_moe(0, config)
    expected_state = candidate.state_dict()
    candidate.load_state_dict({k: v for k, v in oracle.state_dict().items() if k in expected_state}, strict=True)
    candidate.requires_grad_(False)
    results = []
    for forced_empty in (False, True):
        if forced_empty:
            with torch.no_grad():
                for model in (oracle, candidate):
                    model.gate.bias.fill_(-100)
                    model.gate.bias[:2].fill_(100)
        torch.manual_seed(111 + rank)
        x = torch.randn(1, 5 + rank, 64, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        expected = oracle(x)
        with set_checkpoint_early_stop(False):
            actual = checkpoint(candidate, other, use_reentrant=False)
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.002)
        dy = torch.randn_like(actual)
        expected.backward(dy)
        actual.backward(dy)
        torch.testing.assert_close(other.grad, x.grad, rtol=.05, atol=.003)
        assert all(p.grad is None for p in candidate.parameters())
        results.append({"empty_receivers": forced_empty,
                        "forward_max_abs": (actual - expected).abs().max().item(),
                        "gradient_max_abs": (other.grad - x.grad).abs().max().item()})

    rows, dim = size * 33 + 1, 32
    table = reference.ParallelEngramEmbedding(rows, dim)
    sharded = ep_engram(rows, dim)
    with torch.no_grad():
        torch.manual_seed(880)
        table.weight.copy_(torch.randn(rows, dim, dtype=torch.float32).to(table.weight.dtype))
        table.scale.copy_(torch.ones(rows, 1, dtype=torch.float32).to(table.scale.dtype))
        first = rank * sharded.rows
        count = min(sharded.rows, rows - first)
        sharded.weight.zero_()
        sharded.scale.copy_(torch.ones_like(sharded.scale, dtype=torch.float32).to(sharded.scale.dtype))
        sharded.weight[:count].copy_(table.weight[first:first + count])
        sharded.scale[:count].copy_(table.scale[first:first + count])
    for ids in (torch.tensor([[0, rows - 1, rank, rank + 1]]), torch.zeros(1, 7, dtype=torch.int64)):
        torch.testing.assert_close(sharded(ids), table(ids), atol=0, rtol=0)
    report = {"rank": rank, "world_size": size, "passed": True, "training_launched": False,
              "kind": "synthetic-EP-and-Engram-qualification", "cases": results,
              "activation_checkpointing": True,
              "engram_lookup_exact": True, "kernel_packages": packages,
              "torch": torch.__version__, "container": os.environ.get("NGA_CONTAINER_DIGEST")}
    with Path(f"{args.output_prefix}-rank{rank}.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
