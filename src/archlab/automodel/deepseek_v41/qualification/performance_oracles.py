"""GPU numerical oracles for grouped experts and row-sharded Sinkhorn updates."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(Path(os.environ["ARCHLAB_CONTAINER_KERNEL_PACKAGES"]))
    import torch
    import torch.distributed as dist
    from nemo_automodel.components.moe.experts import GroupedExperts, swiglu_clamped_deepep
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from archlab.automodel.deepseek_v41_official_moe import _native_up_grouped_down
    from archlab.optimizers.sharded_adafactor import ShardedAdafactor
    from archlab.optimizers.sinkhorn import EngramSinkhornAdafactor, sinkhorn_direction

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    torch.manual_seed(81)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    reports = {}
    try:
        update = torch.randn(139, 6, device="cuda")
        update[:3] = 0
        update[3] *= 1e-6
        update[:, -1] = 0
        expected = update.clone()
        rho = expected.norm(dim=1)
        expected[rho <= 1e-3 * rho.mean()] = 0
        for k in range(11):
            expected /= expected.norm(dim=1 if k % 2 == 0 else 0, keepdim=True) + 1e-20
        expected *= 6**0.5
        shard = distribute_tensor(update, mesh, [Shard(0)])
        actual = sinkhorn_direction(shard.to_local(), global_rows=139, group=mesh.get_group())
        reference = distribute_tensor(expected, mesh, [Shard(0)]).to_local()
        torch.testing.assert_close(actual, reference, atol=3e-6, rtol=3e-6)
        reports["sinkhorn_max_error"] = float((actual - reference).abs().max())
        # Uneven row shards, zero rows/columns, and persistent FP32 momentum.
        params = [
            torch.nn.Parameter(
                distribute_tensor(torch.randn(139, 6, device="cuda"), mesh, [Shard(0)])
            )
            for _ in range(2)
        ]
        ordinary = torch.nn.Parameter(torch.randn(13, device="cuda"))
        opt = EngramSinkhornAdafactor(
            [
                ("a.engram.embed.weight", params[0]),
                ("b.engram.embed.weight", params[1]),
                ("norm", ordinary),
            ],
            batched_updates=True,
        )
        for p in params:
            p.grad = shard.clone()
        ordinary.grad = torch.ones_like(ordinary)
        opt.step()
        assert all(opt.state[p]["momentum"].dtype == torch.float32 for p in params)
        reports["sinkhorn_optimizer_finite"] = all(
            bool(torch.isfinite(p.to_local()).all()) for p in params
        )
        # Batched Adafactor matches individual expert factorization on row shards.
        bank = torch.randn(7, 35, 13, device="cuda")
        p = torch.nn.Parameter(distribute_tensor(bank, mesh, [Shard(1)]))
        q = torch.nn.Parameter(bank.clone())
        fast = ShardedAdafactor(
            [p], batched_updates=True, stochastic_rounding=False, chunk_elements=1024
        )
        slow = ShardedAdafactor([q], stochastic_rounding=False)
        for _ in range(3):
            g = torch.randn_like(bank)
            p.grad = distribute_tensor(g, mesh, [Shard(1)])
            q.grad = g.clone()
            fast.step()
            slow.step()
        torch.testing.assert_close(p.full_tensor(), q, atol=3e-7, rtol=3e-6)
        reports["batched_adafactor_passed"] = True
        # Upstream grouped computation against the separate expert reference,
        # including input/router/weight gradients and an empty expert owner.
        owner = SimpleNamespace(
            config=SimpleNamespace(swiglu_limit=7.0, apply_router_weight_after_down=False),
            expert_bias=False,
            use_mxfp8=False,
        )
        owner._forward_grouped_mm = GroupedExperts._forward_grouped_mm.__get__(owner)
        owner.expert_activation_grouped = lambda x, probs: swiglu_clamped_deepep(x, probs, 7.0)
        rows, experts, width, middle = 73, 4, 120, 128
        indices = torch.randint(0, experts, (rows, 2), device="cuda")
        indices[:, 1] = (indices[:, 0] + 1) % experts
        mask = torch.arange(rows, device="cuda") < 67
        values = [
            torch.randn(rows, width, device="cuda", dtype=torch.bfloat16) * 0.1,
            torch.rand(rows, 2, device="cuda"),
            torch.randn(experts, width, middle * 2, device="cuda", dtype=torch.bfloat16) * 0.02,
            torch.randn(experts, middle, width, device="cuda", dtype=torch.bfloat16) * 0.02,
        ]
        errors = []
        for offset in (0, experts):
            outputs, gradients = [], []
            for grouped in (False, True):
                owner._archlab_scratch_grouped_gemm = grouped
                x, weights, up, down = [v.detach().clone().requires_grad_() for v in values]
                y = _native_up_grouped_down(
                    owner, x.float(), mask, weights, indices, up, down, experts, offset
                )
                gs = torch.autograd.grad(y.square().sum(), (x, weights, up, down))
                outputs.append(y.detach())
                gradients.append(gs)
            torch.testing.assert_close(outputs[1], outputs[0], atol=3e-5, rtol=0.02)
            for a, b in zip(gradients[1], gradients[0], strict=True):
                torch.testing.assert_close(a, b, atol=3e-5, rtol=0.04)
            errors.append(float((outputs[1] - outputs[0]).abs().max()))
        reports["grouped_max_errors"] = errors
        from archlab.automodel.deepseek_v41_full_training import reduce_replicated_gradients

        gradients = [torch.randn(17, 11, device="cuda", dtype=dtype).T for dtype in
                     (torch.float32, torch.bfloat16, torch.float32)]
        for gradient in gradients:
            gradient.mul_(dist.get_rank() + 1)
        expected_gradients = [gradient.contiguous() for gradient in gradients]
        for gradient in expected_gradients:
            dist.all_reduce(gradient)
        reduce_replicated_gradients(gradients)
        for actual, expected in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        reports["batched_replicated_reduction_passed"] = True
        reports["passed"] = True
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank-{dist.get_rank():02d}.json").write_text(json.dumps(reports, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
