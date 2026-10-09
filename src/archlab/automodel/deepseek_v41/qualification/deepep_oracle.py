"""Compare upstream DeepEP dispatch gradients with the qualified EP gather path."""

import argparse
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(Path(os.environ["ARCHLAB_CONTAINER_KERNEL_PACKAGES"]))
    import torch
    import torch.distributed as dist
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from archlab.automodel.deepseek_v41_official_moe import _fp32_grouped_experts_forward

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    torch.manual_seed(319)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    config = SimpleNamespace(
        n_routed_experts=32,
        n_activated_experts=2,
        expert_bias=False,
        expert_activation="swiglu",
        swiglu_limit=7.0,
        expert_dim=128,
        moe_inter_dim=128,
        dtype=torch.bfloat16,
        apply_router_weight_after_down=False,
    )
    backend = BackendConfig(experts="torch_mm", dispatcher="deepep")
    reference = GroupedExperts(config, backend)
    reference._archlab_scratch_grouped_gemm = True
    reference.forward = MethodType(_fp32_grouped_experts_forward, reference)
    actual = GroupedExpertsDeepEP(config, backend)
    for name, shape in [("gate_and_up_projs", (32, 128, 256)), ("down_projs", (32, 128, 128))]:
        value = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 0.02
        for module in (reference, actual):
            setattr(
                module, name, torch.nn.Parameter(distribute_tensor(value.clone(), mesh, [Shard(0)]))
            )
    actual.init_token_dispatcher(mesh)
    rank = dist.get_rank()
    reports = []
    try:
        for empty_owners in (False, True):
            length = 33 + rank
            x = torch.randn(length, 128, device="cuda", dtype=torch.bfloat16) * 0.1
            weights = torch.rand(length, 2, device="cuda")
            indices = torch.randint(0, 16 if empty_owners else 32, (length, 2), device="cuda")
            indices[:, 1] = (indices[:, 0] + 1) % (16 if empty_owners else 32)
            mask = torch.arange(length, device="cuda") < (0 if rank == 7 else length - 3)
            results = []
            for module in (reference, actual):
                module.zero_grad(set_to_none=True)
                xx, ww = (
                    x.detach().clone().requires_grad_(),
                    weights.detach().clone().requires_grad_(),
                )
                y = module(xx, mask, ww, indices).bfloat16()
                y.float().square().sum().backward()
                results.append(
                    [
                        y.detach(),
                        xx.grad,
                        ww.grad,
                        module.gate_and_up_projs.grad.to_local(),
                        module.down_projs.grad.to_local(),
                    ]
                )
            errors = []
            for got, want in zip(results[1], results[0], strict=True):
                torch.testing.assert_close(got, want, atol=3e-5, rtol=0.04)
                relative = float(
                    (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-20)
                )
                assert relative < 0.025, relative
                errors.append(relative)
            reports.append({"empty_owners": empty_owners, "relative_l2_errors": errors})
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank-{rank:02d}.json").write_text(
            json.dumps({"passed": True, "cases": reports}, indent=2)
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
