"""Bounded NCCL/FSDP policy-head and distributed-rollout numerical oracle."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1]
    tested_files = (
        "automodel/deepseek_v41_rl_fsdp_probe.py",
        "automodel/deepseek_v41_rl_head.py",
        "rl/rollout.py",
        "rl/weight_residency.py",
        "automodel/deepseek_v41_rl_update.py",
    )
    tested_source = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in tested_files
    }
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from archlab.automodel.deepseek_v41_rl_head import install_rl_head
    from archlab.rl.rollout import sample_rollouts

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=10),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(72)

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(257, 32, device="cuda")
            self.lm_head = nn.Linear(32, 257, bias=False, device="cuda")

        def forward(self, input_ids, attention_mask=None, return_hidden_states=True):
            from types import SimpleNamespace

            return SimpleNamespace(hidden_states=self.embedding(input_ids))

    try:
        model = Policy()
        original = model.lm_head.weight.detach().clone().requires_grad_()
        mesh = init_device_mesh("cuda", (world,))
        fully_shard(model.lm_head, mesh=mesh, reshard_after_forward=True)
        model.lm_head.set_gradient_divide_factor(1.0)
        install_rl_head(model)
        generator = torch.Generator(device="cuda").manual_seed(500 + rank)
        hidden = torch.randn(2, 7, 32, device="cuda", generator=generator, requires_grad=True)
        targets = torch.randint(0, 257, (2, 7), device="cuda", generator=generator)
        targets[:, :2] = -100
        coefficients = torch.randn(2, 7, device="cuda", generator=generator) / world
        actual = model.lm_head.rl_log_probs(hidden, targets, chunk_size=3)
        dense_hidden = hidden.detach().clone().requires_grad_()
        dense = torch.nn.functional.linear(dense_hidden, original).log_softmax(-1)
        expected = dense.gather(-1, targets.clamp_min(0)[..., None]).squeeze(-1)
        expected = expected.masked_fill(targets == -100, 0)
        (expected * coefficients).sum().backward()
        (actual * coefficients).sum().backward()
        dist.all_reduce(original.grad)
        head_gradient = model.lm_head.weight.grad.full_tensor()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(hidden.grad, dense_hidden.grad, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(head_gradient, original.grad, rtol=3e-5, atol=3e-6)
        model.zero_grad(set_to_none=True)
        rollout = sample_rollouts(
            model,
            [[3, 5, 7 + rank], [11, 13, 17]],
            policy_version="synthetic-qualification-only",
            max_new_tokens=4,
            context_limit=32,
            eos_token_ids={256},
            pad_token_id=0,
            seed=701,
            bucket_multiple=8,
            prompt_group_ids=[f"rank-{rank}-first", f"rank-{rank}-second"],
        )
        retained = sample_rollouts(
            model,
            [[3, 5, 7 + rank], [11, 13, 17]],
            policy_version="synthetic-qualification-only",
            max_new_tokens=4,
            context_limit=32,
            eos_token_ids={256},
            pad_token_id=0,
            seed=701,
            bucket_multiple=8,
            prompt_group_ids=[f"rank-{rank}-first", f"rank-{rank}-second"],
            retain_weights=True,
        )
        if retained.generated_ids != rollout.generated_ids:
            raise AssertionError("retained weights changed sampled tokens")
        torch.testing.assert_close(
            retained.policy_log_probs, rollout.policy_log_probs, atol=0, rtol=0
        )
        if not retained.receipt["weight_residency"]["cleanup_verified"]:
            raise AssertionError("retained weights were not reshared")
        replay_hidden = model(input_ids=rollout.input_ids).hidden_states
        replay = model.lm_head.rl_log_probs(replay_hidden, rollout.labels, chunk_size=3)
        mask = rollout.response_mask
        error = float((replay[mask].detach() - rollout.policy_log_probs[mask]).abs().max())
        if error > 2e-6:
            raise AssertionError(f"sampled/replayed token scores differ: {error}")
        del replay, replay_hidden
        model.zero_grad(set_to_none=True)
        from archlab.automodel.deepseek_v41_rl_update import policy_gradient_step

        grouped = sample_rollouts(
            model,
            [[3, 5, 7 + rank]] * 4,
            policy_version="prefix-qualification-only",
            max_new_tokens=4,
            context_limit=32,
            eos_token_ids={256},
            pad_token_id=0,
            seed=703,
            bucket_multiple=8,
            retain_weights=True,
        )
        model._archlab_rl_policy_version = "prefix-qualification-only"
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        audit = policy_gradient_step(
            model,
            optimizer,
            [],
            grouped,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
            lr=0.001,
            group_size=4,
            audit=True,
            audit_only=True,
            replay_mode="sampled-prefix",
            replay_prefixes=4,
            replay_seed=88,
        )
        if not audit.get("numerical_qualification_passed") or optimizer.state:
            raise AssertionError("prefix-gradient audit failed or changed optimizer")
        record = {
            "rank": rank,
            "passed": True,
            "gradient_reference": "dense-FP32-global-sum",
            "signed_coefficients": True,
            "masked_targets": True,
            "rollout_replay_max_abs_error": error,
            "rollout": rollout.receipt,
            "retained_rollout": retained.receipt,
            "prefix_gradient_audit": audit,
        }
        records = [None] * world
        dist.all_gather_object(records, record)
        if rank == 0:
            if tested_source != {
                name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                for name in tested_files
            }:
                raise RuntimeError("qualified source changed while probe was running")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "world_size": world,
                        "scope": "synthetic component qualification; no training-policy update",
                        "implementation_sha256": tested_source,
                        "weight_residency_qualified": True,
                        "sampled_prefix_policy_gradient_qualified": True,
                        "numerical_precision": {
                            "allow_tf32": False,
                            "allow_bf16_reduced_precision_reduction": False,
                        },
                        "container_image": os.environ.get("NGA_CONTAINER_DIGEST"),
                        "torch": importlib.metadata.version("torch"),
                        "cuda": torch.version.cuda,
                        "nccl": torch.cuda.nccl.version(),
                        "ranks": records,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps({"passed": True, "world_size": world, "output": str(args.output)}))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
