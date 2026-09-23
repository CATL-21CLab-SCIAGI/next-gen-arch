"""Native EP8/FSDP2 numerical probes and synthetic full-geometry memory checks."""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path


def _full_memory_actor(args, packages):
    """Exercise full parameter/activation geometry without reading checkpoint payloads."""
    import torch
    from nemo_automodel.components.moe.layers import Gate

    from archlab.automodel.deepseek_v41_rl_model import _construct_full_shell
    from archlab.optimizers.sharded_adafactor import local_tensor

    model, indexers, loading = _construct_full_shell(
        variant=args.variant,
        assets=Path(os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"]),
        weights=Path(os.environ["ARCHLAB_DEEPSEEK_V41_CHECKPOINT"]),
        tiny=False,
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = local_tensor(parameter)
            if name.endswith("lm_head.weight"):
                value.normal_(0, 0.001)
            elif ".ffn.gate." in name or name.endswith(".base"):
                value.zero_()
            elif name.endswith(".fn"):
                value.normal_(0, 0.001)
            elif value.ndim < 2:
                value.fill_(1 if "norm" in name or name.endswith(".scale") else 0)
            else:
                value.normal_(0, 0.01)
        gates = [module for module in model.modules() if isinstance(module, Gate)]
        if len(gates) != 40:
            raise ValueError("expected all forty full-geometry routers")
        for gate in gates:
            if gate.e_score_correction_bias is None:
                raise ValueError("full geometry needs the native routing bias for the stress test")
            bias = local_tensor(gate.e_score_correction_bias)
            if bias.numel() != gate.n_experts:
                raise ValueError("memory-probe routing bias must remain replicated")
            bias.zero_()
            bias[: gate.topk].fill_(1000)

    model._archlab_memory_routing_checks = 0

    def verify_owner(module, _inputs, output):
        indices = output[1]
        if not bool(((indices >= 0) & (indices < module.n_experts // 8)).all()):
            raise AssertionError("synthetic routing did not exercise one expert owner")
        model._archlab_memory_routing_checks += 1

    for gate in gates:
        gate.register_forward_hook(verify_owner)
    loading.update(
        resolved_kernel_packages=packages,
        actor_weight_origin="synthetic-memory-only",
        checkpoint_payloads_read=False,
        routing="all selected experts on EP owner zero",
    )
    return model, indexers, loading, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("normal", "simplicial"), default="normal")
    parser.add_argument("--cache-policy", action="store_true")
    parser.add_argument("--new-tokens", type=int)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--freeze-router", action="store_true")
    parser.add_argument("--checkpoint-input-offload", action="store_true")
    parser.add_argument("--inplace-moe-accumulation", action="store_true")
    parser.add_argument("--hc-activation-offload", action="store_true")
    parser.add_argument("--checkpoint-expert-activations", action="store_true")
    parser.add_argument("--serialize-backward-gathers", action="store_true")
    parser.add_argument("--unshard-on-compute-stream", action="store_true")
    parser.add_argument("--full-memory-audit", action="store_true")
    parser.add_argument("--memory-context", type=int, default=2048)
    parser.add_argument("--memory-budget-gib", type=float, default=198)
    parser.add_argument("--evaluation-reserve-gib", type=float, default=64)
    parser.add_argument(
        "--loss-normalization",
        choices=("sequence_sum", "prompt_token_mean"),
        default="sequence_sum",
    )
    args = parser.parse_args()
    if args.checkpoint_expert_activations and not args.inplace_moe_accumulation:
        parser.error("expert checkpoints require the native memory policy")
    if args.full_memory_audit and (
        args.cache_policy or not args.freeze_router or not 128 <= args.memory_context <= 4096
    ):
        parser.error(
            "full memory audit requires frozen routers, uncached execution and context >=128"
        )
    new_tokens = (
        args.new_tokens if args.new_tokens is not None else (64 if args.cache_policy else 2)
    )
    prompt_tokens = (
        args.prompt_tokens if args.prompt_tokens is not None else (31 if args.cache_policy else 3)
    )
    if min(new_tokens, prompt_tokens) < 1 or prompt_tokens + 2 + new_tokens > 1024:
        parser.error("probe token budgets must be positive and fit the bounded 1024-token context")
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    packages = select_container_kernel_packages(
        Path(os.environ["ARCHLAB_CONTAINER_KERNEL_PACKAGES"])
    )
    import torch
    import torch.distributed as dist

    from archlab.artifacts import atomic_write_json, sha256_file
    from archlab.automodel.deepseek_v41_rl_head import install_rl_head
    from archlab.automodel.deepseek_v41_rl_model import EXECUTION_CHANGES, construct_rl_actor
    from archlab.automodel.deepseek_v41_rl_training import source_identity
    from archlab.automodel.deepseek_v41_rl_update import policy_gradient_step
    from archlab.optimizers.sharded_adafactor import ShardedAdafactor
    from archlab.rl.rollout import sample_rollouts

    sources = {
        **source_identity(),
        "automodel/deepseek_v41_rl_native_probe.py": sha256_file(__file__),
    }

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # Qualification may share a node with a live actor. Bound allocator demand.
    if args.full_memory_audit:
        from archlab.automodel.deepseek_v41_rl_memory import configure_gpu_budget

        configure_gpu_budget(args.memory_budget_gib)
    else:
        torch.cuda.set_per_process_memory_fraction(0.02)
    torch.set_num_threads(4)
    torch.manual_seed(82)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=15),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    rank = dist.get_rank()
    try:
        if args.output.exists():
            raise FileExistsError("use a fresh qualification output")
        model, indexers, loading, _ = (
            _full_memory_actor(args, packages)
            if args.full_memory_audit
            else construct_rl_actor(
                checkpoint=None,
                family="full",
                variant=args.variant,
                tiny=True,
                assets=Path(os.environ["ARCHLAB_DEEPSEEK_V41_ASSETS"]),
                weights=Path(os.environ["ARCHLAB_DEEPSEEK_V41_CHECKPOINT"]),
                declared_execution_changes=EXECUTION_CHANGES,
                resolved_kernel_packages=packages,
            )
        )
        install_rl_head(model)
        from archlab.automodel.deepseek_v41_rl_memory import (
            input_offload_statistics,
            install_checkpoint_input_offload,
        )
        from archlab.automodel.deepseek_v41_rl_model import configure_rl_trainability

        strategy = configure_rl_trainability(model, freeze_router=args.freeze_router)
        from archlab.automodel.deepseek_v41_rl_memory_policy import (
            hc_offload_statistics,
            install_hc_activation_offload,
            install_inplace_moe_accumulation,
            serialize_backward_gathers,
            unshard_on_compute_stream,
        )

        if args.inplace_moe_accumulation:
            strategy["moe_accumulation"] = install_inplace_moe_accumulation(
                model, checkpoint_activations=args.checkpoint_expert_activations
            )
        if args.serialize_backward_gathers:
            strategy["backward_gathers"] = serialize_backward_gathers(model)
        if args.unshard_on_compute_stream:
            strategy["gather_allocation"] = unshard_on_compute_stream(model)
        if args.hc_activation_offload:
            strategy["hc_activations"] = install_hc_activation_offload(model)
        if args.checkpoint_input_offload:
            strategy["checkpoint_inputs"] = install_checkpoint_input_offload(model)
        if args.full_memory_audit:
            from archlab.automodel.deepseek_v41_rl_memory import qualify_replay_memory
            from archlab.rl.weight_residency import _state_plan

            if rank == 0:
                loading["fsdp_gather_groups"] = [
                    {"name": row["name"], "gather_buffer_gib": row["extra_bytes"] / 2**30}
                    for row in _state_plan(model)
                ]
                atomic_write_json(args.output.with_suffix(".layout.json"), loading, allow_nan=False)

            model._archlab_rl_policy_version = "synthetic-full-geometry-memory-only"
            optimizer = ShardedAdafactor(
                (p for p in model.parameters() if p.requires_grad), lr=1e-6
            )
            config = {
                "context_limit": args.memory_context,
                "group_size": 4,
                "seed": 891,
                "learning_rate": 1e-6,
                "replay_tolerance": 0.02,
                "loss_normalization": args.loss_normalization,
                "evaluation_reserve_gib": args.evaluation_reserve_gib,
                "gpu_memory_budget_gib": args.memory_budget_gib,
            }
            result = qualify_replay_memory(
                model,
                optimizer,
                indexers,
                [[100 + rank, 101, 102]] * 4,
                config=config,
                policy_version=model._archlab_rl_policy_version,
                pad=2,
            )
            if sources != {
                **source_identity(),
                "automodel/deepseek_v41_rl_native_probe.py": sha256_file(__file__),
            }:
                raise RuntimeError("probe source changed during the synthetic memory test")
            if rank == 0:
                atomic_write_json(
                    args.output,
                    {
                        **result,
                        "kind": "synthetic-full-geometry-memory-only-v1",
                        "scope": "Synthetic weights and worst-case single-owner routing; no quality evaluation or production admission",
                        "world_size": dist.get_world_size(),
                        "allocator_budget_gib": args.memory_budget_gib,
                        "variant": args.variant,
                        "implementation_sha256": sources,
                        "runtime": loading,
                        "strategy": strategy,
                        "verified_router_calls_rank0": model._archlab_memory_routing_checks,
                    },
                    allow_nan=False,
                )
            if not result["passed"]:
                raise AssertionError("synthetic full-geometry memory gate failed")
            return
        cache_equivalence = None
        if args.cache_policy:
            from archlab.automodel.deepseek_v41_rl_cache import qualify_resident_cache
            from archlab.optimizers.sharded_adafactor import local_tensor

            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "simplicial_adapter.output.weight" in name:
                        local_tensor(parameter).normal_(0, 0.001)
        prompts = [
            [100 + rank, *range(101, 100 + prompt_tokens + (rank % 3 if args.cache_policy else 0))]
        ] * 4
        context_limit = ((prompt_tokens + 2 + max(new_tokens, 16) + 127) // 128) * 128
        if args.cache_policy:
            cache_equivalence = qualify_resident_cache(
                model,
                prompts,
                pad_token_id=2,
                context_limit=context_limit,
                steps=16,
                tolerance=0.02,
            )
            atomic_write_json(
                args.output.with_name(f"{args.output.stem}-rank-{rank:02d}-prefix.json"),
                cache_equivalence,
                allow_nan=False,
            )
            if not cache_equivalence["passed"]:
                raise AssertionError(
                    "distributed resident cache disagrees with padded full prefixes"
                )
        common = dict(
            policy_version="native-random-qualification",
            max_new_tokens=new_tokens,
            context_limit=context_limit,
            eos_token_ids={1},
            pad_token_id=2,
            seed=891,
            prompt_group_ids=[f"qual-{rank}"] * 4,
        )
        first = sample_rollouts(model, prompts, retain_weights=args.cache_policy, **common)
        second = sample_rollouts(
            model, prompts, retain_weights=True, cache_policy=args.cache_policy, **common
        )
        if not args.cache_policy:
            if first.generated_ids != second.generated_ids:
                raise AssertionError("native retained weights changed sampled actions")
            torch.testing.assert_close(
                first.policy_log_probs, second.policy_log_probs, atol=2e-6, rtol=1e-6
            )
        if not second.receipt["weight_residency"]["cleanup_verified"]:
            raise AssertionError("native ownership cleanup failed")
        model._archlab_rl_policy_version = common["policy_version"]
        optimizer = ShardedAdafactor((p for p in model.parameters() if p.requires_grad), lr=1e-6)
        audit = policy_gradient_step(
            model,
            optimizer,
            indexers,
            second,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
            lr=1e-6,
            group_size=4,
            replay_mode="sampled-prefix",
            replay_prefixes=min(4, new_tokens),
            replay_seed=9,
            audit_only=True,
            audit=True,
            loss_normalization=args.loss_normalization,
        )
        if not audit.get("numerical_qualification_passed") or any(optimizer.state.values()):
            raise AssertionError("native gradient qualification failed or updated optimizer")
        offload_stats = input_offload_statistics(model)
        hc_stats = hc_offload_statistics(model)
        if args.hc_activation_offload and hc_stats["tensor_copies"] == 0:
            raise AssertionError("requested HC activation offload was not exercised")
        if args.checkpoint_input_offload and offload_stats["tensor_copies"] == 0:
            raise AssertionError("requested checkpoint activation offload was not exercised")
        records = [None] * dist.get_world_size()
        dist.all_gather_object(
            records,
            {
                "rank": rank,
                "passed": True,
                "audit": audit,
                "residency": second.receipt["weight_residency"],
                "cache_equivalence": cache_equivalence,
                "uncached_rollout": first.receipt,
                "candidate_rollout": second.receipt,
                "same_sampled_actions": first.generated_ids == second.generated_ids,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "input_offload": offload_stats,
                "hc_offload": hc_stats,
            },
        )
        if rank == 0:
            if sources != {
                **source_identity(),
                "automodel/deepseek_v41_rl_native_probe.py": sha256_file(__file__),
            }:
                raise RuntimeError("probe source changed during qualification")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "world_size": dist.get_world_size(),
                        "scope": "random tiny native V4.1; EP8/FSDP2; no optimizer update",
                        "variant": args.variant,
                        "cache_policy": args.cache_policy,
                        "strategy": strategy,
                        "loss_normalization": args.loss_normalization,
                        "implementation_sha256": sources,
                        "runtime": loading,
                        "ranks": records,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps({"passed": True, "output": str(args.output)}), flush=True)
    except BaseException:
        import traceback

        atomic_write_json(
            args.output.with_name(f"{args.output.stem}-rank-{rank:02d}-failure.json"),
            {"traceback": traceback.format_exc()},
            allow_nan=False,
        )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
