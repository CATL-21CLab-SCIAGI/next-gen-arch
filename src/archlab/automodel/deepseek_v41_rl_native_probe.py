"""Small native V4.1 EP8/FSDP2 replay/residency qualification; no policy update."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('normal', 'simplicial'), default='normal')
    parser.add_argument('--cache-policy', action='store_true')
    parser.add_argument('--new-tokens', type=int)
    parser.add_argument('--prompt-tokens', type=int)
    parser.add_argument('--freeze-router', action='store_true')
    parser.add_argument('--checkpoint-input-offload', action='store_true')
    parser.add_argument('--loss-normalization', choices=('sequence_sum', 'prompt_token_mean'), default='sequence_sum')
    args = parser.parse_args()
    new_tokens = args.new_tokens if args.new_tokens is not None else (64 if args.cache_policy else 2)
    prompt_tokens = args.prompt_tokens if args.prompt_tokens is not None else (31 if args.cache_policy else 3)
    if min(new_tokens, prompt_tokens) < 1 or prompt_tokens + 2 + new_tokens > 1024:
        parser.error('probe token budgets must be positive and fit the bounded 1024-token context')
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages
    packages = select_container_kernel_packages(Path(os.environ['ARCHLAB_CONTAINER_KERNEL_PACKAGES']))
    import torch
    import torch.distributed as dist
    from archlab.automodel.deepseek_v41_rl_model import construct_rl_actor, EXECUTION_CHANGES
    from archlab.automodel.deepseek_v41_rl_head import install_rl_head
    from archlab.automodel.deepseek_v41_rl_update import policy_gradient_step
    from archlab.optimizers.sharded_adafactor import ShardedAdafactor
    from archlab.rl.rollout import sample_rollouts
    from archlab.automodel.deepseek_v41_rl_training import source_identity
    from archlab.artifacts import atomic_write_json, sha256_file

    sources = {**source_identity(), 'automodel/deepseek_v41_rl_native_probe.py': sha256_file(__file__)}

    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    # Qualification may share a node with a live actor. Bound allocator demand.
    torch.cuda.set_per_process_memory_fraction(.02)
    torch.set_num_threads(4)
    torch.manual_seed(82)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=15),
                            device_id=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    rank = dist.get_rank()
    try:
        if args.output.exists():
            raise FileExistsError('use a fresh qualification output')
        model, indexers, loading, _ = construct_rl_actor(checkpoint=None, family='full',
            variant=args.variant, tiny=True, assets=Path(os.environ['ARCHLAB_DEEPSEEK_V41_ASSETS']),
            weights=Path(os.environ['ARCHLAB_DEEPSEEK_V41_CHECKPOINT']),
            declared_execution_changes=EXECUTION_CHANGES, resolved_kernel_packages=packages)
        install_rl_head(model)
        from archlab.automodel.deepseek_v41_rl_model import configure_rl_trainability
        from archlab.automodel.deepseek_v41_rl_memory import install_checkpoint_input_offload, input_offload_statistics
        strategy = configure_rl_trainability(model, freeze_router=args.freeze_router)
        if args.checkpoint_input_offload:
            strategy['checkpoint_inputs'] = install_checkpoint_input_offload(model)
        cache_equivalence = None
        if args.cache_policy:
            from archlab.optimizers.sharded_adafactor import local_tensor
            from archlab.automodel.deepseek_v41_rl_cache import qualify_resident_cache
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if 'simplicial_adapter.output.weight' in name:
                        local_tensor(parameter).normal_(0, .001)
        prompts = [[100 + rank, *range(101, 100 + prompt_tokens + (rank % 3 if args.cache_policy else 0))]] * 4
        context_limit = ((prompt_tokens + 2 + max(new_tokens, 16) + 127) // 128) * 128
        if args.cache_policy:
            cache_equivalence = qualify_resident_cache(model, prompts, pad_token_id=2,
                context_limit=context_limit, steps=16, tolerance=.02)
            atomic_write_json(args.output.with_name(f'{args.output.stem}-rank-{rank:02d}-prefix.json'),
                              cache_equivalence, allow_nan=False)
            if not cache_equivalence['passed']:
                raise AssertionError('distributed resident cache disagrees with padded full prefixes')
        common = dict(policy_version='native-random-qualification', max_new_tokens=new_tokens,
            context_limit=context_limit, eos_token_ids={1}, pad_token_id=2, seed=891,
            prompt_group_ids=[f'qual-{rank}'] * 4)
        first = sample_rollouts(model, prompts, retain_weights=args.cache_policy, **common)
        second = sample_rollouts(model, prompts, retain_weights=True, cache_policy=args.cache_policy, **common)
        if not args.cache_policy:
            if first.generated_ids != second.generated_ids:
                raise AssertionError('native retained weights changed sampled actions')
            torch.testing.assert_close(first.policy_log_probs, second.policy_log_probs, atol=2e-6, rtol=1e-6)
        if not second.receipt['weight_residency']['cleanup_verified']:
            raise AssertionError('native ownership cleanup failed')
        model._archlab_rl_policy_version = common['policy_version']
        optimizer = ShardedAdafactor((p for p in model.parameters() if p.requires_grad), lr=1e-6)
        audit = policy_gradient_step(model, optimizer, indexers, second,
            torch.tensor([[1., 0., 0., 0.]], device='cuda'), lr=1e-6, group_size=4,
            replay_mode='sampled-prefix', replay_prefixes=min(4, new_tokens), replay_seed=9,
            audit_only=True, audit=True, loss_normalization=args.loss_normalization)
        if not audit.get('numerical_qualification_passed') or any(optimizer.state.values()):
            raise AssertionError('native gradient qualification failed or updated optimizer')
        offload_stats = input_offload_statistics(model)
        if args.checkpoint_input_offload and offload_stats['tensor_copies'] == 0:
            raise AssertionError('requested checkpoint activation offload was not exercised')
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, {'rank': rank, 'passed': True, 'audit': audit,
            'residency': second.receipt['weight_residency'], 'cache_equivalence': cache_equivalence,
            'uncached_rollout': first.receipt, 'candidate_rollout': second.receipt,
            'same_sampled_actions': first.generated_ids == second.generated_ids,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'input_offload': offload_stats})
        if rank == 0:
            if sources != {**source_identity(), 'automodel/deepseek_v41_rl_native_probe.py': sha256_file(__file__)}:
                raise RuntimeError('probe source changed during qualification')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'passed': True, 'world_size': dist.get_world_size(),
                'scope': 'random tiny native V4.1; EP8/FSDP2; no optimizer update',
                'variant': args.variant, 'cache_policy': args.cache_policy,
                'strategy': strategy, 'loss_normalization': args.loss_normalization,
                'implementation_sha256': sources, 'runtime': loading, 'ranks': records}, indent=2) + '\n')
            print(json.dumps({'passed': True, 'output': str(args.output)}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
