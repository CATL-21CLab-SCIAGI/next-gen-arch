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
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages
    packages = select_container_kernel_packages(Path(os.environ['ARCHLAB_CONTAINER_KERNEL_PACKAGES']))
    import torch
    import torch.distributed as dist
    from archlab.automodel.deepseek_v41_rl_model import construct_rl_actor, EXECUTION_CHANGES
    from archlab.automodel.deepseek_v41_rl_head import install_rl_head
    from archlab.automodel.deepseek_v41_rl_update import policy_gradient_step
    from archlab.optimizers.sharded_adafactor import ShardedAdafactor
    from archlab.rl.rollout import sample_rollouts

    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(4)
    torch.manual_seed(82)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=15),
                            device_id=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    rank = dist.get_rank()
    try:
        model, indexers, loading, _ = construct_rl_actor(checkpoint=None, family='full',
            variant='normal', tiny=True, assets=Path(os.environ['ARCHLAB_DEEPSEEK_V41_ASSETS']),
            weights=Path(os.environ['ARCHLAB_DEEPSEEK_V41_CHECKPOINT']),
            declared_execution_changes=EXECUTION_CHANGES, resolved_kernel_packages=packages)
        install_rl_head(model)
        prompts = [[3, 5, 7 + rank]] * 4
        common = dict(policy_version='native-random-qualification', max_new_tokens=2,
            context_limit=128, eos_token_ids={1}, pad_token_id=2, seed=891,
            prompt_group_ids=[f'qual-{rank}'] * 4)
        first = sample_rollouts(model, prompts, retain_weights=False, **common)
        second = sample_rollouts(model, prompts, retain_weights=True, **common)
        if first.generated_ids != second.generated_ids:
            raise AssertionError('native retained weights changed sampled actions')
        torch.testing.assert_close(first.policy_log_probs, second.policy_log_probs, atol=2e-6, rtol=1e-6)
        if not second.receipt['weight_residency']['cleanup_verified']:
            raise AssertionError('native ownership cleanup failed')
        model._archlab_rl_policy_version = common['policy_version']
        optimizer = ShardedAdafactor(model.parameters(), lr=1e-6)
        audit = policy_gradient_step(model, optimizer, indexers, second,
            torch.tensor([[1., 0., 0., 0.]], device='cuda'), lr=1e-6, group_size=4,
            replay_mode='sampled-prefix', replay_prefixes=2, replay_seed=9,
            audit_only=True, audit=True)
        if not audit.get('numerical_qualification_passed') or any(optimizer.state.values()):
            raise AssertionError('native gradient qualification failed or updated optimizer')
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, {'rank': rank, 'passed': True, 'audit': audit,
            'residency': second.receipt['weight_residency']})
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'passed': True, 'world_size': dist.get_world_size(),
                'scope': 'random tiny native V4.1; EP8/FSDP2; no optimizer update',
                'runtime': loading, 'ranks': records}, indent=2) + '\n')
            print(json.dumps({'passed': True, 'output': str(args.output)}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
