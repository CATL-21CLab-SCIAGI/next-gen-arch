"""Observability hooks for the pinned Miles math recipe; no reward changes.

Reward contrast is a proxy for available GRPO signal, before importance-weight
clipping. Token counts refer to collected responses, not measured GPU work.
"""

import logging
import math
from collections import defaultdict
from numbers import Real

logger = logging.getLogger(__name__)
_sampling_metrics = {}


def normalize_reward_scalars(samples):
    """Miles' zero-variance logger requires float keys (0.0 rather than 0)."""
    for sample in samples:
        if isinstance(sample.reward, Real):
            sample.reward = float(sample.reward)


def reward_group_metrics(args, samples):
    groups = defaultdict(list)
    for sample in samples:
        if sample.group_index is None:
            raise ValueError("reward-yield metrics require explicit prompt group IDs")
        groups[sample.group_index].append(sample)
    counts = dict(groups=len(groups), constant_groups=0, contrast_groups=0,
                  invalid_groups=0, all_zero_groups=0, all_one_groups=0,
                  response_tokens=0, reward_contrast_tokens=0)
    rewards = []
    for group in groups.values():
        values = [sample.get_reward_value(args) for sample in group]
        tokens = sum(0 if sample.remove_sample else sample.effective_response_length
                     for sample in group)
        counts['response_tokens'] += tokens
        if not all(isinstance(value, Real) and math.isfinite(value) for value in values):
            counts['invalid_groups'] += 1
            continue
        rewards.extend(values)
        if all(value == values[0] for value in values):
            counts['constant_groups'] += 1
            counts['all_zero_groups'] += int(values[0] == 0)
            counts['all_one_groups'] += int(values[0] == 1)
        else:
            counts['contrast_groups'] += 1
            counts['reward_contrast_tokens'] += tokens
    counts['group_contrast_fraction'] = counts['contrast_groups'] / max(1, len(groups))
    counts['token_contrast_fraction'] = counts['reward_contrast_tokens'] / max(1, counts['response_tokens'])
    counts['raw_reward'] = sum(rewards) / len(rewards) if rewards else 0.0
    return counts


def generated_tokens_at_version(samples, version):
    """Count newly generated spans, excluding prefixes retained from old steps."""
    return sum(span.abs_end - span.abs_start for sample in samples
               for span in sample.all_weight_version_spans if str(span.version) == str(version))


def record_all_samples(args, groups, data_source):
    """Record completed candidates before filtering, using Miles' public hook."""
    del data_source
    samples = [sample for group in groups for sample in group]
    normalize_reward_scalars(samples)
    _sampling_metrics.clear()
    _sampling_metrics.update(reward_group_metrics(args, samples))
    _sampling_metrics['new_response_tokens'] = generated_tokens_at_version(samples, args.archlab_policy_version)


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    normalize_reward_scalars(samples)
    values = {f'rollout/gradient_yield/{key}': value
              for key, value in reward_group_metrics(args, samples).items()}
    values.update({f'sampling/unfiltered/{key}': value for key, value in _sampling_metrics.items()})
    _sampling_metrics.clear()
    replayed = bool((rollout_extra_metrics or {}).get('qualification/replayed_parent_batch'))
    if replayed:
        from miles.ray.rollout.metrics import _compute_metrics_from_samples

        values.update({f'rollout/{key}': value for key, value in _compute_metrics_from_samples(args, samples).items()})
        values.update(rollout_extra_metrics)
        values['qualification/rollout_replay_seconds'] = rollout_time
    else:
        new_tokens = values.get('sampling/unfiltered/new_response_tokens', 0)
        new_tokens += (rollout_extra_metrics or {}).get('sampling/partial_new_response_tokens', 0)
        values['sampling/new_response_tokens'] = new_tokens
        values['sampling/new_tokens_per_rollout_gpu_second'] = new_tokens / max(1e-9, rollout_time * args.rollout_num_gpus)
    logger.info('rollout %s: %s', rollout_id, values)
    from miles.utils.metric_utils import compute_rollout_step
    from miles.utils.tracking_utils.tracking import log

    values['rollout/step'] = compute_rollout_step(args, rollout_id)
    log(args, values, step_key='rollout/step')
    # File replay is not generation throughput. Live batches keep native metrics
    # as well as the new-token counters that exclude reused prefixes.
    return replayed


def log_eval(rollout_id, args, data, extra_metrics):
    del rollout_id, extra_metrics
    configs = {config.name: config for config in args.eval_datasets}
    for name, dataset in data.items():
        samples = dataset.get('samples') or []
        normalize_reward_scalars(samples)
        # The pinned single-turn evaluator numbers rows but omits group IDs.
        # Restore these only for that known contiguous, complete-group format.
        count = configs[name].n_samples_per_eval_prompt
        if (samples and all(sample.group_index is None for sample in samples)
                and len(samples) % count == 0
                and sorted(sample.index for sample in samples if sample.index is not None) == list(range(len(samples)))):
            for sample in samples:
                sample.group_index = sample.index // count
    return False
