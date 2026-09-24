"""MiMo V2.6 GRPO equations (1)--(4), using generation-time behavior scores."""

from collections import defaultdict

import torch

from archlab.rl.regularization import DEFAULT_LENGTH_PENALTY, successful_length_rewards
from archlab.rl.rewards import verify_math_answer


async def reward(args, sample, **kwargs):
    return verify_math_answer(sample.response, sample.label).reward


def groups(samples, expected):
    result = defaultdict(list)
    for i, sample in enumerate(samples):
        if sample.group_index is None:
            raise ValueError("MiMo normalization requires explicit prompt identities")
        result[sample.group_index].append(i)
    if any(len(indices) != expected for indices in result.values()):
        raise ValueError("MiMo admission requires complete prompt groups")
    return list(result.values())


def process_rewards(args, samples):
    raw = [sample.get_reward_value(args) for sample in samples]
    centered = [0.0] * len(samples)
    for indices in groups(samples, args.n_samples_per_prompt):
        values = torch.tensor([[raw[i] for i in indices]])
        lengths = torch.tensor([[samples[i].response_length for i in indices]])
        adjusted, _ = successful_length_rewards(values, lengths, {**DEFAULT_LENGTH_PENALTY, "enabled": True})
        advantage = adjusted[0] - adjusted.mean()
        for i, value in zip(indices, advantage.tolist(), strict=True):
            centered[i] = value
    return raw, centered


def convert_samples(args, samples):
    from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data

    result = convert_samples_to_train_data(args, samples, {}, None, process_rewards)
    for indices in groups(samples, args.n_samples_per_prompt):
        count = sum(sum(result["loss_masks"][i]) for i in indices)
        if count == 0:
            raise ValueError("empty prompt group")
        for i in indices:
            result["rollout_mask_sums"][i] = count
    return result


def masked_importance_objective(logp, behavior, advantage, mask, denominator, group_size):
    if not torch.isfinite(logp).all() or not torch.isfinite(behavior).all():
        raise FloatingPointError("nonfinite policy scores")
    ratio = (logp.detach() - behavior.detach()).exp()
    admitted = (ratio >= 0.2) & (ratio <= 5.0)
    weight = torch.where(admitted, ratio, 0) * advantage.detach() * mask
    return -(weight * logp).sum() * group_size / denominator, admitted.float().mean()


def loss(args, batch, logits, sum_of_sample_mean):
    from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy

    if args.context_parallel_size != 1 or args.n_samples_per_prompt != 16:
        raise ValueError("MiMo launch admission is CP1, 16 responses per prompt")
    scored = get_log_probs_and_entropy(
        logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"],
        with_entropy=True, entropy_requires_grad=False, max_seq_lens=batch.get("max_seq_lens"))
    objective, admitted, score_deltas = [], [], []
    for current, old, advantage, mask, denom in zip(
            scored["log_probs"], batch["rollout_log_probs"], batch["advantages"],
            batch["loss_masks"], batch["rollout_mask_sums"], strict=True):
        value, fraction = masked_importance_objective(current, old, advantage, mask,
                                                       denom, args.n_samples_per_prompt)
        objective.append(value)
        admitted.append(fraction)
        score_deltas.append((current.detach() - old.detach()).abs().mean())
        import os
        if os.environ.get("ARCHLAB_RL_OFFLOAD_POLICY") == "forbidden":
            from archlab.megatron.miles_v41_policy_parity import active
            if active is not None:
                active.record(current, old, mask)
    total = torch.stack(objective).sum()
    return total, {"loss": total.detach(), "mimo_ratio_admitted": torch.stack(admitted).sum(),
                   "behavior_logprob_abs_delta": torch.stack(score_deltas).sum(),
                   "entropy": sum_of_sample_mean(torch.cat(scored["entropy"])).detach()}
