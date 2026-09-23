"""Synchronous rollouts from the resident distributed training policy.

Each rank owns independent prompt rows. All ranks execute the same number of
model/head forwards and the same sequence-length bucket, including ranks whose
rows already ended. This is a correctness-first online sampling primitive, not
a separate serving engine. Callers supply tokenized native prompts and explicit
native EOS/turn-end IDs; no conversation template is guessed here.
"""

import hashlib
import json
import math
import time
from collections.abc import Collection, Sequence
from contextlib import ExitStack
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn import functional as F


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor
    policy_log_probs: torch.Tensor
    behavior_log_probs: torch.Tensor
    generated_ids: list[list[int]]
    prompt_lengths: list[int]
    finish_reasons: list[str]
    receipt: dict


def _maximum(value, device):
    packet = torch.tensor(value, device=device, dtype=torch.long)
    if dist.is_initialized():
        dist.all_reduce(packet, op=dist.ReduceOp.MAX)
    return int(packet)


def _bucket(length, multiple, limit):
    return min(limit, ((length + multiple - 1) // multiple) * multiple)


def _inputs(sequences, length, pad_token_id, device):
    ids = torch.full((len(sequences), length), pad_token_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        ids[row, :len(sequence)] = torch.tensor(sequence, device=device, dtype=torch.long)
        mask[row, :len(sequence)] = True
    return ids, mask


def sample_rollouts(
    model,
    prompts: Sequence[Sequence[int]],
    *,
    policy_version: str,
    max_new_tokens: int,
    context_limit: int,
    eos_token_ids: Collection[int],
    pad_token_id: int,
    seed: int,
    temperature: float = 1.,
    top_p: float = 1.,
    bucket_multiple: int = 128,
    prompt_group_ids: Sequence[str] | None = None,
    device=None,
    retain_weights: bool | None = None,
    cache_policy: bool = False,
):
    """Sample current-policy continuations, returning next-token-aligned targets.

    ``labels[b,t]`` is a generated token predicted by ``input_ids[b,:t+1]``;
    prompt and padding positions have label -100 and zero recorded log probability.
    EOS belongs to the response and is trained. A length-truncated response is
    marked ``length`` and never gains an invented EOS. Use temperature=top_p=1
    for sampling directly from the policy; other settings are explicitly recorded
    as a different behavior distribution. Temperature zero (with top_p=1) is
    deterministic greedy evaluation, with selected behavior log probability zero.
    Each rank's seed is ``seed + rank``.

    Model construction must allow boolean right-padding masks. Batch size,
    sampling settings, context limit and policy_version must match across ranks;
    prompt lengths and contents may differ. Call outside an optimizer update.
    ``retain_weights=None`` inherits ``model._archlab_rl_retain_weights`` (default
    False). True keeps existing FSDP-gathered weights for this call only; all
    ownership and policies are restored before returning. This does not add a KV
    cache or change sampling math. An explicit False overrides the model setting.
    ``cache_policy=True`` selects the inference-only resident V4.1 cache and
    requires equal-length prompt rows and retained FSDP weights. Its use in
    production requires separate B300 replay qualification.
    """
    from archlab.automodel.deepseek_v41_live_window import inference_head

    if device is None:
        device = model.lm_head.weight.device
    device = torch.device(device)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    error = None
    try:
        if not isinstance(policy_version, str) or not policy_version:
            raise ValueError("a nonempty immutable policy version is required")
        retain = getattr(model, "_archlab_rl_retain_weights", False) if retain_weights is None else retain_weights
        reserve_gib = getattr(model, "_archlab_rl_weight_reserve_gib", 16.)
        if type(retain) is not bool:
            raise ValueError("retain_weights and its model default must be boolean")
        if type(cache_policy) is not bool or cache_policy and not retain:
            raise ValueError("cached rollouts require a boolean flag and retained weights")
        if retain and (not isinstance(reserve_gib, (int, float)) or not math.isfinite(reserve_gib) or reserve_gib < 16):
            raise ValueError("retained-weight reserve must be at least 16 GiB")
        for name, value in (("max_new_tokens", max_new_tokens), ("context_limit", context_limit),
                            ("bucket_multiple", bucket_multiple)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(seed) is not int or type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError("seed and nonnegative pad_token_id must be integers")
        if not math.isfinite(temperature) or temperature < 0 or not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("sampling requires finite temperature >= 0 and 0 < top_p <= 1")
        if temperature == 0 and top_p != 1:
            raise ValueError("greedy evaluation requires top_p=1")
        sequences = [list(prompt) for prompt in prompts]
        if not sequences or any(not row or any(type(t) is not int or t < 0 for t in row) for row in sequences):
            raise ValueError("each rank needs nonempty integer-token prompt rows")
        if any(len(row) + max_new_tokens > context_limit for row in sequences):
            raise ValueError("prompt plus generation budget exceeds context_limit")
        if cache_policy and len({len(row) for row in sequences}) != 1:
            raise ValueError("cached rollouts require equal-length local prompt rows")
        stops = set(eos_token_ids)
        if not stops or any(type(t) is not int or t < 0 for t in stops):
            raise ValueError("explicit nonnegative native stop token IDs are required")
        vocabulary_size = model.lm_head.weight.shape[0]
        if (pad_token_id >= vocabulary_size or max(stops) >= vocabulary_size
                or any(max(row) >= vocabulary_size for row in sequences)):
            raise ValueError("prompt, pad, or stop token outside the vocabulary")
        if prompt_group_ids is not None and len(prompt_group_ids) != len(sequences):
            raise ValueError("prompt_group_ids must identify every local prompt row")
        configuration = (len(sequences), policy_version, max_new_tokens, context_limit,
                         tuple(sorted(stops)), pad_token_id, seed, temperature, top_p, bucket_multiple,
                         retain, reserve_gib if retain else None, cache_policy)
    except (TypeError, ValueError) as caught:
        error = str(caught)
        configuration = None
    packets = [(error, configuration)]
    if dist.is_initialized():
        packets = [None] * world
        dist.all_gather_object(packets, (error, configuration))
    if any(packet[0] for packet in packets):
        raise ValueError(f"invalid distributed rollout inputs: {[p[0] for p in packets]}")
    if any(packet[1] != configuration for packet in packets):
        raise ValueError("rollout settings and local batch sizes must match across ranks")

    original_prompts = [list(row) for row in sequences]
    lengths = [len(row) for row in sequences]
    generated = [[] for _ in sequences]
    policy_scores = [[] for _ in sequences]
    behavior_scores = [[] for _ in sequences]
    active = [True] * len(sequences)
    reasons = ["length"] * len(sequences)
    modes = [(module, module.training) for module in model.modules()]
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    forward_shapes = []
    replay_shapes = []
    residency = None
    decode_cache = None
    started = time.perf_counter()
    try:
        model.eval()
        with ExitStack() as scope:
            scope.enter_context(torch.no_grad())
            if retain:
                from archlab.rl.weight_residency import retained_fsdp_weights
                residency = scope.enter_context(retained_fsdp_weights(model, minimum_free_gib=reserve_gib))
            head_context = residency.inference_head if residency is not None else inference_head
            if cache_policy:
                from archlab.automodel.deepseek_v41_rl_cache import V41PolicyCache
                decode_cache = V41PolicyCache(model)
            for generation_step in range(max_new_tokens):
                if not _maximum(int(any(active)), device):
                    break
                length = _bucket(_maximum(max(map(len, sequences)), device), bucket_multiple, context_limit)
                if cache_policy:
                    replay_shapes.append([len(sequences), length])
                    if generation_step == 0:
                        prefill_ids, _ = _inputs(sequences, len(sequences[0]), pad_token_id, device)
                        final_hidden = decode_cache.prefill(prefill_ids)
                        forward_shapes.append([len(sequences), prefill_ids.shape[1]])
                    else:
                        next_ids = torch.tensor(selected, device=device, dtype=torch.long).unsqueeze(1)
                        final_hidden = decode_cache.decode(next_ids)
                        forward_shapes.append([len(sequences), 1])
                else:
                    ids, mask = _inputs(sequences, length, pad_token_id, device)
                    hidden = model(input_ids=ids, attention_mask=mask, return_hidden_states=True).hidden_states
                    positions = torch.tensor([len(row) - 1 for row in sequences], device=device)
                    final_hidden = hidden[torch.arange(len(sequences), device=device), positions]
                with head_context(model.lm_head) as head, torch.autocast(device.type, enabled=False):
                    logits = F.linear(final_hidden.float(), head.weight)
                if _maximum(int(not bool(logits.isfinite().all())), device):
                    raise FloatingPointError("nonfinite rollout logits on at least one rank")
                raw_logp = logits.log_softmax(-1)
                if temperature == 0:
                    choices = logits.argmax(-1, keepdim=True)
                    selected_behavior = [0.] * len(sequences)
                else:
                    behavior_logits = logits / temperature
                    if _maximum(int(not bool(behavior_logits.isfinite().all())), device):
                        raise FloatingPointError("nonfinite temperature-scaled logits on at least one rank")
                    if top_p < 1:
                        ordered, indices = behavior_logits.sort(descending=True)
                        probabilities = ordered.softmax(-1)
                        remove = probabilities.cumsum(-1) - probabilities >= top_p
                        behavior_logits = behavior_logits.scatter(1, indices, ordered.masked_fill(remove, -torch.inf))
                    behavior_logp = behavior_logits.log_softmax(-1)
                    choices = torch.multinomial(behavior_logp.exp(), 1, generator=generator)
                    selected_behavior = behavior_logp.gather(1, choices).squeeze(1).tolist()
                selected = choices.squeeze(1).tolist()
                selected_policy = raw_logp.gather(1, choices).squeeze(1).tolist()
                if not cache_policy:
                    forward_shapes.append([len(sequences), length])
                for row, token in enumerate(selected):
                    if active[row]:
                        sequences[row].append(token)
                        generated[row].append(token)
                        policy_scores[row].append(selected_policy[row])
                        behavior_scores[row].append(selected_behavior[row])
                        if token in stops:
                            active[row] = False
                            reasons[row] = "stop"
                if not cache_policy:
                    del hidden, final_hidden
                del logits, raw_logp
                if temperature:
                    del behavior_logits, behavior_logp
            # Release cache activations before FSDP reshares weights and before
            # the subsequent gradient replay allocates full-prefix activations.
            if cache_policy:
                del final_hidden, decode_cache
    finally:
        # Restore individually: wrappers may intentionally have mixed train/eval modes.
        for module, training in modes:
            module.training = training

    length = _bucket(_maximum(max(map(len, sequences)), device), bucket_multiple, context_limit)
    ids, mask = _inputs(sequences, length, pad_token_id, device)
    labels = torch.full_like(ids, -100)
    policy = torch.zeros_like(ids, dtype=torch.float32)
    behavior = torch.zeros_like(policy)
    for row, tokens in enumerate(generated):
        start, end = lengths[row] - 1, lengths[row] - 1 + len(tokens)
        labels[row, start:end] = torch.tensor(tokens, device=device)
        policy[row, start:end] = torch.tensor(policy_scores[row], device=device)
        behavior[row, start:end] = torch.tensor(behavior_scores[row], device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    receipt = {
        "policy_version": policy_version, "rank": rank, "world_size": world,
        "seed": seed, "effective_rank_seed": seed + rank, "temperature": temperature,
        "top_p": top_p, "on_policy_sampling": temperature == 1 and top_p == 1,
        "max_new_tokens": max_new_tokens, "context_limit": context_limit,
        "eos_token_ids": sorted(stops), "pad_token_id": pad_token_id,
        "prompt_sha256": hashlib.sha256(json.dumps(original_prompts).encode()).hexdigest(),
        "prompt_group_ids": None if prompt_group_ids is None else list(prompt_group_ids),
        "forward_shapes": forward_shapes, "forward_count": len(forward_shapes),
        "generated_tokens": sum(map(len, generated)), "seconds": time.perf_counter() - started,
        "cached": cache_policy,
        "backend": ("resident-model-v41-cache" if cache_policy else
                    "resident-model-full-prefix-retained-weights" if retain else "resident-model-full-prefix"),
        "retained_weights": retain,
        "weight_residency": None if residency is None else residency.receipt,
    }
    if cache_policy:
        receipt["replay_shapes"] = replay_shapes
        receipt["replay_inputs"] = "equivalent-full-prefix-right-padded"
    return RolloutBatch(ids, mask, labels, labels != -100, policy, behavior,
                        generated, lengths, reasons, receipt)
