"""One fresh-rollout RLOO update on the existing SUM-reduction V4.1 actor.

The only optimized objective is generated-completion log probability weighted by
leave-one-out outcome rewards. No CE, reference trace, indexer teacher loss, router
auxiliary loss, or loss-free router bias update is performed by this boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from archlab.optimizers.sharded_adafactor import local_tensor
from archlab.rl.objectives import group_relative_advantages, group_relative_policy_loss

_CHUNK_ELEMENTS = 16 * 1024 * 1024


def _world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _sum(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _max(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def _check_collectively(errors, *, configuration=None):
    packets = [(errors, configuration)]
    if dist.is_initialized():
        packets = [None] * dist.get_world_size()
        dist.all_gather_object(packets, (errors, configuration))
    if any(packet[0] for packet in packets):
        raise ValueError(f"RL update admission failed: {[packet[0] for packet in packets]}")
    if configuration is not None and any(packet[1] != configuration for packet in packets):
        raise ValueError("RL update configuration or parameter ownership differs across ranks")


@contextmanager
def _pure_policy_forward(model, indexers):
    """Use the rollout's eval-mode policy, retaining autograd and FSDP hooks."""
    modes = [(module, module.training) for module in model.modules()]
    attributes = []
    for indexer in indexers:
        for key, value in (("_archlab_auxiliary_backward_scale", 0.), ("_archlab_auxiliary", None)):
            if hasattr(indexer, key):
                attributes.append((indexer, key, getattr(indexer, key)))
                setattr(indexer, key, value)
    for module in model.modules():
        if hasattr(module, "aux_loss_coeff"):
            for key, value in (("aux_loss_coeff", 0.), ("bias_update_factor", 0.), ("_track_load_balance", False)):
                if hasattr(module, key):
                    attributes.append((module, key, getattr(module, key)))
                    setattr(module, key, value)
    try:
        model.eval()
        yield
    finally:
        for module, key, value in reversed(attributes):
            setattr(module, key, value)
        for module, training in modes:
            module.training = training


def _validate_rollout(model, rollout, rewards, group_size, lr, head_chunk_size,
                      max_grad_norm, replay_tolerance, audit_only,
                      replay_mode, replay_prefixes, replay_seed):
    errors = []
    rank, world = _world()
    try:
        if type(group_size) is not int or group_size < 2:
            raise ValueError("group_size must be >=2")
        if type(head_chunk_size) is not int or head_chunk_size < 1:
            raise ValueError("head_chunk_size must be positive")
        if not math.isfinite(lr) or not 0 < lr <= 1:
            raise ValueError("lr must lie in (0,1]")
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if not math.isfinite(replay_tolerance) or not 0 <= replay_tolerance <= .02:
            raise ValueError("replay_tolerance must lie in [0,.02]")
        if replay_mode not in ("packed", "sampled-prefix"):
            raise ValueError("replay_mode must be packed or sampled-prefix")
        if type(replay_prefixes) is not int or replay_prefixes < 1 or type(replay_seed) is not int:
            raise ValueError("replay_prefixes must be positive and replay_seed an integer")
        if getattr(rollout, "_archlab_rl_consumed", False):
            raise ValueError("rollout was already consumed; sample a fresh group")
        ids = rollout.input_ids
        if ids.ndim != 2 or ids.dtype != torch.long or min(ids.shape) < 1:
            raise ValueError("rollout input_ids must be a nonempty int64 matrix")
        if rewards.ndim != 2 or rewards.shape[1] != group_size or rewards.numel() != ids.shape[0]:
            raise ValueError("rewards must be [local prompts,group_size] in rollout row order")
        if rewards.device != ids.device or not bool(torch.isfinite(rewards).all()):
            raise ValueError("rewards must be finite on the rollout device")
        for name in ("attention_mask", "labels", "response_mask", "policy_log_probs", "behavior_log_probs"):
            value = getattr(rollout, name)
            if value.shape != ids.shape or value.device != ids.device:
                raise ValueError(f"{name} must match rollout input shape and device")
        if rollout.labels.dtype != torch.long or rollout.response_mask.dtype != torch.bool or rollout.attention_mask.dtype != torch.bool:
            raise ValueError("labels must be int64 and masks boolean")
        if not torch.equal(rollout.response_mask, rollout.labels != -100):
            raise ValueError("response mask must exactly identify generated targets")
        mask = rollout.response_mask
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every completion must include at least one generated token")
        for value in (rollout.policy_log_probs, rollout.behavior_log_probs):
            if not value.is_floating_point() or value.requires_grad or not bool(torch.isfinite(value[mask]).all()):
                raise ValueError("sampled log probabilities must be finite and detached")
            if bool((value[mask] > 1e-5).any()):
                raise ValueError("sampled log probabilities must be normalized")
        if not torch.equal(rollout.policy_log_probs[mask], rollout.behavior_log_probs[mask]):
            raise ValueError("behavior distribution differs from policy")
        receipt = rollout.receipt
        if (receipt.get("temperature") != 1 or receipt.get("top_p") != 1
                or receipt.get("on_policy_sampling") is not True):
            raise ValueError("RLOO requires fresh temperature=1, top_p=1 policy sampling")
        if receipt.get("rank") != rank or receipt.get("world_size") != world:
            raise ValueError("rollout rank/world receipt differs from current ownership")
        policy_version = receipt.get("policy_version")
        if not isinstance(policy_version, str) or not policy_version:
            raise ValueError("rollout needs an immutable policy_version")
        active_version = getattr(model, "_archlab_rl_policy_version", policy_version)
        if policy_version != active_version:
            raise ValueError("rollout belongs to a different current policy_version")
        if not hasattr(model.lm_head, "rl_log_probs"):
            raise ValueError("install the differentiable RL head before updating")
        vocabulary_size = model.lm_head.weight.shape[0]
        if bool(((ids < 0) | (ids >= vocabulary_size)).any()):
            raise ValueError("rollout input token outside the vocabulary")
        if len(rollout.prompt_lengths) != ids.shape[0] or len(rollout.generated_ids) != ids.shape[0]:
            raise ValueError("missing per-completion prompt/generation metadata")
        prompts = []
        for row, (length, generated) in enumerate(zip(rollout.prompt_lengths, rollout.generated_ids, strict=True)):
            if type(length) is not int or length < 1 or not generated or length + len(generated) > ids.shape[1]:
                raise ValueError("invalid prompt or generation length")
            if any(type(token) is not int or not 0 <= token < vocabulary_size for token in generated):
                raise ValueError("generated IDs must be valid vocabulary integers")
            expected = torch.zeros_like(mask[row])
            expected[length - 1:length - 1 + len(generated)] = True
            if not torch.equal(mask[row], expected):
                raise ValueError("completion mask is not next-token aligned")
            tokens = torch.tensor(generated, device=ids.device, dtype=torch.long)
            if (not torch.equal(rollout.labels[row][expected], tokens)
                    or not torch.equal(ids[row, length:length + len(generated)], tokens)):
                raise ValueError("generated targets do not match sampled continuation tokens")
            attention = torch.arange(ids.shape[1], device=ids.device) < length + len(generated)
            if not torch.equal(rollout.attention_mask[row], attention):
                raise ValueError("attention mask must preserve prompt and generated prefix")
            prompts.append(ids[row, :length].tolist())
        for first in range(0, len(prompts), group_size):
            if any(prompt != prompts[first] for prompt in prompts[first:first + group_size]):
                raise ValueError("each reward group must contain rollouts of the same prompt")
        if receipt.get("prompt_sha256") != hashlib.sha256(json.dumps(prompts).encode()).hexdigest():
            raise ValueError("prompt tokens differ from the sampling receipt")
        if receipt.get("generated_tokens") != int(mask.sum()):
            raise ValueError("generated token count differs from the sampling receipt")
        if any(not parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("the accepted full-actor experiment requires all weights trainable")
        for parameter in model.parameters():
            if local_tensor(parameter).device != ids.device:
                raise ValueError("model and rollouts must share the resident device")
            if isinstance(parameter, DTensor) and any(not isinstance(p, Shard) for p in parameter.placements):
                raise ValueError("DTensor optimizer parameters must have unique shard ownership")
        plan = [(name, list(parameter.shape), str(parameter.dtype), isinstance(parameter, DTensor))
                for name, parameter in model.named_parameters()]
        configuration = (group_size, lr, head_chunk_size, max_grad_norm, replay_tolerance,
                         bool(audit_only), policy_version, tuple(ids.shape), replay_mode,
                         replay_prefixes, replay_seed,
                         hashlib.sha256(json.dumps(plan).encode()).hexdigest())
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        errors.append(str(error))
        configuration = None
    _check_collectively(errors, configuration=configuration)


def select_prefix_times(total_steps, count, seed):
    """Uniform distinct generation times, using an isolated, reward-independent RNG."""
    if type(total_steps) is not int or total_steps < 1 or type(count) is not int or count < 1:
        raise ValueError("generation steps and prefix count must be positive integers")
    if type(seed) is not int:
        raise ValueError("prefix selection seed must be an integer")
    return tuple(sorted(random.Random(seed).sample(range(total_steps), min(count, total_steps))))


def _prefix_plan(rollout, count, seed, vocabulary_size):
    """Authenticate enough sampling metadata to reconstruct every exact prefix."""
    errors, configuration = [], None
    try:
        receipt = rollout.receipt
        steps = receipt["forward_count"]
        shapes = receipt["forward_shapes"]
        pad = receipt["pad_token_id"]
        stops = receipt["eos_token_ids"]
        batch, final_canvas = rollout.input_ids.shape
        if type(steps) is not int or steps < 1 or len(shapes) != steps:
            raise ValueError("missing complete global sampling forward history")
        if (type(pad) is not int or not 0 <= pad < vocabulary_size or not stops
                or any(type(token) is not int or not 0 <= token < vocabulary_size for token in stops)):
            raise ValueError("prefix replay needs recorded padding and stop tokens")
        if bool((rollout.input_ids[~rollout.attention_mask] != pad).any()):
            raise ValueError("final padding differs from the sampling pad token")
        for step, shape in enumerate(shapes):
            if (not isinstance(shape, (list, tuple)) or len(shape) != 2
                    or shape[0] != batch or type(shape[1]) is not int
                    or not 1 <= shape[1] <= final_canvas):
                raise ValueError("invalid recorded generation canvas")
            if any(prompt + min(step, len(generated)) > shape[1]
                   for prompt, generated in zip(rollout.prompt_lengths, rollout.generated_ids, strict=True)):
                raise ValueError("recorded canvas cannot contain its original prefix")
        if len(rollout.finish_reasons) != batch:
            raise ValueError("prefix replay needs every completion's stop reason")
        for generated, reason in zip(rollout.generated_ids, rollout.finish_reasons, strict=True):
            if len(generated) > steps or any(token in stops for token in generated[:-1]):
                raise ValueError("invalid sampled completion stop history")
            if (reason == "stop") != (generated[-1] in stops) or reason not in ("stop", "length"):
                raise ValueError("stop reason differs from generated tokens")
            if len(generated) < steps and reason != "stop":
                raise ValueError("a short sampled row must have stopped before other rows")
        times = select_prefix_times(steps, count, seed)
        configuration = (steps, tuple(tuple(shape) for shape in shapes), pad, tuple(stops), times)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        errors.append(str(error))
    _check_collectively(errors, configuration=configuration)
    longest = int(_max(torch.tensor(max(map(len, rollout.generated_ids)),
                                    device=rollout.input_ids.device, dtype=torch.long)))
    _check_collectively([] if longest == steps else ["global forward count differs from longest sampled response"])
    return steps, times


def reconstruct_prefix(rollout, step):
    """Restore the exact sampled IDs/mask/canvas and last-position action target.

    Finished rows retain their EOS in the prefix and a valid dummy target keeps
    the vocabulary GEMM's row count equal to sampling. Their loss weight is zero.
    """
    batch, canvas = rollout.receipt["forward_shapes"][step]
    device = rollout.input_ids.device
    pad = rollout.receipt["pad_token_id"]
    inputs = torch.full((batch, canvas), pad, device=device, dtype=torch.long)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    positions = torch.empty(batch, device=device, dtype=torch.long)
    labels = torch.full((batch, 1), pad, device=device, dtype=torch.long)
    active = torch.zeros((batch, 1), device=device, dtype=torch.bool)
    recorded = torch.zeros((batch, 1), device=device, dtype=torch.float32)
    for row, (prompt, generated) in enumerate(zip(rollout.prompt_lengths, rollout.generated_ids, strict=True)):
        length = prompt + min(step, len(generated))
        inputs[row, :length] = rollout.input_ids[row, :length]
        mask[row, :length] = True
        positions[row] = length - 1
        if step < len(generated):
            active[row, 0] = True
            labels[row, 0] = generated[step]
            recorded[row, 0] = rollout.behavior_log_probs[row, prompt - 1 + step]
    return inputs, mask, positions, labels, active, recorded


def policy_gradient_step(model, optimizer, indexers, rollout, rewards, *, lr, group_size,
                         head_chunk_size=128, max_grad_norm=1., replay_tolerance=.02,
                         audit=False, audit_only=False, replay_mode="packed",
                         replay_prefixes=4, replay_seed=0):
    """Return JSON metrics after one global-trajectory-mean, pure RLOO update.

    ``rewards[B,G]`` groups contiguous rows of this rank's rollout. FSDP owns SUM
    reductions; non-DTensor replicated adapter gradients receive exactly one
    explicit world SUM. Missing gradients get zero-filled only when some rank
    has that parameter's gradient, keeping optimizer collective order identical.
    Globally missing hard-selector/indexer gradients are expected and stay None.

    Recomputed sampled-token log probabilities must match the rollout within the
    declared absolute tolerance (default .02 nats). This is a fail-closed numerical
    bound for the actual sampling path versus differentiable replay, not permission
    to use a stale policy; the caller records actual replay errors. Sampling and
    recomputation both use eval mode, with autograd enabled only for the latter.

    ``audit_only`` verifies replay and backward without optimizer or LR changes;
    it is numerical qualification, never a claim of reward-driven learning. An
    all-flat global batch skips every forward/optimizer/auxiliary update. One
    RolloutBatch object may be consumed once, including by numerical qualification.

    ``sampled-prefix`` draws K distinct, uniformly selected global generation
    times out of T. It replays their original prefix IDs, padding masks, and
    canvases, then sums -(T/K)*A*logp/global_trajectories. Conditional on the
    rollout this is an unbiased estimator of the complete trajectory-score
    gradient before clipping. Backward runs once per selected time; parameters
    are updated once. The packed mode remains an explicitly checked diagnostic.
    """
    started = time.perf_counter()
    _validate_rollout(model, rollout, rewards, group_size, lr, head_chunk_size,
                      max_grad_norm, replay_tolerance, audit_only,
                      replay_mode, replay_prefixes, replay_seed)
    rank, world = _world()
    device = rollout.input_ids.device
    reward_errors = []
    try:
        advantages = group_relative_advantages(rewards, baseline="leave_one_out")
        reward64 = rewards.detach().double()
        if not bool(torch.isfinite(reward64.square().sum())):
            reward_errors.append("reward moments overflowed")
    except (ValueError, FloatingPointError) as error:
        reward_errors.append(str(error))
    _check_collectively(reward_errors)
    totals = torch.tensor([
        rewards.numel(), int(rollout.response_mask.sum()), rewards.shape[0],
        int((advantages != 0).any(dim=-1).sum()), float(reward64.sum()), float(reward64.square().sum()),
    ], device=device, dtype=torch.float64)
    _sum(totals)
    global_sequences, global_tokens = int(totals[0]), int(totals[1])
    mean = float(totals[4] / totals[0])
    metric = {
        "algorithm": "RLOO", "normalization": "global-trajectory-mean-of-token-sums",
        "policy_version": rollout.receipt["policy_version"], "group_size": group_size,
        "global_sequences": global_sequences, "completion_tokens": global_tokens,
        "global_prompt_groups": int(totals[2]), "nonflat_prompt_groups": int(totals[3]),
        "reward_mean": mean, "reward_std": math.sqrt(max(0., float(totals[5] / totals[0]) - mean * mean)),
        "learning_rate": lr, "reference_kl_available": False,
        "indexer_auxiliary_coefficient": 0., "router_auxiliary_coefficient": 0.,
        "router_bias_update": False, "audit_only": bool(audit_only),
        "updated": False, "optimizer_step_applied": False, "update_skipped": True,
        "policy_loss": 0., "gradient_norm_before_clip": 0., "gradient_norm": 0., "gradient_clip_scale": 1.,
        "replay_max_abs_error": None, "replay_mean_abs_error": None,
        "replay_tolerance": replay_tolerance, "replay_verified": False,
        "changed_local_elements": 0, "changed_elements_sum_across_ranks": 0,
        "replay_mode": replay_mode,
    }

    if replay_mode == "sampled-prefix":
        generation_steps, selected_times = _prefix_plan(rollout, replay_prefixes, replay_seed,
                                                       model.lm_head.weight.shape[0])
        metric.update(replay_total_steps=generation_steps, replay_prefix_count=len(selected_times),
                      replay_selected_times=list(selected_times), replay_seed=replay_seed,
                      replay_time_weight=generation_steps / len(selected_times),
                      replay_head_rows=rollout.input_ids.shape[0],
                      gradient_estimator="uniform-time-subsampled-trajectory-score")

    def finish(reason):
        rollout._archlab_rl_consumed = True
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = torch.tensor(time.perf_counter() - started, device=device, dtype=torch.float64)
        _max(elapsed)
        metric["update_seconds"] = float(elapsed)
        metric["skip_reason"] = reason
        return metric

    if not int(totals[3]):
        return finish("all_groups_flat")
    optimizer.zero_grad(set_to_none=True)
    # Install explicitly rather than assuming an earlier SFT helper chose SUM.
    from torch.distributed.fsdp import FSDPModule
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.)
    replay_max, replay_sum, replay_tokens, loss_value = 0., 0., 0, 0.
    time_indices = selected_times if replay_mode == "sampled-prefix" else (None,)
    try:
        with _pure_policy_forward(model, indexers):
            for step in time_indices:
                if step is None:
                    inputs, attention = rollout.input_ids, rollout.attention_mask
                    labels, active = rollout.labels, rollout.response_mask
                    recorded, positions = rollout.policy_log_probs, None
                else:
                    inputs, attention, positions, labels, active, recorded = reconstruct_prefix(rollout, step)
                hidden = model(input_ids=inputs, attention_mask=attention,
                               return_hidden_states=True).hidden_states
                if positions is not None:
                    hidden = hidden[torch.arange(inputs.shape[0], device=device), positions].unsqueeze(1)
                # Prefix targets are all valid (including inactive dummy rows),
                # so one GEMM uses exactly the sampler's B rows, never a subset.
                chunk = inputs.shape[0] if step is not None else head_chunk_size
                logp = model.lm_head.rl_log_probs(hidden, labels, chunk_size=chunk)
                errors = []
                if logp.shape != labels.shape or not bool(torch.isfinite(logp).all()):
                    errors.append("recomputed policy scores have invalid shape or nonfinite values")
                elif not logp.requires_grad or bool((logp[active] > 1e-5).any()):
                    errors.append("recomputed policy scores must be differentiable normalized log probabilities")
                _check_collectively(errors)
                delta = (logp.detach()[active] - recorded[active]).abs()
                local_max = delta.max().double() if delta.numel() else torch.zeros((), device=device, dtype=torch.float64)
                maximum = float(_max(local_max))
                error_sum = float(_sum(delta.double().sum()))
                scored = int(_sum(active.sum().long()))
                replay_max, replay_sum, replay_tokens = max(replay_max, maximum), replay_sum + error_sum, replay_tokens + scored
                if maximum > replay_tolerance:
                    raise ValueError(f"policy replay exceeds declared tolerance: {maximum} > {replay_tolerance}; prefix={step}")
                if step is None:
                    loss = group_relative_policy_loss(
                        logp.reshape(*rewards.shape, -1), active.reshape(*rewards.shape, -1),
                        advantages, normalization="sequence_sum",
                    ).loss * (rewards.numel() / global_sequences)
                else:
                    loss = -(logp.squeeze(1) * active.squeeze(1) * advantages.flatten()).sum()
                    loss = loss * (generation_steps / len(selected_times) / global_sequences)
                _check_collectively([] if bool(torch.isfinite(loss)) else ["nonfinite policy loss"])
                loss_value += float(_sum(loss.detach().double()))
                loss.backward()  # Locally inactive times still participate in every collective.
                del hidden, logp, loss
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    metric.update(replay_max_abs_error=replay_max, replay_mean_abs_error=replay_sum / max(1, replay_tokens),
                  replay_scored_tokens=replay_tokens, replay_verified=True, policy_loss=loss_value)

    # Match the established full-training boundary: inspect the restored sharded
    # parameter objects after FSDP backward, not transient unsharded references.
    parameters = list(model.named_parameters())
    selector_ids = {id(parameter) for indexer in indexers for parameter in indexer.parameters()}
    presence = torch.tensor([parameter.grad is not None for _, parameter in parameters], device=device, dtype=torch.int64)
    _sum(presence)
    gradient_errors = []
    for name, parameter in parameters:
        if parameter.grad is not None:
            gradient = local_tensor(parameter.grad)
            if gradient.is_sparse or gradient.shape != local_tensor(parameter).shape:
                gradient_errors.append(f"unsupported sparse or mismatched local gradient: {name}")
    _check_collectively(gradient_errors)
    errors, coverage = [], []
    squares = torch.zeros((), device=device, dtype=torch.float64)
    missing_selectors = []
    for (name, parameter), present in zip(parameters, presence.tolist(), strict=True):
        if not present:
            if id(parameter) not in selector_ids:
                errors.append(f"globally missing non-selector gradient: {name}")
            else:
                missing_selectors.append(name)
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradient = local_tensor(parameter.grad)
        replicated = not isinstance(parameter, DTensor)
        if replicated:
            _sum(gradient)
        magnitude = torch.zeros((), device=device, dtype=torch.float64)
        finite = torch.ones((), device=device, dtype=torch.bool)
        nonzero = torch.zeros((), device=device, dtype=torch.int64)
        for chunk in gradient.reshape(-1).split(_CHUNK_ELEMENTS):
            finite &= torch.isfinite(chunk).all()
            magnitude += torch.linalg.vector_norm(chunk, dtype=torch.float32).double().square()
            if audit:
                nonzero += chunk.count_nonzero()
        if not bool(finite):
            errors.append(f"nonfinite gradient: {name}")
        squares += magnitude / (world if replicated else 1)
        if audit:
            coverage.append({"name": name, "global_shape": list(parameter.shape),
                             "local_shape": list(gradient.shape), "replicated": replicated,
                             "gradient_l2_local": float(magnitude.sqrt()),
                             "nonzero_local_elements": int(nonzero),
                             "gradient_present_ranks": present})
    _check_collectively(errors)
    norm = float(_sum(squares).sqrt())
    if not math.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("nonfinite global RL gradient norm")
    metric["gradient_norm_before_clip"] = norm
    metric["gradient_norm"] = norm
    metric["globally_missing_hard_selector_gradients"] = len(missing_selectors)
    if audit:
        metric["audit"] = {"coverage": coverage, "missing_hard_selector_gradients": missing_selectors,
                           "local_advantages": advantages.tolist(), "signed_advantages": bool((advantages > 0).any() and (advantages < 0).any()),
                           "prompt_and_padding_loss_masked": True, "reference_trace_used": False}
    if norm == 0:
        optimizer.zero_grad(set_to_none=True)
        return finish("zero_gradient")
    clip = min(1., max_grad_norm / norm)
    metric["gradient_clip_scale"] = clip
    if clip < 1:
        for _, parameter in parameters:
            if parameter.grad is not None:
                local_tensor(parameter.grad).mul_(clip)
    if audit_only:
        metric["numerical_qualification_passed"] = True
        optimizer.zero_grad(set_to_none=True)
        return finish("audit_only")
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    metric["optimizer_step_applied"] = True
    metric["update_skipped"] = False
    optimizer_metrics = getattr(optimizer, "last_metrics", {})
    changed = optimizer_metrics.get("changed_local_elements")
    _check_collectively([] if type(changed) is int and changed >= 0 else ["optimizer omitted changed_local_elements evidence"])
    changed_global = int(_sum(torch.tensor(changed, device=device, dtype=torch.int64)))
    metric.update(changed_local_elements=changed, changed_elements_sum_across_ranks=changed_global,
                  updated=changed_global > 0,
                  updated_parameter_tensors=optimizer_metrics.get("updated_parameter_tensors"))
    # The next operation is long autoregressive sampling, not another backward;
    # retaining the full actor's gradients would needlessly occupy GPU memory.
    optimizer.zero_grad(set_to_none=True)
    return finish(None if changed_global else "no_weight_change")
