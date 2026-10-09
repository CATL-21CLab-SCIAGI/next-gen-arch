"""Admit matched Limite SFT checkpoints to graph sampling and native RL replay.

Uses disposable models with identical checkpoint values. Small teacher-forced
checks compare native eager scores and every trainable gradient with chunked
replay. Full-context admission retains the production actor, graph pool,
snapshots and Adam states while updating disposable learner weights. The
source checkpoint is immutable. Synthetic inputs test execution, not quality.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from archlab.automodel.checkpoint_oracle import assert_state_equal
from archlab.automodel.limite_adapter_common import build_model, file_hash, runtime_contract
from archlab.automodel.limite_native_rl_qualification import (
    decode_oracle,
    gradient_report,
    parameter_fingerprint,
)
from archlab.rl.limite_checkpoint import capture_rng, restore_rng
from archlab.rl.limite_scoring import enable_chunked_policy_scores


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _repeat_ids(ids, length):
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or length < 2:
        raise ValueError("qualification requires one nonempty token row and length >= 2")
    return ids.repeat(1, (length + ids.shape[1] - 1) // ids.shape[1])[:, :length]


def _record(output, report, phase):
    report["phase"] = phase
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(dict(phase=phase, variant=report["variant"], output=str(output))), flush=True)


def score_admitted(report):
    return (
        report["finite_scores"] and report["mean_abs_error"] < .02
        and report["max_abs_error"] < .1 and report["clip_fraction"] <= .05
        and .5 <= report["min_ratio"] <= report["max_ratio"] <= 2
    )


def score_error(actual, expected):
    delta = actual.detach().float() - expected.detach().float()
    ratio = delta.exp()
    result = dict(
        finite_scores=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        mean_abs_error=float(delta.abs().mean()), max_abs_error=float(delta.abs().max()),
        min_ratio=float(ratio.min()), max_ratio=float(ratio.max()),
        clip_fraction=float(((ratio < .8) | (ratio > 1.2)).float().mean()),
        tolerance=dict(mean_abs_error=.02, max_abs_error=.1, clip_fraction=.05,
                       ratio_range=[.5, 2.]),
    )
    result["passed"] = score_admitted(result)
    return result


def _eager_scores(model, ids, keep):
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=keep + 1).logits[:, :-1].float()
    return logits.log_softmax(-1).gather(-1, ids[:, -keep:, None]).squeeze(-1)


def _replay_scores(model, ids, keep):
    scores, _ = model(input_ids=ids, attention_mask=None, use_cache=False,
                      archlab_replay=(keep, 1., False))
    return scores


def _objective(scores):
    # Nonuniform, deterministic token weights exercise alignment and head
    # derivatives. They are a numerical fixture, not a changed RL objective.
    weights = torch.linspace(.5, 1.5, scores.shape[-1], device=scores.device)
    return -(scores * weights).mean()


def gradient_error(model, expected, *, repeat_gradients=(), update_report=None):
    missing, nonfinite, worst = [], [], []
    squared_error, squared_reference = 0., 0.
    groups = {"adapter": [0., 0.], "backbone": [0., 0.]}
    outliers = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target, actual = expected.get(name), parameter.grad
        if (target is None) != (actual is None):
            missing.append(name)
            continue
        if actual is None:
            continue
        actual = actual.detach().cpu().float()
        target = target.float()
        if not bool(torch.isfinite(actual).all() and torch.isfinite(target).all()):
            nonfinite.append(name)
            continue
        reference_norm = float(torch.linalg.vector_norm(target))
        difference_norm = float(torch.linalg.vector_norm(actual - target))
        reference_squared, difference_squared = reference_norm**2, difference_norm**2
        squared_reference += reference_squared
        squared_error += difference_squared
        group = "adapter" if name.startswith("model.adapters.") else "backbone"
        groups[group][0] += difference_squared
        groups[group][1] += reference_squared
        relative = difference_norm / max(reference_norm, 1e-8)
        baseline = [target, *(row[name].float() for row in repeat_gradients if row[name] is not None)]
        noise = max((float(torch.linalg.vector_norm(left - right))
                     for index, left in enumerate(baseline) for right in baseline[index + 1:]), default=0.)
        gradient_bound = max(.2 * reference_norm, 3 * noise)
        equivalent_update = update_report is not None and update_report["per_parameter"].get(name, False)
        if difference_norm > gradient_bound and not equivalent_update:
            outliers.append(name)
        worst.append(dict(name=name, relative_l2=relative, difference_norm=difference_norm,
                          reference_norm=reference_norm, repeat_reference_noise_norm=noise,
                          gradient_bound=gradient_bound, equivalent_first_update=equivalent_update))
    relative = math.sqrt(squared_error / max(squared_reference, 1e-16))
    maximum = max((row["relative_l2"] for row in worst), default=0.)
    result = dict(
        relative_l2=relative, max_tensor_relative_l2=maximum, missing=missing, nonfinite=nonfinite,
        groups={name: dict(relative_l2=math.sqrt(error / max(reference, 1e-16)),
                           reference_norm=math.sqrt(reference))
                for name, (error, reference) in groups.items()},
        worst_tensors=sorted(worst, key=lambda row: row["relative_l2"], reverse=True)[:10],
        outliers=outliers,
        tolerance=dict(relative_l2=.04, group_relative_l2=.04, tensor_signal_fraction=.2,
                       repeat_reference_noise_multiplier=3,
                       marginal_tensor="must satisfy gradient bound or first-update parity"),
    )
    result["passed"] = (bool(worst) and not missing and not nonfinite and not outliers and relative < .04
                        and all(row["relative_l2"] < .04 for row in result["groups"].values())
                        and (update_report is None or update_report["passed"]))
    return result


def update_error(actual, expected, repeats):
    """Bound actual FP32-master update differences against repeat-eager noise."""
    sums = {"all": [0., 0., 0.], "adapter": [0., 0., 0.], "backbone": [0., 0., 0.]}
    per_parameter, failures = {}, []
    for name, target in expected.items():
        value = actual[name]
        if value is None or target is None:
            per_parameter[name] = value is None and target is None
            if not per_parameter[name]:
                failures.append(name)
            continue
        reference_norm = float(torch.linalg.vector_norm(target))
        error = float(torch.linalg.vector_norm(value - target))
        baseline = [target, *(row[name] for row in repeats if row[name] is not None)]
        noise = max((float(torch.linalg.vector_norm(left - right))
                     for index, left in enumerate(baseline) for right in baseline[index + 1:]), default=0.)
        bound = max(.02 * reference_norm, 3 * noise)
        per_parameter[name] = math.isfinite(error) and error <= bound
        if not per_parameter[name]:
            failures.append(name)
        group = "adapter" if name.startswith("model.adapters.") else "backbone"
        for category in ("all", group):
            sums[category][0] += error**2
            sums[category][1] += reference_norm**2
            sums[category][2] += noise**2
    groups = {}
    for group, (error, reference, noise) in sums.items():
        bound = max(.02 * math.sqrt(reference), 3 * math.sqrt(noise))
        relative = math.sqrt(error / max(reference, 1e-16))
        groups[group] = dict(relative_l2=relative, difference_norm=math.sqrt(error),
                             repeat_reference_noise_norm=math.sqrt(noise), bound=bound,
                             passed=math.sqrt(error) <= bound and relative < .1)
    return dict(
        passed=not failures and all(row["passed"] for row in groups.values()),
        groups=groups, failed_tensors=failures, per_parameter=per_parameter,
        tolerance=dict(tensor_signal_fraction=.02, repeat_reference_noise_multiplier=3,
                       global_and_group_relative_l2=.1),
        oracle="fresh Adam FP32 master deltas with native clipping and subtraction rounding",
        scope="first fresh-optimizer update only; multi-step distributed/resume fixture required separately",
    )


def replay_oracle(reference, optimized, ids, length):
    """Compare the real scoring interfaces and all gradients on identical IDs."""
    ids = _repeat_ids(ids, length)
    keep = max(1, length // 2)
    rng = capture_rng()
    modes = (reference.training, optimized.training)
    previous = [{name: parameter.grad for name, parameter in model.named_parameters()}
                for model in (reference, optimized)]
    started = time.perf_counter()
    try:
        from archlab.rl.limite_update_oracle import first_adam_updates

        reference.train()
        optimized.train()
        reference.zero_grad(set_to_none=True)
        optimized.zero_grad(set_to_none=True)
        expected = _eager_scores(reference, ids, keep)
        _objective(expected).backward()
        expected = expected.detach()
        reference_gradients = {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                              for name, parameter in reference.named_parameters() if parameter.requires_grad}
        reference_evidence = gradient_report(reference)
        parameters = {name: parameter.detach().cpu().clone()
                      for name, parameter in reference.named_parameters() if parameter.requires_grad}
        reference_updates, optimizer_contract = first_adam_updates(parameters, reference_gradients)
        repeats, repeat_updates = [], []
        for _ in range(2):
            reference.zero_grad(set_to_none=True)
            restore_rng(rng)
            _objective(_eager_scores(reference, ids, keep)).backward()
            gradients = {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                         for name, parameter in reference.named_parameters() if parameter.requires_grad}
            repeats.append(gradients)
            repeat_updates.append(first_adam_updates(parameters, gradients)[0])
        reference.zero_grad(set_to_none=True)
        restore_rng(rng)
        actual = _replay_scores(optimized, ids, keep)
        loss = _objective(actual)
        loss.backward()
        _synchronize(ids.device)
        actual_gradients = {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                           for name, parameter in optimized.named_parameters() if parameter.requires_grad}
        updates = first_adam_updates(parameters, actual_gradients)[0]
        update_report = update_error(updates, reference_updates, repeat_updates)
        result = dict(
            length=length, prompt_tokens=length - keep, completion_tokens=keep,
            scores=score_error(actual, expected),
            gradients=gradient_error(optimized, reference_gradients, repeat_gradients=repeats,
                                     update_report=update_report),
            first_update={key: value for key, value in update_report.items() if key != "per_parameter"},
            optimizer_contract=optimizer_contract, repeat_reference_backwards=3,
            reference_gradients=reference_evidence, replay_gradients=gradient_report(optimized),
            loss=float(loss.detach()), seconds=time.perf_counter() - started,
            reference="same_checkpoint_native_eager_softcapped_logits",
            actual="checkpointed_chunked_policy_scores", no_optimizer_step=True,
        )
        result["passed"] = (result["scores"]["passed"] and result["gradients"]["passed"]
                            and result["reference_gradients"]["all_trainable_gradients_finite"]
                            and result["replay_gradients"]["all_trainable_gradients_finite"])
        return result
    finally:
        for model, gradients, mode in zip((reference, optimized), previous, modes, strict=True):
            for name, parameter in model.named_parameters():
                parameter.grad = gradients[name]
            model.train(mode)
        restore_rng(rng)


def full_context_oracle(model, ids, length, prompt_limit):
    """Stress the production RL score/backward path at its actual token limit."""
    if not 0 < prompt_limit < length <= model.config.max_position_embeddings:
        raise ValueError("stress prompt and completion must fit the native context")
    before = parameter_fingerprint(model)
    rng, was_training = capture_rng(), model.training
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    tokens = _repeat_ids(ids, length)
    started = time.perf_counter()
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        _synchronize(tokens.device)
        if tokens.is_cuda:
            torch.cuda.reset_peak_memory_stats(tokens.device)
        scores = _replay_scores(model, tokens, length - prompt_limit)
        loss = _objective(scores)
        loss.backward()
        _synchronize(tokens.device)
        report = dict(
            **gradient_report(model), sequence_length=length, prompt_tokens=prompt_limit,
            completion_tokens=length - prompt_limit, loss=float(loss.detach()),
            finite_scores=bool(torch.isfinite(scores).all()), seconds=time.perf_counter() - started,
            peak_memory_gib=torch.cuda.max_memory_allocated(tokens.device) / 1024**3 if tokens.is_cuda else None,
            peak_reserved_gib=torch.cuda.max_memory_reserved(tokens.device) / 1024**3 if tokens.is_cuda else None,
            no_optimizer_step=True, numerical_input="repeated versioned nonbenchmark canary",
            execution="production chunked policy scores and checkpointed native replay",
        )
    finally:
        for name, parameter in model.named_parameters():
            parameter.grad = gradients[name]
        model.train(was_training)
        restore_rng(rng)
    report["weights_unchanged"] = parameter_fingerprint(model) == before
    report["gradients_restored"] = all(parameter.grad is gradients[name]
                                       for name, parameter in model.named_parameters())
    report["training_mode_restored"] = model.training == was_training
    report["passed"] = (report["all_trainable_gradients_finite"] and report["gradient_norm"] > 0
                        and report["finite_scores"] and math.isfinite(report["loss"])
                        and report["weights_unchanged"] and report["gradients_restored"]
                        and report["training_mode_restored"])
    return report


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def resident_context_oracle(model, actor, ids, length, prompt_limit, tokenizer, *, num_generations=4):
    """Exercise the resident production actor, learner and initialized optimizer.

    This gate updates disposable weights, republishes them and checks every
    captured actor batch. Restoring the learner afterwards retains the original
    checkpoint fingerprint. It tests one rank's memory contract; distributed
    checkpoint/resume admission is a separate requirement.
    """
    from transformers import GenerationConfig

    from archlab.automodel.limite_adapter_rl import optimizer_for_model
    from archlab.automodel.limite_decode_qualification import distribution_error
    from archlab.rl.limite_actor import native_actor_queue, policy_snapshot

    if not ids.is_cuda or not 0 < prompt_limit < length <= model.config.max_position_embeddings:
        raise ValueError("resident admission requires CUDA and a valid native-context sequence")
    before, rng, mode = parameter_fingerprint(model), capture_rng(), model.training
    originals = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    queue = None
    started = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats(ids.device)
        config = GenerationConfig(max_new_tokens=actor.config.max_position_embeddings,
                                  do_sample=True, temperature=1., top_p=1., top_k=0)
        config.archlab_budget_mode = "native_context"
        trainer = SimpleNamespace(
            args=SimpleNamespace(num_generations=num_generations), generation_config=config,
            processing_class=tokenizer, archlab_stop_requested=lambda: False,
            archlab_compact_decode=True, archlab_max_policy_lag=1,
        )
        actor.requires_grad_(False).eval()
        queue = native_actor_queue(actor, model, trainer, 0)
        previous_snapshot = queue.snapshot
        pool = queue.archlab_graph_pool
        expected_keys = {(batch, actor.config.max_position_embeddings)
                         for batch in range(1, num_generations + 1)}
        pool_complete = set(pool.entries) == expected_keys and all(
            decoder.graph is not None for decoder in pool.entries.values()
        ) and not pool.capture_allowed
        optimizer = optimizer_for_model(model)
        # Public TE initialization allocates FP32 master weights and both
        # moments without advancing Adam's counter or fabricating an update.
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                optimizer.initialize_state(parameter, optimizer.store_param_remainders)
        optimizer_bytes = _tensor_bytes(optimizer.state)
        initialized = all({"exp_avg", "exp_avg_sq", "master_param"} <= optimizer.state[p].keys()
                          for group in optimizer.param_groups for p in group["params"])
        model.train()
        model.zero_grad(set_to_none=True)
        tokens = _repeat_ids(ids, length)
        scores = _replay_scores(model, tokens, length - prompt_limit)
        loss = _objective(scores)
        loss.backward()
        _synchronize(ids.device)
        evidence = gradient_report(model)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        optimizer.step()
        _synchronize(ids.device)
        changed = {"adapter": 0, "backbone": 0}
        for name, parameter in model.named_parameters():
            if not torch.equal(parameter.detach().cpu(), originals[name]):
                changed["adapter" if name.startswith("model.adapters.") else "backbone"] += 1
        updated_fingerprint = parameter_fingerprint(model)
        current_snapshot = policy_snapshot(model, 1)
        queue.publish(current_snapshot)
        queue.archlab_refresh_actor(current_snapshot)
        actor_matches = parameter_fingerprint(actor) == updated_fingerprint
        refresh = []
        with torch.no_grad():
            for batch in range(1, num_generations + 1):
                token = ids[:, :1].repeat(batch, 1)
                source = actor(input_ids=token, use_cache=True, logits_to_keep=1).past_key_values
                reference_cache = actor(input_ids=token, use_cache=True, logits_to_keep=1).past_key_values
                decoder = pool.get(source, token, actor.config.max_position_embeddings)
                actual = decoder(token, 1).clone()
                expected = actor(input_ids=token, past_key_values=reference_cache,
                                 use_cache=True, logits_to_keep=1).logits[:, -1]
                error = distribution_error(actual, expected)
                refresh.append(dict(batch_size=batch, **error, finite_logits=bool(torch.isfinite(actual).all())))
        _synchronize(ids.device)
        refresh_passed = all(row["finite_logits"] and row["weighted_error"] < .02 and row["kl"] < .001
                             for row in refresh)
        report = dict(
            **evidence, sequence_length=length, prompt_tokens=prompt_limit,
            completion_tokens=length - prompt_limit, loss=float(loss.detach()),
            finite_scores=bool(torch.isfinite(scores).all()), seconds=time.perf_counter() - started,
            peak_memory_gib=torch.cuda.max_memory_allocated(ids.device) / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(ids.device) / 1024**3,
            actor_graph_pool_complete=pool_complete, actor_graph_keys=sorted(pool.entries),
            actor_graph_capacity=actor.config.max_position_embeddings,
            actor_model_resident=True, actor_snapshot_bytes=_tensor_bytes(previous_snapshot.weights),
            simultaneous_policy_snapshots=2,
            snapshot_bytes_at_publish=_tensor_bytes(previous_snapshot.weights) + _tensor_bytes(current_snapshot.weights),
            optimizer="production SignalFusedAdam", optimizer_state_bytes=optimizer_bytes,
            optimizer_all_states_initialized=initialized, optimizer_group_steps=[g.get("step") for g in optimizer.param_groups],
            clipped_gradient_norm=gradient_norm, applied_disposable_updates=1, changed_tensors=changed,
            updated_actor_matches_learner=actor_matches, actor_refresh=refresh,
            distributed_scope="one rank; distributed optimizer/checkpoint/resume fixture required separately",
            numerical_input="repeated versioned nonbenchmark canary",
        )
        report["passed"] = (pool_complete and initialized and evidence["all_trainable_gradients_finite"]
                            and evidence["gradient_norm"] > 0 and report["finite_scores"]
                            and math.isfinite(report["loss"]) and all(changed.values())
                            and actor_matches and refresh_passed)
    finally:
        if queue is not None:
            queue.close()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(originals[name])
                parameter.grad = gradients[name]
        model.train(mode)
        restore_rng(rng)
    report["learner_weights_restored"] = parameter_fingerprint(model) == before
    report["passed"] &= report["learner_weights_restored"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--checkpoint", "--warmup", dest="checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-cache", type=Path)
    parser.add_argument("--variant", choices=("normal", "simplicial"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[257, 1027, 2053])
    parser.add_argument("--prompt-length", type=int, default=1027)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--graph-capacity", type=int, default=131072)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--replay-backend", choices=("sdpa_native", "sdpa_bounded", "fa4", "sdpa"),
                        default="sdpa_native")
    parser.add_argument("--native-gqa-backend", choices=("sdpa", "flash_attn_kvcache"),
                        default="flash_attn_kvcache")
    parser.add_argument("--full-context-stress", action="store_true")
    parser.add_argument("--stress-length", type=int, default=131072)
    parser.add_argument("--prompt-limit", type=int, default=2048)
    args = parser.parse_args()
    if min(args.lengths) < 2 or args.decode_steps < 8 or args.chunk_size < 1:
        parser.error("lengths >= 2, decode steps >= 8, and positive head chunks are required")
    if not 0 < args.prompt_length < args.graph_capacity - args.decode_steps:
        parser.error("graph capacity must exceed prompt length plus decode steps")
    import yaml
    from transformers import AutoTokenizer

    from archlab.architectures.limite_adapter import enable_runtime_sequence_attention
    from archlab.architectures.limite_gqa import set_native_decode_gqa
    from archlab.architectures.limite_replay import enable_native_replay

    receipt = json.loads((args.checkpoint / "COMPLETE.json").read_text())
    if receipt.get("tokens") != 10_000_000_000 or receipt.get("trainable_mode") != "full":
        raise ValueError("matched RL admission requires the completed 10B full-weight SFT checkpoint")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    rng = capture_rng()
    report = dict(
        passed=False, production_admitted=False, variant=args.variant,
        checkpoint=str(args.checkpoint), receipt_sha256=file_hash(args.checkpoint / "COMPLETE.json"),
        checkpoint_tokens=receipt["tokens"], checkpoint_files=receipt["files"],
        runtime=runtime_contract(), replay_backend=args.replay_backend,
        optimizer_constructed=False, benchmark_data_used=False,
        full_context_requested=args.full_context_stress,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model,
                                                 local_files_only=True, trust_remote_code=False)
        fixture = yaml.safe_load((Path(__file__).parents[1] / "prompts/limite_rl_canary_v1.yaml").read_text())
        prompt = tokenizer.apply_chat_template(fixture["messages"], tokenize=True,
                                              return_dict=False, add_generation_prompt=True)
        ids = torch.tensor([prompt], device="cuda")
        reference = build_model(args.model, args.variant, "cuda", args.checkpoint,
                                trainable_mode="full", checkpoint_cache=args.checkpoint_cache)
        optimized = build_model(args.model, args.variant, "cuda", args.checkpoint,
                                trainable_mode="full", checkpoint_cache=args.checkpoint_cache)
        before = parameter_fingerprint(reference)
        if parameter_fingerprint(optimized) != before:
            raise ValueError("reference and optimized checkpoint weights differ")
        report["parameter_sha256_before"] = before
        maximum = reference.config.max_position_embeddings
        if max([*args.lengths, args.graph_capacity, args.stress_length]) > maximum:
            raise ValueError("qualification length exceeds native model context")
        enable_runtime_sequence_attention(reference)
        enable_runtime_sequence_attention(optimized)
        graph_ids = _repeat_ids(ids, args.prompt_length).repeat(4, 1)
        report["decode"] = decode_oracle(
            reference, optimized, graph_ids, args.graph_capacity, steps=args.decode_steps,
            reference_name="same_SFT_checkpoint_native_eager_dynamic_cache",
            native_gqa_backend=args.native_gqa_backend, teacher_forcing="sampled",
        )
        report["decode"]["passed"] &= (
            report["decode"]["teacher_forced_mean_logprob_error"] < .02
            and report["decode"]["teacher_forced_max_logprob_error"] < .1
        )
        _record(args.output, report, "decode_complete")
        gc.collect()
        torch.cuda.empty_cache()
        enable_chunked_policy_scores(optimized, chunk_size=args.chunk_size)
        report["head_only"] = []
        for length in args.lengths:
            report["head_only"].append(replay_oracle(reference, optimized, ids, length))
            _record(args.output, report, f"head_only_{length}_complete")
        # The helper preserves existing decode bindings. Production configures
        # this same order before entering replay; no attention weights change.
        set_native_decode_gqa(optimized, backend=args.native_gqa_backend)
        enable_native_replay(optimized, attention_backend=args.replay_backend, checkpoint_layers=True)
        report["replay"] = []
        for length in args.lengths:
            report["replay"].append(replay_oracle(reference, optimized, ids, length))
            _record(args.output, report, f"replay_{length}_complete")
        report["reference_weights_unchanged"] = parameter_fingerprint(reference) == before
        numerical_passed = (report["decode"]["passed"] and report["reference_weights_unchanged"]
                            and all(row["passed"] for row in report["head_only"] + report["replay"]))
        del reference
        gc.collect()
        torch.cuda.empty_cache()
        if args.full_context_stress and numerical_passed:
            _record(args.output, report, "full_context_starting")
            actor = build_model(args.model, args.variant, "cuda", args.checkpoint,
                                trainable_mode="full", checkpoint_cache=args.checkpoint_cache)
            enable_runtime_sequence_attention(actor)
            set_native_decode_gqa(actor, backend=args.native_gqa_backend)
            report["optimizer_constructed"] = True
            report["full_context"] = resident_context_oracle(
                optimized, actor, ids, args.stress_length, args.prompt_limit, tokenizer,
            )
        elif args.full_context_stress:
            report["full_context"] = dict(passed=False, skipped="small-shape numerical admission failed")
        report["parameter_sha256_after"] = parameter_fingerprint(optimized)
        report["weights_unchanged"] = report["reference_weights_unchanged"] and report["parameter_sha256_after"] == before
        report["passed"] = (report["decode"]["passed"] and report["weights_unchanged"]
                            and all(row["passed"] for row in report["head_only"] + report["replay"])
                            and (not args.full_context_stress or report["full_context"]["passed"]))
        report["single_rank_admitted"] = (report["passed"] and args.full_context_stress
                                           and args.stress_length == maximum and args.graph_capacity == maximum)
        # A single process cannot establish distributed optimizer/checkpoint
        # correctness, even after its full resident memory test passes.
        report["production_admitted"] = False
        report["remaining_admission"] = ["distributed optimizer/checkpoint/resume fixture"]
    except BaseException as error:
        report["passed"] = report["production_admitted"] = False
        report["error"] = dict(type=type(error).__name__, message=str(error)[:2000])
        raise
    finally:
        restore_rng(rng)
        try:
            assert_state_equal(capture_rng(), rng)
            report["rng_preserved"] = True
        except AssertionError:
            report["rng_preserved"] = False
            report["passed"] = report["production_admitted"] = False
        _record(args.output, report, "complete" if report["passed"] else "failed")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise AssertionError("matched RL graph/replay qualification failed")


if __name__ == "__main__":
    main()
