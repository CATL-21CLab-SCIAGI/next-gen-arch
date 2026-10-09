"""Bounded native Violetto graph/decode and full-context backward admission.

Runs on a disposable publisher model before production. Existing graph/cache
and native chunked-head oracles provide the numerical path; no optimizer step,
benchmark questions, or production learner state is involved.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from archlab.automodel.checkpoint_oracle import assert_state_equal
from archlab.automodel.limite_adapter_common import loss_sum, runtime_contract
from archlab.automodel.limite_native_checkpoint import (
    build_native_model,
    check_native_publisher,
    publisher_identity,
)
from archlab.rl.limite_checkpoint import capture_rng, restore_rng


def parameter_fingerprint(model):
    """Compare every learned value/dtype, rather than a sampled parameter probe."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gradient_report(model):
    missing, nonfinite, zero, norms = [], [], [], []
    trainable = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable += 1
        grad = parameter.grad
        if grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(grad).all()):
            nonfinite.append(name)
        elif not bool(grad.count_nonzero()):
            zero.append(name)
        else:
            norms.append(torch.linalg.vector_norm(grad.float()).double())
    norm = float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.0
    return dict(
        trainable_tensors=trainable, gradient_tensors=trainable - len(missing),
        missing=missing, nonfinite=nonfinite, zero=zero, gradient_norm=norm,
        all_trainable_gradients_finite=not missing and not nonfinite,
        all_trainable_gradients_nonzero=not missing and not nonfinite and not zero,
    )


def decode_admitted(report):
    return (
        report["mean_weighted_error"] < .02
        and report["mean_kl"] < .001
        and report["compaction_max_weighted_error"] < .02
        and report["compaction_max_kl"] < .001
        and report["greedy_ids_exact"]
        and report["pool_reuse_same_graph"]
        and report["finite_logits"]
    )


def importance_distribution_control(actor_logits, native_logits, *, clip_epsilon=.2):
    """Measure conditional PPO clipping mass over the complete vocabulary.

    Actual graph sampling is the behavior distribution in the denominator.
    Log-space second moments avoid 0*inf in negligible-probability tails.
    """
    if actor_logits.shape != native_logits.shape or not 0 < clip_epsilon < 1:
        raise ValueError("importance controls require matching logits and a valid clipping epsilon")
    actor = actor_logits.float().log_softmax(-1)
    native = native_logits.float().log_softmax(-1)
    log_ratio = native - actor
    outside = (log_ratio < math.log1p(-clip_epsilon)) | (log_ratio > math.log1p(clip_epsilon))
    actor_clip = (actor.exp() * outside).sum(-1)
    native_clip = (native.exp() * outside).sum(-1)
    normalization = (actor + log_ratio).logsumexp(-1).exp()
    ess = (-(2 * native - actor).logsumexp(-1)).exp()
    finite = bool(torch.isfinite(actor).all() and torch.isfinite(native).all()
                  and torch.isfinite(ess).all() and torch.isfinite(normalization).all())
    result = dict(
        finite_full_support=finite, clip_epsilon=clip_epsilon,
        max_actor_clip_mass=float(actor_clip.max()), mean_actor_clip_mass=float(actor_clip.mean()),
        max_native_clip_mass=float(native_clip.max()), mean_native_clip_mass=float(native_clip.mean()),
        min_effective_sample_fraction=float(ess.min()), mean_effective_sample_fraction=float(ess.mean()),
        max_normalization_error=float((normalization - 1).abs().max()),
        ratio_direction="native numerator / actual graph behavior denominator",
        scope="exact full-vocabulary conditional distribution at identical forced histories",
        tolerance=dict(clipped_probability_mass=.05, normalization_error=1e-5),
    )
    result["passed"] = (finite and result["max_actor_clip_mass"] <= .05
                        and result["max_native_clip_mass"] <= .05
                        and result["max_normalization_error"] <= 1e-5)
    return result


def summarize_importance_controls(comparisons, active_row_counts):
    """Bound expected clipped-token mass over the tested forced histories.

    Each active row contributes one token opportunity. Weighting each step's
    exact conditional clipping mass by its row count therefore estimates the
    same token fraction bounded by the sampled GRPO health check. Conditional
    maxima remain diagnostics; support, normalization and ESS stay worst-case.
    This is a numerical-health check, separate from graph/cache correctness and
    the required actual full-completion behavior/replay test.
    """
    if not comparisons or len(comparisons) != len(active_row_counts):
        raise ValueError("importance controls require one active-row count per comparison")
    if any(type(count) is not int or count <= 0 for count in active_row_counts):
        raise ValueError("importance control active-row counts must be positive integers")
    keys = ("mean_actor_clip_mass", "max_actor_clip_mass", "mean_native_clip_mass",
            "max_native_clip_mass", "min_effective_sample_fraction", "max_normalization_error")
    finite = all(row["finite_full_support"] and all(math.isfinite(row[key]) for key in keys)
                 for row in comparisons)
    valid_mass = all(0 <= row[f"mean_{kind}_clip_mass"] <= row[f"max_{kind}_clip_mass"] <= 1
                     for row in comparisons for kind in ("actor", "native"))
    total = sum(active_row_counts)
    result = dict(
        finite_full_support=finite,
        mean_actor_clip_mass=sum(row["mean_actor_clip_mass"] * count
                                 for row, count in zip(comparisons, active_row_counts, strict=True)) / total,
        mean_native_clip_mass=sum(row["mean_native_clip_mass"] * count
                                  for row, count in zip(comparisons, active_row_counts, strict=True)) / total,
        max_actor_clip_mass=max(row["max_actor_clip_mass"] for row in comparisons),
        max_native_clip_mass=max(row["max_native_clip_mass"] for row in comparisons),
        min_effective_sample_fraction=min(row["min_effective_sample_fraction"] for row in comparisons),
        max_normalization_error=max(row["max_normalization_error"] for row in comparisons),
        active_row_counts=list(active_row_counts), active_row_token_comparisons=total,
        aggregation="sum(active_rows * conditional_mean_clip_mass) / sum(active_rows)",
        scope="exact expected clipped-token fraction on tested forced histories; full-completion IS required separately",
        per_conditional_max_bound_passed=all(row["passed"] for row in comparisons),
        comparisons=comparisons,
        tolerance=dict(clipped_probability_mass=.05, minimum_effective_sample_fraction=.95,
                       normalization_error=1e-5),
    )
    result["passed"] = (
        finite and valid_mass
        and result["mean_actor_clip_mass"] <= .05 and result["mean_native_clip_mass"] <= .05
        and .95 <= result["min_effective_sample_fraction"] <= 1 + 1e-5
        and all(0 <= row["max_normalization_error"] <= 1e-5 for row in comparisons)
    )
    return result


def decode_oracle(reference, optimized, ids, capacity, *, steps=8,
                  reference_name="unmodified_publisher_eager_dynamic_cache",
                  native_gqa_backend="flash_attn_kvcache", teacher_forcing="greedy", seed=1234,
                  clip_epsilon=.2):
    from archlab.architectures.limite_decode import GraphDecoderPool
    from archlab.architectures.limite_decode_state import cache_rows
    from archlab.architectures.limite_gqa import set_native_decode_gqa
    from archlab.automodel.limite_decode_qualification import distribution_error, select_native_rows

    if teacher_forcing not in ("greedy", "sampled"):
        raise ValueError("decode qualification requires greedy or sampled teacher forcing")
    if not 0 < clip_epsilon < 1:
        raise ValueError("decode clipping epsilon must be in (0, 1)")
    generator = torch.Generator(device=ids.device).manual_seed(seed)
    reference.eval()
    optimized.eval()
    set_native_decode_gqa(optimized, backend=native_gqa_backend)
    with torch.no_grad():
        reference_output = reference(input_ids=ids, use_cache=True, logits_to_keep=1)
        optimized_output = optimized(input_ids=ids, use_cache=True, logits_to_keep=1)
        native_cache, source = reference_output.past_key_values, optimized_output.past_key_values
        compact_native_cache = copy.deepcopy(native_cache)
        tokens = reference_output.logits[:, -1].argmax(-1, keepdim=True)
        del reference_output, optimized_output
        pool = GraphDecoderPool(optimized)
        pool.synchronize()
        decoder = pool.get(source, tokens, capacity)
        reused = pool.get(source, tokens, capacity)
        same_graph = reused is decoder
        rows = torch.arange(ids.shape[0], device=ids.device)
        metrics, compact_metrics, greedy, selected_logprob_deltas, batch_shape_controls = [], [], [], [], []
        selected_tokens, importance_controls, importance_row_counts = [], [], []
        max_logit_error, finite = 0.0, True
        prefix = ids.shape[1]
        for step in range(steps):
            if step in (2, 4, 6) and rows.numel() > 1:
                kept = torch.arange(1, len(rows), device=ids.device)
                view = cache_rows(decoder.cache, kept, length=prefix + step)
                select_native_rows(compact_native_cache, kept)
                rows = rows[1:]
                pool.synchronize()
                decoder = pool.get(view, tokens[rows], capacity)
            actual = decoder(tokens[rows], prefix + step).clone()
            expected = reference(
                input_ids=tokens, past_key_values=native_cache, use_cache=True, logits_to_keep=1,
            ).logits[:, -1]
            target = reference(
                input_ids=tokens[rows], past_key_values=compact_native_cache, use_cache=True, logits_to_keep=1,
            ).logits[:, -1]
            batch_shape_controls.append(distribution_error(target, expected.index_select(0, rows)))
            importance_controls.append(importance_distribution_control(actual, target, clip_epsilon=clip_epsilon))
            importance_row_counts.append(len(rows))
            error = distribution_error(actual, target)
            metrics.append(error)
            if len(rows) != ids.shape[0]:
                compact_metrics.append(error)
            max_logit_error = max(max_logit_error, float((actual.float() - target.float()).abs().max()))
            finite = finite and bool(torch.isfinite(actual).all()) and bool(torch.isfinite(target).all())
            actual_greedy, target_greedy = actual.argmax(-1), target.argmax(-1)
            forced_all = expected.argmax(-1, keepdim=True)
            if teacher_forcing == "sampled":
                # Draw real active tokens from the actual actor distribution;
                # all reference caches replay this same history. Retired rows
                # continue only for the full-batch rounding diagnostic.
                forced_all.index_copy_(0, rows, torch.multinomial(actual.float().softmax(-1), 1,
                                                                 generator=generator))
            forced = forced_all.index_select(0, rows)
            actual_selected = actual.float().log_softmax(-1).gather(-1, forced).flatten()
            target_selected = target.float().log_softmax(-1).gather(-1, forced).flatten()
            selected_logprob_deltas.append(actual_selected - target_selected)
            selected_tokens.append(dict(
                step=step, active_rows=rows.tolist(), token_ids=forced.flatten().tolist(),
                graph_logprobs=actual_selected.tolist(), native_same_batch_logprobs=target_selected.tolist(),
            ))
            greedy.append(dict(step=step, rows=rows.tolist(),
                               optimized=actual_greedy.tolist(), publisher=target_greedy.tolist(),
                               exact=torch.equal(actual_greedy, target_greedy)))
            # Teacher-force identical publisher-selected tokens into all caches.
            # Any distribution error therefore cannot hide behind divergent histories.
            tokens = forced_all
        torch.cuda.synchronize()
        selected_delta = torch.cat(selected_logprob_deltas)
        # The learner/native distribution is the numerator, while the actual
        # graph distribution supplies the recorded behavior denominator.
        selected_ratio = (-selected_delta).exp()
        report = dict(
            steps=steps, batch_size=ids.shape[0], prompt_length=prefix, capacity=capacity,
            backend=native_gqa_backend, reference=reference_name,
            teacher_forcing=teacher_forcing, teacher_forcing_seed=seed if teacher_forcing == "sampled" else None,
            teacher_forcing_source="actual graph behavior policy" if teacher_forcing == "sampled" else "native greedy",
            max_abs_logit_error=max_logit_error,
            mean_weighted_error=sum(x["weighted_error"] for x in metrics) / len(metrics),
            mean_kl=sum(x["kl"] for x in metrics) / len(metrics),
            compaction_max_weighted_error=max(x["weighted_error"] for x in compact_metrics),
            compaction_max_kl=max(x["kl"] for x in compact_metrics),
            greedy_ids_exact=all(row["exact"] for row in greedy), greedy_ids=greedy,
            pool_reuse_same_graph=same_graph, pool_hits=pool.hits, finite_logits=finite,
            teacher_forced_mean_logprob_error=float(selected_delta.abs().mean()),
            teacher_forced_max_logprob_error=float(selected_delta.abs().max()),
            teacher_forced_min_ratio=float(selected_ratio.min()),
            teacher_forced_max_ratio=float(selected_ratio.max()),
            teacher_forced_clip_fraction=float(((selected_ratio < 1 - clip_epsilon)
                                                | (selected_ratio > 1 + clip_epsilon)).float().mean()),
            teacher_forced_clip_epsilon=clip_epsilon,
            teacher_forced_tokens=selected_delta.numel(),
            teacher_forced_token_scores=selected_tokens,
            importance_distribution_controls=summarize_importance_controls(
                importance_controls, importance_row_counts,
            ),
            native_batch_shape_control=dict(
                mean_weighted_error=sum(x["weighted_error"] for x in batch_shape_controls) / len(batch_shape_controls),
                max_weighted_error=max(x["weighted_error"] for x in batch_shape_controls),
                max_kl=max(x["kl"] for x in batch_shape_controls),
                comparisons=batch_shape_controls,
                interpretation="native DynamicCache compact batch versus native full batch; diagnostic only",
            ),
            optimization_reference="native DynamicCache with identical active batch and teacher-forced history",
            tolerance=dict(weighted_error=.02, kl=.001, greedy_ids="exact"),
            oracle="archlab.automodel.limite_decode_qualification.distribution_error",
        )
        report["passed"] = decode_admitted(report)
    return report


def backward_oracle(model, ids, length, prompt_limit):
    before = parameter_fingerprint(model)
    was_training = model.training
    model.train()
    previous_gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    tokens = ids[:1].repeat(1, (length + 1) // ids.shape[1] + 1)[:, :length + 1]
    targets = tokens[:, 1:].clone()
    targets[:, :prompt_limit] = -100
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with torch.autocast("cuda", enabled=False):
            loss = loss_sum(model, tokens[:, :-1], targets, chunk=512, checkpoint_head=True)
            loss = loss / (targets != -100).sum()
        loss.backward()
        torch.cuda.synchronize()
        gradients = gradient_report(model)
        report = dict(
            **gradients, backward_length=length, prompt_limit=prompt_limit,
            response_tokens=length - prompt_limit, loss=float(loss.detach()),
            native_head="loss_sum: publisher softcap + token CE, chunk512/checkpoint",
            peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
            seconds=time.perf_counter() - started, no_optimizer_step=True,
            layer_count=len(model.model.layers),
            numerical_input="repeated versioned nonbenchmark math canary; differentiability only",
        )
    finally:
        for name, parameter in model.named_parameters():
            parameter.grad = previous_gradients[name]
        model.train(was_training)
    report["weights_unchanged"] = parameter_fingerprint(model) == before
    report["gradients_restored"] = all(parameter.grad is previous_gradients[name]
                                       for name, parameter in model.named_parameters())
    report["training_mode_restored"] = model.training == was_training
    report["passed"] = (report["all_trainable_gradients_finite"]
                        and report["all_trainable_gradients_nonzero"]
                        and report["gradient_norm"] > 0 and report["weights_unchanged"]
                        and report["gradients_restored"] and report["training_mode_restored"]
                        and bool(torch.isfinite(loss)))
    return report


def main():
    import yaml
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    recipe = yaml.safe_load(args.recipe.read_text())
    identity = publisher_identity(args.model)
    check_native_publisher(args.model, args.tokenizer, recipe, identity)
    initial_rng = capture_rng()
    report = dict(passed=False, publisher_identity=identity, model_kind="native",
                  source_recipe=str(args.recipe), runtime=runtime_contract())
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
        fixture = yaml.safe_load((Path(__file__).parents[1] / "prompts/limite_rl_canary_v1.yaml").read_text())
        prompt = tokenizer.apply_chat_template(fixture["messages"], tokenize=True,
                                              return_dict=False, add_generation_prompt=True)
        ids = torch.tensor([prompt], device="cuda").repeat(4, 1027 // len(prompt) + 1)[:, :1027]
        prompt_limit = recipe["data"]["prompt_limit"]
        capacity = prompt_limit + recipe["rollout"]["max_tokens"]
        reference = build_native_model(args.model, "cuda")
        optimized = build_native_model(args.model, "cuda")
        report["decode"] = decode_oracle(reference, optimized, ids, capacity)
        del optimized
        gc.collect()
        torch.cuda.empty_cache()
        report["backward"] = backward_oracle(reference, ids, capacity, prompt_limit)
        report["passed"] = report["decode"]["passed"] and report["backward"]["passed"]
    except BaseException as error:
        report["error"] = dict(type=type(error).__name__, message=str(error)[:2000])
        raise
    finally:
        restore_rng(initial_rng)
        try:
            assert_state_equal(capture_rng(), initial_rng)
            report["rng_preserved"] = True
        except AssertionError:
            report["rng_preserved"] = False
            report["passed"] = False
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise AssertionError("native publisher RL failed graph/backward admission")


if __name__ == "__main__":
    main()
