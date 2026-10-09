"""Bounded native Violetto graph/decode and full-context backward admission.

Runs on a disposable publisher model before production. Existing graph/cache
and native chunked-head oracles provide the numerical path; no optimizer step,
benchmark questions, or production learner state is involved.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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


def decode_oracle(reference, optimized, ids, capacity, *, steps=8,
                  reference_name="unmodified_publisher_eager_dynamic_cache",
                  native_gqa_backend="flash_attn_kvcache"):
    from archlab.architectures.limite_decode import GraphDecoderPool
    from archlab.architectures.limite_decode_state import cache_rows
    from archlab.architectures.limite_gqa import set_native_decode_gqa
    from archlab.automodel.limite_decode_qualification import distribution_error

    reference.eval()
    optimized.eval()
    set_native_decode_gqa(optimized, backend=native_gqa_backend)
    with torch.no_grad():
        reference_output = reference(input_ids=ids, use_cache=True, logits_to_keep=1)
        optimized_output = optimized(input_ids=ids, use_cache=True, logits_to_keep=1)
        native_cache, source = reference_output.past_key_values, optimized_output.past_key_values
        tokens = reference_output.logits[:, -1].argmax(-1, keepdim=True)
        del reference_output, optimized_output
        pool = GraphDecoderPool(optimized)
        pool.synchronize()
        decoder = pool.get(source, tokens, capacity)
        reused = pool.get(source, tokens, capacity)
        same_graph = reused is decoder
        rows = torch.arange(ids.shape[0], device=ids.device)
        metrics, compact_metrics, greedy, selected_logprob_errors = [], [], [], []
        max_logit_error, finite = 0.0, True
        prefix = ids.shape[1]
        for step in range(steps):
            if step in (2, 4, 6) and rows.numel() > 1:
                kept = torch.arange(1, len(rows), device=ids.device)
                view = cache_rows(decoder.cache, kept, length=prefix + step)
                rows = rows[1:]
                pool.synchronize()
                decoder = pool.get(view, tokens[rows], capacity)
            actual = decoder(tokens[rows], prefix + step).clone()
            expected = reference(
                input_ids=tokens, past_key_values=native_cache, use_cache=True, logits_to_keep=1,
            ).logits[:, -1]
            target = expected.index_select(0, rows)
            error = distribution_error(actual, target)
            metrics.append(error)
            if len(rows) != ids.shape[0]:
                compact_metrics.append(error)
            max_logit_error = max(max_logit_error, float((actual.float() - target.float()).abs().max()))
            finite = finite and bool(torch.isfinite(actual).all()) and bool(torch.isfinite(target).all())
            actual_greedy, target_greedy = actual.argmax(-1), target.argmax(-1)
            selected_logprob_errors.append((
                actual.float().log_softmax(-1).gather(-1, target_greedy[:, None])
                - target.float().log_softmax(-1).gather(-1, target_greedy[:, None])
            ).abs().flatten())
            greedy.append(dict(step=step, rows=rows.tolist(),
                               optimized=actual_greedy.tolist(), publisher=target_greedy.tolist(),
                               exact=torch.equal(actual_greedy, target_greedy)))
            # Teacher-force the same publisher-greedy tokens into both caches.
            # Any distribution error therefore cannot hide behind divergent histories.
            tokens = expected.argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        report = dict(
            steps=steps, batch_size=ids.shape[0], prompt_length=prefix, capacity=capacity,
            backend=native_gqa_backend, reference=reference_name,
            max_abs_logit_error=max_logit_error,
            mean_weighted_error=sum(x["weighted_error"] for x in metrics) / len(metrics),
            mean_kl=sum(x["kl"] for x in metrics) / len(metrics),
            compaction_max_weighted_error=max(x["weighted_error"] for x in compact_metrics),
            compaction_max_kl=max(x["kl"] for x in compact_metrics),
            greedy_ids_exact=all(row["exact"] for row in greedy), greedy_ids=greedy,
            pool_reuse_same_graph=same_graph, pool_hits=pool.hits, finite_logits=finite,
            teacher_forced_mean_logprob_error=float(torch.cat(selected_logprob_errors).mean()),
            teacher_forced_max_logprob_error=float(torch.cat(selected_logprob_errors).max()),
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
