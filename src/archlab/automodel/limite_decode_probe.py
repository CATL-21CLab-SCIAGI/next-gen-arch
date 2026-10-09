"""Native-cache versus CUDA-graph numerical and throughput admission."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from archlab.architectures.limite_adapter import enable_runtime_sequence_attention
from archlab.architectures.limite_decode import GraphDecoder
from archlab.automodel.limite_adapter_common import build_model, runtime_contract


def main():
    import yaml
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt-length", type=int, default=1027)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--capacity", type=int, default=18432)
    p.add_argument("--generation-tokens", type=int, default=1024)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model = build_model(a.model, a.variant, "cuda", a.checkpoint)
    enable_runtime_sequence_attention(model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    fixture = yaml.safe_load((Path(__file__).parents[1] / "prompts" / "limite_rl_canary_v1.yaml").read_text())
    prompt = tokenizer.apply_chat_template(fixture["messages"], tokenize=True, return_dict=False, add_generation_prompt=True)
    ids = torch.tensor([prompt], device="cuda")
    ids = ids.repeat(4, a.prompt_length // ids.shape[1] + 1)[:, :a.prompt_length]
    tokens = torch.full((4, 1), tokenizer.encode("42", add_special_tokens=False)[0], device="cuda")
    with torch.no_grad():
        output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        eager_cache = output.past_key_values
        started = time.perf_counter()
        decoder = GraphDecoder(model, eager_cache, tokens, a.capacity)
        capture_seconds = time.perf_counter() - started
        del output
        errors, weighted_errors, divergence, selected_clip = [], [], [], []
        for i in range(a.steps):
            graph_logits = decoder(tokens, a.prompt_length + i).clone().float()
            eager_logits = model(input_ids=tokens, past_key_values=eager_cache, use_cache=True, logits_to_keep=1).logits[:, -1].float()
            delta = graph_logits.log_softmax(-1) - eager_logits.log_softmax(-1)
            errors.append(delta.abs())
            probs = graph_logits.softmax(-1)
            weighted_errors.append((delta.abs() * probs).sum(-1))
            divergence.append((delta * probs).sum(-1))
            samples = torch.multinomial(probs, 128, replacement=True)
            ratios = delta.gather(-1, samples).exp()
            selected_clip.append(((ratios < .8) | (ratios > 1.2)).float().mean())
        errors = torch.stack(errors)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for i in range(a.steps):
            decoder(tokens, a.prompt_length + a.steps + i)
        torch.cuda.synchronize()
        graph_seconds = (time.perf_counter() - started) / a.steps
        started = time.perf_counter()
        for _ in range(a.steps):
            model(input_ids=tokens, past_key_values=eager_cache, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        eager_seconds = (time.perf_counter() - started) / a.steps
    report = dict(
        variant=a.variant, prompt_length=a.prompt_length, steps=a.steps,
        capacity=a.capacity, capture_seconds=capture_seconds,
        logprob_mean_abs_error=float(errors.mean()), logprob_max_abs_error=float(errors.max()),
        logprob_p99_abs_error=float(errors.reshape(-1)[::max(1, errors.numel() // 1000000)].quantile(.99)),
        probability_weighted_logprob_error=float(torch.cat(weighted_errors).mean()),
        kl_graph_to_native=float(torch.cat(divergence).mean()),
        sampled_probability_clip_fraction=float(torch.stack(selected_clip).mean()),
        graph_ms=graph_seconds * 1000, eager_ms=eager_seconds * 1000,
        speedup=eager_seconds / graph_seconds, runtime=runtime_contract(),
    )
    # All-vocabulary error overweights practically impossible tokens. Admission
    # uses probability-weighted distribution agreement and actual sampled-token
    # ratios, followed by the existing native train-replay oracle below.
    from transformers import GenerationConfig

    from archlab.rl.limite_generation import graph_generate

    del decoder, eager_cache, errors
    model.eval()
    completions, behavior, lengths, reasons, _ = graph_generate(
        model, ids, GenerationConfig(do_sample=True, temperature=1., top_p=1., top_k=0,
                                     max_new_tokens=a.generation_tokens, eos_token_id=[151643, 151645]),
        tokenizer, lambda: False, capacity=a.capacity,
    )
    model.train()
    deltas = []
    with torch.no_grad():
        for i, length in enumerate(lengths):
            sequence = torch.cat((ids[i:i + 1], completions[i:i + 1, :length]), dim=1)
            logits = model(input_ids=sequence, use_cache=False, logits_to_keep=length + 1).logits[:, :-1].float()
            replay = logits.log_softmax(-1).gather(-1, completions[i:i + 1, :length, None]).squeeze(-1)
            deltas.append((replay[0] - behavior[i, :length]).detach())
    deltas = torch.cat(deltas)
    ratios = deltas.exp()
    report.update(
        replay_logprob_mean_abs_error=float(deltas.abs().mean()),
        replay_logprob_max_abs_error=float(deltas.abs().max()),
        replay_importance_min=float(ratios.min()), replay_importance_max=float(ratios.max()),
        replay_importance_clip_fraction=float(((ratios < .8) | (ratios > 1.2)).float().mean()),
        generated_lengths=lengths, finish_reasons=reasons,
    )
    report["passed"] = (
        report["probability_weighted_logprob_error"] < .02
        and report["kl_graph_to_native"] < .001
        and report["sampled_probability_clip_fraction"] < .01
        and report["replay_importance_clip_fraction"] <= .05
        and .5 <= report["replay_importance_min"] <= report["replay_importance_max"] <= 2
        and report["speedup"] > 1
    )
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "runtime"}), flush=True)
    if not report["passed"]:
        raise AssertionError("graph decode did not pass native probability and speed admission")


if __name__ == "__main__":
    main()
