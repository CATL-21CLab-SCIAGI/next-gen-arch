"""Bounded native Limite cache/replay and differentiability admission."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path


def main():
    import torch

    from archlab.architectures.limite_loader import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--variant", choices=("normal", "simplicial"))
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backward-length", type=int, default=0)
    parser.add_argument("--prompt-length", type=int, default=0)
    parser.add_argument("--sample-tokens", type=int, default=128)
    parser.add_argument("--canary", type=Path)
    parser.add_argument("--cache-implementation", default="dynamic", choices=["dynamic", "static"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--eval-generation", action="store_true")
    parser.add_argument("--runtime-sequence-attention", action="store_true")
    parser.add_argument("--compare-streamed-behavior", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(0.60 if args.adapter_checkpoint else 0.45)
    if args.data:
        row = json.loads((args.data / "prompts.jsonl").open().readline())
        ids = torch.tensor([row["input_ids"]], device="cuda")
    elif not args.canary:
        parser.error("--data or --canary is required")
    if args.canary:
        import yaml
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer or args.model, local_files_only=True, trust_remote_code=False
        )
        messages = yaml.safe_load(args.canary.read_text())["messages"]
        ids = torch.tensor(
            [
                tokenizer.apply_chat_template(
                    messages, tokenize=True, return_dict=False, add_generation_prompt=True
                )
            ],
            device="cuda",
        )
    if args.prompt_length:
        ids = ids.repeat(1, args.prompt_length // ids.shape[1] + 1)[:, : args.prompt_length]
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    ids = ids.repeat(args.batch_size, 1)
    if args.adapter_checkpoint:
        from archlab.automodel.limite_adapter_common import build_model, frozen_fingerprint

        model = build_model(args.model, args.variant, "cuda", args.adapter_checkpoint)
        frozen_before = frozen_fingerprint(model)
    else:
        model = load_model(args.model, attn_implementation="sdpa", device_map="cuda")
    if args.runtime_sequence_attention:
        from archlab.architectures.limite_adapter import enable_runtime_sequence_attention

        enable_runtime_sequence_attention(model)
    started = time.monotonic()
    # Production native_rollout samples in eval mode and replays in train mode.
    # Keep train-mode generation available for explicit numerical comparisons.
    model.train(not args.eval_generation)
    rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())
    streamed_report = None
    with torch.no_grad():
        generated = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=args.sample_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            cache_implementation=args.cache_implementation,
            eos_token_id=[151645, 151643],
            disable_compile=not args.compile,
        )
        tokens = generated.sequences
        behavior = torch.stack(
            [
                score.float()
                .log_softmax(-1)
                .gather(-1, tokens[:, ids.shape[1] + i, None])
                .squeeze(-1)
                for i, score in enumerate(generated.scores)
            ],
            dim=1,
        )
        if args.compare_streamed_behavior:
            from transformers import GenerationConfig

            from archlab.rl.limite_rollout import SamplingLogprobs

            stored_bytes = sum(score.numel() * score.element_size() for score in generated.scores)
            generated.scores = None
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state(rng["cuda"])
            recorder = SamplingLogprobs(GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0))
            streamed = model.generate(
                ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=args.sample_tokens, do_sample=True,
                temperature=1.0, top_p=1.0, top_k=0,
                return_dict_in_generate=True, output_scores=False,
                logits_processor=[recorder], use_cache=True,
                cache_implementation=args.cache_implementation,
                eos_token_id=[151645, 151643], disable_compile=not args.compile,
            )
            selected = recorder.finish(streamed.sequences)
            streamed_report = dict(
                sampled_tokens_exact=torch.equal(streamed.sequences, tokens),
                behavior_logprobs_exact=torch.equal(selected, behavior),
                retained_full_scores_bytes=stored_bytes,
                streamed_distribution_bytes=recorder.max_distribution_bytes,
            )
            if not all(streamed_report[key] for key in ("sampled_tokens_exact", "behavior_logprobs_exact")):
                raise AssertionError("streamed behavior probabilities changed native sampling")
            del selected, streamed, recorder
        model.train()
        logits = model(tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
        replay = (
            logits[:, ids.shape[1] - 1 : -1]
            .float()
            .log_softmax(-1)
            .gather(-1, tokens[:, ids.shape[1] :, None])
            .squeeze(-1)
        )
        delta = (behavior - replay).abs()
        del logits
    model.zero_grad(set_to_none=True)
    if args.backward_length:
        tokens = tokens.repeat(1, (args.backward_length // tokens.shape[1]) + 1)[
            :, : args.backward_length
        ]
    # A real all-parameter backward, without modifying the parent policy.
    with torch.autocast("cuda", enabled=False):
        loss = model(
            tokens, attention_mask=torch.ones_like(tokens), use_cache=False, labels=tokens
        ).loss
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in gradients)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if args.adapter_checkpoint:
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert frozen_fingerprint(model) == frozen_before
    report = dict(
        cached_replay_mean_abs_error=float(delta.mean()),
        cached_replay_max_abs_error=float(delta.max()),
        finite_gradients=finite,
        gradient_norm=norm,
        backward_loss=float(loss.detach()),
        parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_sha256=frozen_before if args.adapter_checkpoint else None,
        gradient_tensors=len(gradients),
        peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
        seconds=time.monotonic() - started,
        backward_length=tokens.shape[1],
        prompt_length=ids.shape[1],
        batch_size=args.batch_size,
        runtime_sequence_attention=args.runtime_sequence_attention,
        streamed_behavior=streamed_report,
        versions={
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "transformers",
                "trl",
                "accelerate",
                "datasets",
                "transformer_engine",
                "safetensors",
            )
        },
        cuda=torch.version.cuda,
        hardware="NVIDIA B300 (NVML label L20D)",
    )
    ratios = (replay - behavior).exp()
    report["importance_ratio_min"] = float(ratios.min())
    report["importance_ratio_max"] = float(ratios.max())
    report["importance_clip_fraction"] = float(((ratios < 0.8) | (ratios > 1.2)).float().mean())
    report["numerical_contract"] = (
        "Native mixed-dtype policy; actual sampling probabilities; upstream 0.8–1.2 clipped surrogate; <=5% probe clipping"
    )
    report["passed"] = (
        finite
        and norm > 0
        and report["importance_clip_fraction"] <= 0.05
        and bool(((ratios >= 0.5) & (ratios <= 2.0)).all())
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, local_files_only=True, trust_remote_code=False
    )
    report["completion_preview"] = tokenizer.decode(
        generated.sequences[0, ids.shape[1] :], skip_special_tokens=True
    )
    report["eod_positions"] = (
        (generated.sequences[0, ids.shape[1] :] == 151643).nonzero().flatten().tolist()
    )
    report["im_end_positions"] = (
        (generated.sequences[0, ids.shape[1] :] == 151645).nonzero().flatten().tolist()
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("native model admission failed")


if __name__ == "__main__":
    main()
