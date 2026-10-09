"""Qualify full-context graph inference against native DynamicCache decoding.

Long probes repeat already-computed K/V states to test cache indexing/masking
at exact positions without quadratic long-prefill work. These are numerical
tests, never benchmark scores or evidence of long-context model quality.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch

from archlab.architectures.limite_adapter import enable_runtime_sequence_attention
from archlab.architectures.limite_decode import GraphDecoder
from archlab.architectures.limite_gqa import set_native_decode_gqa
from archlab.artifacts import atomic_write_json
from archlab.automodel.limite_adapter_common import build_model, runtime_contract
from archlab.automodel.limite_decode_qualification import distribution_error


def extend_cache(source, length, global_layers):
    """Build a native reference cache with identical repeated global K/V."""
    cache = copy.deepcopy(source)
    for index, layer in enumerate(cache.layers):
        if index in global_layers:
            for name in ("keys", "values"):
                value = getattr(layer, name)
                setattr(layer, name, value.repeat(1, 1, (length + value.shape[2] - 1) // value.shape[2], 1)
                        [:, :, :length].contiguous())
        elif hasattr(layer, "cumulative_length"):
            layer.cumulative_length = length
    if hasattr(cache, "archlab_native_preludes"):
        cache.archlab_native_preludes = extend_cache(source.archlab_native_preludes, length, global_layers)
    return cache


def main():
    import yaml
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-cache", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("normal", "simplicial"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model = build_model(args.model, args.variant, "cuda", args.checkpoint, checkpoint_cache=args.checkpoint_cache)
    enable_runtime_sequence_attention(model)
    model.requires_grad_(False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    fixture = yaml.safe_load((Path(__file__).parents[1] / "prompts/limite_rl_canary_v1.yaml").read_text())
    prompt = tokenizer.apply_chat_template(fixture["messages"], tokenize=True, return_dict=False,
                                           add_generation_prompt=True)
    ids = torch.tensor([prompt], device="cuda").repeat(1, 1027 // len(prompt) + 1)[:, :1027]
    token = torch.tensor([[tokenizer.encode("42", add_special_tokens=False)[0]]], device="cuda")
    report = dict(variant=args.variant, checkpoint=str(args.checkpoint), runtime=runtime_contract(), probes=[],
                  synthetic_cache=True, benchmark_score=False, context_limit=131072)
    with torch.no_grad():
        source = model(input_ids=ids, use_cache=True, logits_to_keep=1).past_key_values
        set_native_decode_gqa(model, backend="sdpa" if args.native_only else "flash_attn_kvcache")
        for length in (1027, 32760, 65528, 131064):
            native = extend_cache(source, length, set(model.config.global_layers))
            if args.native_only:
                timings, finite = [], True
                for _ in range(8):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    output = model(input_ids=token, past_key_values=native, use_cache=True, logits_to_keep=1)
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - started)
                    finite = finite and bool(torch.isfinite(output.logits).all())
                    del output
                row = dict(position=length, steps=8, native_eager=True, finite_logits=finite,
                           cache_length=int(native.get_seq_length()), decode_ms=1000 * sum(timings) / len(timings),
                           peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
                row["passed"] = finite and row["cache_length"] == length + 8
                report["probes"].append(row)
                atomic_write_json(args.output, report)
                print(json.dumps(row), flush=True)
                del native
                gc.collect()
                torch.cuda.empty_cache()
                continue
            torch.cuda.synchronize()
            started = time.perf_counter()
            graph = GraphDecoder(model, native, token, 131072)
            torch.cuda.synchronize()
            capture = time.perf_counter() - started
            errors, timings = [], []
            for step in range(8):
                torch.cuda.synchronize()
                started = time.perf_counter()
                actual = graph(token, length + step).clone()
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
                expected = model(input_ids=token, past_key_values=native, use_cache=True,
                                 logits_to_keep=1).logits[:, -1]
                errors.append(distribution_error(actual, expected))
            row = dict(position=length, steps=8, max_weighted_error=max(e["weighted_error"] for e in errors),
                       max_kl=max(abs(e["kl"]) for e in errors),
                       decode_ms=1000 * sum(timings) / len(timings), capture_seconds=capture,
                       peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
            row["passed"] = row["max_weighted_error"] < .02 and row["max_kl"] < .001
            report["probes"].append(row)
            atomic_write_json(args.output, report)
            print(json.dumps(row), flush=True)
            del graph, native, actual, expected
            gc.collect()
            torch.cuda.empty_cache()
    report["passed"] = all(p["passed"] for p in report["probes"])
    atomic_write_json(args.output, report)
    if not report["passed"]:
        raise RuntimeError("full-context graph qualification failed")


if __name__ == "__main__":
    main()
