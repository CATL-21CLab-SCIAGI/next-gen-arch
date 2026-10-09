"""Qualify reusable native GQA decode, row compaction, and policy refresh."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from archlab.architectures.limite_adapter import enable_runtime_sequence_attention
from archlab.architectures.limite_decode import GraphDecoder, GraphDecoderPool
from archlab.architectures.limite_decode_state import cache_rows
from archlab.architectures.limite_gqa import set_native_decode_gqa
from archlab.automodel.limite_adapter_common import build_model, runtime_contract


def distribution_error(actual, expected):
    delta = actual.float().log_softmax(-1) - expected.float().log_softmax(-1)
    probabilities = actual.float().softmax(-1)
    return dict(weighted_error=float((delta.abs() * probabilities).sum(-1).mean()),
                kl=float((delta * probabilities).sum(-1).mean()))


def select_native_rows(cache, rows):
    """Apply the publisher cache operation to every architecture-owned history."""
    cache.batch_select_indices(rows)
    for index, values in getattr(cache, 'archlab_short', {}).items():
        cache.archlab_short[index] = [value.index_select(0, rows) for value in values]
    if hasattr(cache, 'archlab_native_preludes'):
        select_native_rows(cache.archlab_native_preludes, rows)


def main():
    import yaml
    from transformers import AutoTokenizer, GenerationConfig

    from archlab.rl.limite_generation import graph_generate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--checkpoint-cache', type=Path)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', default=64, type=int)
    parser.add_argument('--prompt-length', default=1027, type=int)
    parser.add_argument('--capacity', default=18432, type=int)
    parser.add_argument('--generation-tokens', default=512, type=int)
    parser.add_argument('--native-gqa-backend', default='sdpa', choices=['sdpa', 'flash_attn_kvcache'])
    a = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model = build_model(a.model, a.variant, 'cuda', a.checkpoint, checkpoint_cache=a.checkpoint_cache)
    enable_runtime_sequence_attention(model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token_id = 151643
    fixture = yaml.safe_load((Path(__file__).parents[1] / 'prompts/limite_rl_canary_v1.yaml').read_text())
    prompt = tokenizer.apply_chat_template(fixture['messages'], tokenize=True, return_dict=False, add_generation_prompt=True)
    ids = torch.tensor([prompt], device='cuda').repeat(4, a.prompt_length // len(prompt) + 1)[:, :a.prompt_length]
    tokens = torch.full((4, 1), tokenizer.encode('42', add_special_tokens=False)[0], device='cuda')
    report = dict(variant=a.variant, steps=a.steps, prompt_length=a.prompt_length, capacity=a.capacity,
                  checkpoint=str(a.checkpoint), native_gqa_backend=a.native_gqa_backend,
                  measurements={}, numerical={}, runtime=runtime_contract())
    with torch.no_grad():
        output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        source = output.past_key_values
        del output
        original_reference = GraphDecoder(model, source, tokens, a.capacity)
        for label, enabled in [('original_graph', False), ('native_gqa_graph', True)]:
            if enabled:
                set_native_decode_gqa(model, backend=a.native_gqa_backend)
            started = time.perf_counter()
            decoder = GraphDecoder(model, source, tokens, a.capacity)
            torch.cuda.synchronize()
            capture = time.perf_counter() - started
            for i in range(8):
                decoder(tokens, a.prompt_length + i)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for i in range(a.steps):
                decoder(tokens, a.prompt_length + 8 + i)
            torch.cuda.synchronize()
            seconds = (time.perf_counter() - started) / a.steps
            report['measurements'][label] = dict(decode_ms=seconds * 1000, capture_seconds=capture)
            del decoder
            print(json.dumps(dict(phase=label, **report['measurements'][label])), flush=True)
        pool = GraphDecoderPool(model)
        pool.synchronize()
        decoder = pool.get(source, tokens, a.capacity)
        decoder(tokens, a.prompt_length)
        torch.cuda.synchronize()
        started = time.perf_counter()
        reused = pool.get(source, tokens, a.capacity)
        torch.cuda.synchronize()
        report['measurements']['pool_reuse'] = dict(reset_seconds=time.perf_counter() - started,
                                                  same_graph=reused is decoder, hits=pool.hits)
        errors = []
        native_errors = []
        compact_native = copy.deepcopy(source)
        migration_errors = []
        full_reference = GraphDecoder(model, source, tokens, a.capacity)
        eager = source
        rows = torch.arange(4, device='cuda')
        row_count = 0
        compact_seconds = 0.0
        for i in range(a.steps):
            if i and i % (a.steps // 4) == 0 and rows.numel() > 1:
                kept = torch.arange(1, len(rows), device='cuda')
                view = cache_rows(decoder.cache, kept,
                                  length=a.prompt_length + i)
                select_native_rows(compact_native, kept)
                rows = rows[1:]
                decoder = pool.get(view, tokens[rows], a.capacity)
                migration_reference = GraphDecoder(model, view, tokens[rows], a.capacity)
            else:
                migration_reference = None
            torch.cuda.synchronize()
            started = time.perf_counter()
            actual = decoder(tokens[rows], a.prompt_length + i).clone()
            if migration_reference is not None:
                migrated = migration_reference(tokens[rows], a.prompt_length + i).clone()
                migration_errors.append(float((actual - migrated).abs().max()))
                del migration_reference
            torch.cuda.synchronize()
            compact_seconds += time.perf_counter() - started
            expected = full_reference(tokens, a.prompt_length + i).clone()
            original = original_reference(tokens, a.prompt_length + i).clone()
            native = model(input_ids=tokens, past_key_values=eager, use_cache=True,
                           logits_to_keep=1).logits[:, -1]
            native_compact = model(input_ids=tokens[rows], past_key_values=compact_native,
                                   use_cache=True, logits_to_keep=1).logits[:, -1]
            errors.append(distribution_error(actual, expected[rows]))
            native_errors.append(dict(gqa_vs_original=distribution_error(expected, original),
                                      original_vs_eager=distribution_error(original, native),
                                      compaction=errors[-1],
                                      native_batch_shape=distribution_error(native_compact, native[rows]),
                                      compact_vs_native_compact=distribution_error(actual, native_compact),
                                      optimized_vs_eager=distribution_error(actual, native[rows])))
            row_count += len(rows)
        report['numerical']['compaction'] = dict(
            max_weighted_error=max(x['weighted_error'] for x in errors), max_kl=max(x['kl'] for x in errors),
            mean_weighted_error=sum(x['weighted_error'] for x in errors) / len(errors),
            migration_max_abs_logit_error=max(migration_errors, default=0.0))
        report['numerical']['decode_controls'] = native_errors
        report['numerical']['native_eager_agreement'] = dict(
            mean_weighted_error=sum(distribution['optimized_vs_eager']['weighted_error'] for distribution in native_errors) / len(native_errors),
            mean_kl=sum(distribution['optimized_vs_eager']['kl'] for distribution in native_errors) / len(native_errors),
        )
        report['measurements']['compaction'] = dict(decode_seconds=compact_seconds,
                                                  rows=row_count, full_rows=4 * a.steps)
        # Exercise changes in folded attention weights, gates and MUDD buffers.
        model.train()
        for name, parameter in model.named_parameters():
            if ('qkv' in name or 'gate' in name or 'dense' in name or 'residual_scales' in name):
                parameter.mul_(1.03)
        model.eval()
        fresh = model(input_ids=ids, use_cache=True, logits_to_keep=1).past_key_values
        pool.synchronize()
        reused = pool.get(fresh, tokens, a.capacity)
        actual = reused(tokens, a.prompt_length).clone()
        fresh_decoder = GraphDecoder(model, fresh, tokens, a.capacity)
        expected = fresh_decoder(tokens, a.prompt_length).clone()
        report['numerical']['weight_refresh'] = distribution_error(actual, expected)
        report['numerical']['weight_refresh']['max_abs_logit_error'] = float((actual - expected).abs().max())
        del fresh_decoder, decoder, reused, fresh, source, eager, original_reference, full_reference
        config = GenerationConfig(do_sample=True, temperature=1., top_p=1., top_k=0,
                                  max_new_tokens=a.generation_tokens, eos_token_id=[151643, 151645])
        stats = {}
        completion, behavior, lengths, reasons, captured = graph_generate(
            model, ids, config, tokenizer, lambda: False, pool=pool, compact=True,
            generator=torch.Generator(device='cuda').manual_seed(1234), stats=stats)
        model.train()
        deltas = []
        for row, length in enumerate(lengths):
            sequence = torch.cat([ids[row:row + 1], completion[row:row + 1, :length]], dim=1)
            logits = model(input_ids=sequence, use_cache=False, logits_to_keep=length + 1).logits[:, :-1].float()
            replay = logits.log_softmax(-1).gather(-1, completion[row:row + 1, :length, None]).squeeze()
            deltas.append(replay - behavior[row, :length])
        delta = torch.cat(deltas)
        ratio = delta.exp()
        report['numerical']['sampling_replay'] = dict(
            mean_abs_error=float(delta.abs().mean()), min_ratio=float(ratio.min()), max_ratio=float(ratio.max()),
            clip_fraction=float(((ratio < .8) | (ratio > 1.2)).float().mean()))
        report['generation'] = dict(lengths=lengths, reasons=reasons, capture_seconds=captured, **stats)
    measurements = report['measurements']
    measurements['decode_speedup'] = measurements['original_graph']['decode_ms'] / measurements['native_gqa_graph']['decode_ms']
    report['passed'] = (report['numerical']['native_eager_agreement']['mean_weighted_error'] < .02
                        and report['numerical']['native_eager_agreement']['mean_kl'] < .001
                        and report['numerical']['compaction']['max_weighted_error'] < .02
                        and report['numerical']['compaction']['max_kl'] < .001
                        and report['numerical']['weight_refresh']['max_abs_logit_error'] == 0
                        and report['numerical']['sampling_replay']['clip_fraction'] <= .05
                        and .5 <= ratio.min() <= ratio.max() <= 2
                        and measurements['pool_reuse']['same_graph'])
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'runtime'}), flush=True)
    if not report['passed']:
        raise AssertionError('native decode optimization failed numerical admission')


if __name__ == '__main__':
    main()
