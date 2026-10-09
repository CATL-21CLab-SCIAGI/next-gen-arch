"""Native actor/learner overlap and exact prefetched-state GPU qualification."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from archlab.architectures.limite_adapter import enable_runtime_sequence_attention
from archlab.architectures.limite_gqa import set_native_decode_gqa
from archlab.automodel.limite_adapter_common import build_model, runtime_contract
from archlab.automodel.limite_adapter_rl import optimizer_for_model, parameter_probe
from archlab.rl.limite_actor import native_actor_queue, policy_snapshot


def main():
    import yaml
    from transformers import AutoTokenizer, GenerationConfig

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'tokenizer', 'checkpoint', 'variant', 'output', 'checkpoint-cache'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--native-gqa-backend', default='sdpa', choices=['sdpa', 'flash_attn_kvcache'])
    a = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    checkpoint = Path(a.checkpoint)
    learner = build_model(a.model, a.variant, 'cuda', checkpoint, checkpoint_cache=a.checkpoint_cache)
    actor = build_model(a.model, a.variant, 'cuda', checkpoint, checkpoint_cache=a.checkpoint_cache)
    for model in (learner, actor):
        enable_runtime_sequence_attention(model)
        set_native_decode_gqa(model, backend=a.native_gqa_backend)
    actor.requires_grad_(False)
    learner.train()
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    tok.pad_token_id = 151643
    fixture = yaml.safe_load((Path(__file__).parents[1] / 'prompts/limite_rl_canary_v1.yaml').read_text())
    prompt = tok.apply_chat_template(fixture['messages'], tokenize=False, add_generation_prompt=True)
    prompts = [prompt] * 4
    trainer = SimpleNamespace(args=SimpleNamespace(num_generations=4), processing_class=tok,
                              generation_config=GenerationConfig(do_sample=True, temperature=1., top_p=1., top_k=0,
                                                                 max_new_tokens=256, eos_token_id=[151643, 151645]),
                              archlab_stop_requested=lambda: False, archlab_compact_decode=True, archlab_max_policy_lag=1)
    queue = native_actor_queue(actor, learner, trainer, 0)
    optimizer = optimizer_for_model(learner)
    initial = parameter_probe(learner)
    queue.prefetch(prompts)
    started = time.perf_counter()
    ids = tok.encode(prompt + '\\boxed{42}<|im_end|>', add_special_tokens=False)
    ids = torch.tensor([ids], device='cuda')
    with torch.enable_grad():
        logits = learner(ids, use_cache=False, logits_to_keep=8).logits.float()
        loss = logits.log_softmax(-1)[:, :-1].gather(-1, ids[:, -7:, None]).mean().neg()
        loss.backward()
    torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # Device-wide synchronization is illegal while the actor captures its
    # independent graph and would also erase the overlap being measured.
    torch.cuda.current_stream().synchronize()
    learner_seconds = time.perf_counter() - started
    queue.publish(policy_snapshot(learner, 1))
    first, timings = queue.consume(prompts, 1)
    actor_probe = parameter_probe(actor)
    assert all(torch.equal(value, initial[name]) for name, value in actor_probe.items())
    assert any(not torch.equal(value, initial[name]) for name, value in parameter_probe(learner).items())
    queue.prefetch(prompts)
    state = queue.checkpoint_state()
    serialized = io.BytesIO()
    torch.save(state, serialized)
    serialized.seek(0)
    state = torch.load(serialized, weights_only=True)
    expected, _ = queue.consume(prompts, 1)
    queue.restore(state)
    restored, _ = queue.consume(prompts, 1)
    assert expected == restored
    rng = queue.generator.get_state().clone()
    queue.prefetch(prompts)
    following, _ = queue.consume(prompts, 1)
    queue.restore(initial_rng=rng)
    repeated, _ = queue.consume(prompts, 1)
    assert following == repeated
    queue.close()
    report = dict(passed=True, variant=a.variant, native_gqa_backend=a.native_gqa_backend,
                  learner_seconds=learner_seconds,
                  first_lengths=[len(x) for x in first['completion_ids']], first_timings=timings,
                  snapshot_isolated=True, learner_updated=True, saved_prefetch_exact=True,
                  next_sample_rng_exact=True, peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                  runtime=runtime_contract())
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'runtime'}), flush=True)


if __name__ == '__main__':
    main()
