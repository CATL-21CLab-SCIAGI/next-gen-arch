"""Native unwarped policy sampling with graph decode and explicit termination."""

from __future__ import annotations

import time

import torch

from archlab.architectures.limite_decode_state import cache_rows
from archlab.rl.limite_protocol import degenerate_repetition, response_budget
from archlab.rl.limite_rollout import SamplingLogprobs


@torch.no_grad()
def graph_generate(model, ids, config, tokenizer, stop_requested, *, capacity=None,
                   pool=None, compact=False, generator=None, stats=None, diagnostics=None):
    from archlab.architectures.limite_decode import GraphDecoder

    # Validate exactly the same neutral sampling contract as the eager path.
    SamplingLogprobs(config)
    batch, prompt_length = ids.shape
    budget = response_budget(config, prompt_length, model.config.max_position_embeddings)
    completions = ids.new_full((batch, budget), tokenizer.pad_token_id)
    logprobs = torch.zeros((batch, budget), dtype=torch.float32, device=ids.device)
    active = torch.ones(batch, dtype=torch.bool, device=ids.device)
    endings = torch.tensor(config.eos_token_id, device=ids.device).reshape(-1)
    eos_at = ids.new_full((batch,), -1)
    repeat_at = ids.new_full((batch,), -1)
    if diagnostics is not None:
        diagnostics.mark("actor", "prefill", prompt_tokens=prompt_length, batch=batch, budget=budget)
    if pool is not None:
        pool.synchronize()
    captures_before = pool.capture_seconds if pool is not None else 0.0
    hits_before = pool.hits if pool is not None else 0
    misses_before = pool.misses if pool is not None else 0
    logits_output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    logits = logits_output.logits[:, -1].float()
    decoder = None
    rows = torch.arange(batch, device=ids.device)
    decode_rows = 0
    graph_seconds = 0.0
    stopped = False
    deadline = 0.0
    for step in range(budget):
        if diagnostics is not None and step % 512 == 0:
            diagnostics.mark("actor", "decode", token_index=step, prompt_tokens=prompt_length,
                             batch=batch, decode_batch=rows.numel())
        now = time.monotonic()
        if now >= deadline:
            stopped = stop_requested()
            deadline = now + 0.5
            if stopped and step > 0:
                break
        probabilities = logits.softmax(-1)
        # Retain full logical row order and RNG draw geometry after compaction.
        token = torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)
        selected = logits.log_softmax(-1).gather(-1, token[:, None]).squeeze(-1)
        token = torch.where(active, token, tokenizer.pad_token_id)
        completions[:, step].copy_(token)
        logprobs[:, step].copy_(torch.where(active, selected, 0.0))
        hit_eos = active & (token[:, None] == endings).any(-1)
        eos_at.masked_fill_(hit_eos, step)
        active &= ~hit_eos
        if (step + 1) % 512 == 0:
            texts = tokenizer.batch_decode(completions[:, :step + 1], skip_special_tokens=True)
            repeated = torch.tensor([degenerate_repetition(text) for text in texts], device=ids.device)
            hit_repeat = active & repeated
            repeat_at.masked_fill_(hit_repeat, step)
            active &= ~hit_repeat
        active_count = int(active.sum())
        if active_count == 0 or step + 1 == budget:
            step += 1
            break
        next_rows = active.nonzero().flatten() if compact and active_count != rows.numel() else rows
        if decoder is None or next_rows.numel() != rows.numel():
            if diagnostics is not None:
                diagnostics.mark("actor", "cache_reset", token_index=step, decode_batch=next_rows.numel(),
                                 prompt_tokens=prompt_length, requested_capacity=capacity or prompt_length + budget)
            started = time.perf_counter()
            if decoder is None:
                source = logits_output.past_key_values
                if next_rows.numel() != batch:
                    source = cache_rows(source, next_rows)
                del logits_output
            else:
                # The sampled token is still to be appended; select histories
                # through the last processed token, in logical row order.
                selected_rows = torch.searchsorted(rows, next_rows)
                source = cache_rows(decoder.cache, selected_rows, length=prompt_length + step)
            selected_token = token.index_select(0, next_rows)[:, None]
            decoder = (pool.get(source, selected_token, capacity or prompt_length + budget)
                       if pool is not None else
                       GraphDecoder(model, source, selected_token, capacity or prompt_length + budget))
            if pool is None:
                graph_seconds += time.perf_counter() - started
            rows = next_rows
        selected_token = token[:, None] if rows.numel() == batch else token.index_select(0, rows)[:, None]
        decoded = decoder(selected_token, prompt_length + step).float()
        decode_rows += rows.numel()
        if compact and rows.numel() != batch:
            logits = torch.zeros_like(logits).index_copy_(0, rows, decoded)
        else:
            logits = decoded
    else:
        step = budget
    lengths, reasons = [], []
    if diagnostics is not None:
        diagnostics.mark("actor", "termination_transfer", token_index=step, batch=batch)
    for eos, repeat in zip(eos_at.tolist(), repeat_at.tolist(), strict=True):
        if eos >= 0:
            lengths.append(eos + 1)
            reasons.append("eos")
        elif repeat >= 0:
            lengths.append(repeat + 1)
            reasons.append("repetition")
        else:
            lengths.append(step)
            reasons.append("stop" if stopped else "length")
    if pool is not None:
        graph_seconds = pool.capture_seconds - captures_before
    if stats is not None:
        stats.update(decode_rows=decode_rows, uncompressed_decode_rows=batch * max(0, step - 1),
                     graph_pool_hits=pool.hits - hits_before if pool is not None else 0,
                     graph_pool_misses=pool.misses - misses_before if pool is not None else 0)
    return completions[:, :step], logprobs[:, :step], lengths, reasons, graph_seconds
