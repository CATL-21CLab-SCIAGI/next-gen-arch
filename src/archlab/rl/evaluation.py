"""Exact-budget greedy evaluation of frozen held-out scalar-math examples."""

import hashlib
import json
import random
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist

from archlab.rl.rewards import REWARD_BACKEND, canonical_math_answer, verify_math_answer
from archlab.rl.rollout import sample_rollouts


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _gather(value):
    if not dist.is_initialized():
        return [value]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def evaluate_policy(
    model,
    tokenizer,
    examples,
    *,
    policy_version,
    max_new_tokens,
    context_limit,
    eos_token_ids,
    pad_token_id,
    eval_count,
    local_batch_size,
    seed,
):
    """Return ``(global_summary, records)`` for exactly the first eval_count rows.

    Every rank receives the same ordered example list. Each row has problem_id,
    prompt_ids (the complete native prompt, without its reference), and
    expected_answer (a supported scalar). Rank zero returns all records in frozen
    input order; other ranks return only their own scored rows. Dummy rows keep
    collective call counts/batch shapes aligned and never enter scores or records.

    This evaluates actual greedy generations with the supplied model, never
    teacher-forces references or updates weights. Prompt construction and split
    exclusion happen upstream. Raw generated IDs and decoded completion text are
    recorded so the reported reward can be independently checked.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    device = model.lm_head.weight.device
    error = None
    try:
        if type(eval_count) is not int or not 0 < eval_count <= len(examples):
            raise ValueError("eval_count must select a nonempty exact prefix of examples")
        if type(local_batch_size) is not int or local_batch_size < 1:
            raise ValueError("local_batch_size must be a positive integer")
        selected = examples[:eval_count]
        ids = [row["problem_id"] for row in selected]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != eval_count:
            raise ValueError("the frozen evaluation prefix needs distinct nonempty problem IDs")
        if any(canonical_math_answer(row["expected_answer"]) is None for row in selected):
            raise ValueError("evaluation references must be supported scalar answers")
        prompts = [{"problem_id": row["problem_id"], "prompt_ids": list(row["prompt_ids"])}
                   for row in selected]
        vocabulary = model.lm_head.weight.shape[0]
        for row in prompts:
            tokens = row["prompt_ids"]
            if (not tokens or any(type(token) is not int or not 0 <= token < vocabulary for token in tokens)
                    or len(tokens) + max_new_tokens > context_limit):
                raise ValueError("invalid or over-budget evaluation prompt tokens")
        prompt_digest = _digest(prompts)
        reference_digest = _digest([{"problem_id": row["problem_id"],
                                     "expected_answer": row["expected_answer"]} for row in selected])
        configuration = (prompt_digest, reference_digest, policy_version, eval_count,
                         local_batch_size, max_new_tokens, context_limit,
                         tuple(sorted(eos_token_ids)), pad_token_id, seed)
    except (KeyError, TypeError, ValueError) as caught:
        error, configuration = str(caught), None
    packets = _gather((error, configuration))
    if any(packet[0] for packet in packets):
        raise ValueError(f"invalid distributed evaluation inputs: {[p[0] for p in packets]}")
    if any(packet[1] != configuration for packet in packets):
        raise ValueError("evaluation split, references, or settings differ across ranks")

    # Preserve all RNGs that the existing training/data pipeline may own.
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
    modes = [(module, module.training) for module in model.modules()]
    local_records = []
    rollout_receipts = []
    global_batch = local_batch_size * world
    batches = (eval_count + global_batch - 1) // global_batch
    try:
        model.eval()
        with torch.no_grad():
            for batch_index in range(batches):
                indices = [batch_index * global_batch + rank * local_batch_size + slot
                           for slot in range(local_batch_size)]
                rows = [selected[index] if index < eval_count else selected[0] for index in indices]
                result = sample_rollouts(
                    model, [row["prompt_ids"] for row in rows], policy_version=policy_version,
                    max_new_tokens=max_new_tokens, context_limit=context_limit,
                    eos_token_ids=eos_token_ids, pad_token_id=pad_token_id, seed=seed + batch_index,
                    temperature=0., top_p=1.,
                    prompt_group_ids=[row["problem_id"] if index < eval_count else "__eval_dummy__"
                                      for row, index in zip(rows, indices, strict=True)],
                )
                rollout_receipts.append(result.receipt)
                for slot, (index, row) in enumerate(zip(indices, rows, strict=True)):
                    if index >= eval_count:
                        continue
                    tokens = result.generated_ids[slot]
                    # Remove only the known terminal ID, preserving reasoning delimiters.
                    text_ids = tokens[:-1] if tokens and tokens[-1] in eos_token_ids else tokens
                    completion = tokenizer.decode(text_ids, skip_special_tokens=False)
                    reward = verify_math_answer(completion, row["expected_answer"])
                    local_records.append({
                        "eval_index": index, "problem_id": row["problem_id"],
                        "policy_version": policy_version, "rank": rank,
                        "prompt_ids_sha256": _digest(list(row["prompt_ids"])),
                        "generated_ids": list(tokens), "completion": completion,
                        "expected_answer": row["expected_answer"], "verification": asdict(reward),
                        "correct": reward.correct, "valid_answer": reward.canonical_answer is not None,
                        "finish_reason": result.finish_reasons[slot],
                        "truncated": result.finish_reasons[slot] == "length",
                    })
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state(rng[3], device)
        for module, training in modes:
            module.training = training

    records = sorted([row for partition in _gather(local_records) for row in partition],
                     key=lambda row: row["eval_index"])
    if [row["problem_id"] for row in records] != ids:
        raise RuntimeError("evaluation did not score each frozen problem exactly once")
    correct = sum(row["correct"] for row in records)
    valid = sum(row["valid_answer"] for row in records)
    truncated = sum(row["truncated"] for row in records)
    receipts = _gather(rollout_receipts)
    summary = {
        "format": "archlab-math-greedy-eval-v1", "policy_version": policy_version,
        "count": len(records), "correct": correct, "pass_at_1": correct / eval_count,
        "valid_answers": valid, "valid_answer_rate": valid / eval_count,
        "truncated": truncated, "truncation_rate": truncated / eval_count,
        "prompt_split_digest": prompt_digest, "reference_digest": reference_digest,
        "problem_ids_digest": _digest(ids), "reward_backend": REWARD_BACKEND,
        "temperature": 0., "top_p": 1., "max_new_tokens": max_new_tokens,
        "context_limit": context_limit, "seed": seed, "world_size": world,
        "local_batch_size": local_batch_size, "rollout_batches": batches,
        "dummy_rows_excluded": batches * global_batch - eval_count,
        "rng_restored": True, "records_scope": "global-on-rank-zero-local-otherwise",
        "rollout_receipts_by_rank": receipts,
    }
    return summary, records if rank == 0 else local_records
