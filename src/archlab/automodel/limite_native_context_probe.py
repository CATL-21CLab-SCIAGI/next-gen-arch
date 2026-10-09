"""Qualify native-context RL replay, head gradients, and resident actor caches."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main():
    from archlab.architectures.limite_decode import GraphDecoderPool
    from archlab.architectures.limite_gqa import set_native_decode_gqa
    from archlab.architectures.limite_replay import enable_native_replay
    from archlab.automodel.limite_adapter_common import runtime_contract
    from archlab.automodel.limite_adapter_rl import gradient_evidence, optimizer_for_model
    from archlab.automodel.limite_native_checkpoint import build_native_model
    from archlab.rl.limite_actor import policy_snapshot
    from archlab.rl.limite_scoring import enable_chunked_policy_scores, native_replay_scores

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[16384, 131072])
    parser.add_argument("--parity-sequence-length", type=int, default=128)
    parser.add_argument("--resident-actor", action="store_true")
    parser.add_argument("--capacity-only", action="store_true",
                        help="diagnose memory after recording parity; never emit a passing qualification")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = dict(passed=False, qualification_only=True, capacity_only=args.capacity_only,
                  checkpoint=str(args.checkpoint), stages=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(stage, **values):
        row = dict(stage=stage, time=time.time(), **values)
        report["stages"].append(row)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)

    model = build_native_model(args.model, "cuda", args.checkpoint, checkpoint_cache=args.checkpoint_cache)
    model.train()
    if not 2 <= args.parity_sequence_length <= model.config.max_position_embeddings:
        raise ValueError("parity sequence exceeds publisher context")
    keep = args.parity_sequence_length - 1
    ids = torch.randint(100, 10000, (1, args.parity_sequence_length), device="cuda")
    weights = torch.linspace(-1, 1, keep, device="cuda")[None]
    logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
    expected = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    (expected * weights).mean().backward()
    reference = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
    expected = expected.detach()
    model.zero_grad(set_to_none=True)
    del logits
    set_native_decode_gqa(model, backend="flash_attn_kvcache")
    enable_native_replay(model)
    report["native_replay"] = model.archlab_native_replay
    enable_chunked_policy_scores(model)
    actual, _ = native_replay_scores(model, ids, torch.ones_like(ids), keep, temperature=1.0, compute_entropy=True)
    (actual * weights).mean().backward()
    errors = dict(squared_error=0.0, squared_reference=0.0, squared_actual=0.0, dot=0.0)
    if set(reference) != {name for name, value in model.named_parameters() if value.grad is not None}:
        raise AssertionError("native replay changed the gradient parameter set")
    for name, parameter in model.named_parameters():
        if name not in reference:
            continue
        want, got = reference[name].float(), parameter.grad.detach().cpu().float()
        errors["squared_error"] += float((got - want).square().sum())
        errors["squared_reference"] += float(want.square().sum())
        errors["squared_actual"] += float(got.square().sum())
        errors["dot"] += float((want * got).sum())
    relative = (errors["squared_error"] / errors["squared_reference"]) ** .5
    cosine = errors["dot"] / (errors["squared_reference"] * errors["squared_actual"]) ** .5
    max_error = float((actual.detach() - expected).abs().max())
    record("publisher_numerical_parity", tokens=args.parity_sequence_length,
           gradient_relative_l2=relative, gradient_cosine=cosine,
           selected_logprob_max_error=max_error, parameter_gradients=len(reference))
    parity_passed = relative <= .08 and cosine >= .995 and max_error <= .15
    report["parity_passed"] = parity_passed
    if not parity_passed and not args.capacity_only:
        raise AssertionError("native replay exceeded the declared BF16 forward/backward tolerance")
    del reference, expected, actual, ids
    model.zero_grad(set_to_none=True)
    optimizer = optimizer_for_model(model)
    actor = pool = snapshot = None
    if args.resident_actor:
        actor = build_native_model(args.model, "cuda", args.checkpoint, checkpoint_cache=args.checkpoint_cache)
        actor.requires_grad_(False).eval()
        set_native_decode_gqa(actor, backend="flash_attn_kvcache")
        snapshot = policy_snapshot(model, 0)
        pool = GraphDecoderPool(actor, max_entries=8)
        with torch.no_grad():
            pool.synchronize()
            for batch in range(1, 5):
                token = torch.full((batch, 1), 100, device="cuda", dtype=torch.long)
                source = actor(input_ids=token, use_cache=True, logits_to_keep=1).past_key_values
                decoder = pool.get(source, token, actor.config.max_position_embeddings)
                if not torch.isfinite(decoder(token, actor.config.max_position_embeddings - 1)).all():
                    raise AssertionError("native context final cache position produced nonfinite logits")
            pool.freeze_capture()
        record("resident_actor", entries=len(pool.entries), graph_captures=pool.misses,
               context=actor.config.max_position_embeddings,
               allocated_gib=torch.cuda.memory_allocated() / 2**30)
    for length in args.sequence_lengths:
        if not 2 <= length <= model.config.max_position_embeddings:
            raise ValueError("qualification sequence exceeds publisher context")
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(100, 10000, (1, length), device="cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        scores, entropy = native_replay_scores(model, ids, torch.ones_like(ids), length - 1,
                                               temperature=1.0, compute_entropy=True)
        loss = -scores.mean()
        loss.backward()
        evidence = gradient_evidence(model)
        if not torch.isfinite(scores).all() or not torch.isfinite(entropy).all() or evidence["backbone_gradient_norm"] <= 0:
            raise AssertionError("long-context replay lacks finite scores or useful gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        record("long_replay", tokens=length, seconds=time.perf_counter() - started, loss=float(loss.detach()),
               peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
               peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
               free_gib=free / 2**30, total_gib=total / 2**30, **evidence)
        del ids, scores, entropy, loss
    report.update(passed=parity_passed and not args.capacity_only, runtime=runtime_contract(),
                  capacity_completed=True, capacity_context=max(args.sequence_lengths), resident_actor=actor is not None,
                  qualified_context=0 if args.capacity_only else max(args.sequence_lengths),
                  resident_snapshot=snapshot is not None, resident_graphs=len(pool.entries) if pool is not None else 0)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
