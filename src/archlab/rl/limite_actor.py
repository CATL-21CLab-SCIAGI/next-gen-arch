"""Immutable native policy snapshots on a separate rollout CUDA stream."""

from __future__ import annotations

import time

import torch

from archlab.rl.async_rollout import AsyncRolloutQueue, PolicySnapshot


def policy_snapshot(model, version):
    # Clone on the learner stream before the next forward/backward can mutate
    # parameters; the actor never reads live trainable storage.
    weights = {name: value.detach().clone() for name, value in model.named_parameters()}
    ready = torch.cuda.Event()
    ready.record()
    return PolicySnapshot(version, weights, ready)


def _decode_capacities(budget, prompt_limit, quantum, context_limit, *, budget_mode="fixed_response"):
    if budget_mode == "native_context":
        if budget != context_limit or not 0 < prompt_limit < context_limit:
            raise ValueError("native actor prewarm requires the complete model context")
        return (context_limit,)
    if min(budget, prompt_limit, quantum) < 1 or budget + prompt_limit > context_limit:
        raise ValueError("actor prewarm exceeds the native prompt/context contract")
    start = ((budget + 1 + quantum - 1) // quantum) * quantum
    stop = ((budget + prompt_limit + quantum - 1) // quantum) * quantum
    return tuple(min(capacity, context_limit) for capacity in range(start, stop + 1, quantum))


def native_actor_queue(actor, learner, trainer, version, fixture_generate=None):
    from archlab.architectures.limite_decode import GraphDecoderPool
    from archlab.rl.limite_rollout import generate_native_batch
    from archlab.rl.rollout_diagnostics import RolloutDiagnostics

    device = next(actor.parameters()).device
    stream = torch.cuda.Stream(device=device)
    capacities = (() if fixture_generate is not None else _decode_capacities(
        trainer.generation_config.max_new_tokens, 2048, 1024, actor.config.max_position_embeddings,
        budget_mode=getattr(trainer.generation_config, "archlab_budget_mode", "fixed_response"),
    ))
    pool = GraphDecoderPool(actor, max_entries=max(8, len(capacities) * trainer.args.num_generations))
    actor_version = None
    generator = torch.Generator(device=device)
    generator.set_state(torch.cuda.get_rng_state(device))
    diagnostics = None

    def load_policy(snapshot):
        nonlocal actor_version
        if diagnostics is not None:
            diagnostics.mark("actor", "policy_event_wait", policy_version=snapshot.version)
        stream.wait_event(snapshot.ready)
        if actor_version != snapshot.version:
            if diagnostics is not None:
                diagnostics.mark("actor", "policy_copy", policy_version=snapshot.version)
            values = dict(actor.named_parameters())
            if values.keys() != snapshot.weights.keys():
                raise ValueError("actor and learner parameter topology differs")
            for name, value in values.items():
                value.copy_(snapshot.weights[name])
            # Publisher eval folds must be rebuilt after copying weights.
            if diagnostics is not None:
                diagnostics.mark("actor", "policy_fold", policy_version=snapshot.version)
            actor.train().eval()
            actor_version = snapshot.version

    initial = policy_snapshot(learner, version)
    # Training and heldout data both filter prompts to <=2048 tokens. Capture
    # every corresponding capacity/compacted batch before the executor or
    # learner starts, then forbid background capture. Preparation samples no
    # tokens and preserves the learner RNG as well as the independent actor RNG.
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    started = time.perf_counter()
    try:
        with torch.cuda.stream(stream), torch.no_grad():
            load_policy(initial)
            if fixture_generate is None:
                pool.synchronize()
                for batch in range(1, trainer.args.num_generations + 1):
                    ids = torch.zeros((batch, 1), device=device, dtype=torch.long)
                    source = actor(input_ids=ids, use_cache=True, logits_to_keep=1).past_key_values
                    for capacity in capacities:
                        pool.get(source, ids, capacity)
            pool.freeze_capture()
            stream.synchronize()
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
    prewarm_seconds = time.perf_counter() - started
    startup_entries = len(pool.entries)
    diagnostics = RolloutDiagnostics.from_environment()

    def generate(prompts, snapshot, rng):
        torch.cuda.set_device(device)
        with torch.cuda.stream(stream), torch.no_grad():
            if diagnostics is not None:
                diagnostics.mark("actor", "policy_load", policy_version=snapshot.version)
            load_policy(snapshot)
            eager_before = pool.eager_misses
            if fixture_generate is not None:
                result = fixture_generate(prompts, actor)
                timings = dict(generation_seconds=0.0, graph_capture_seconds=0.0)
            else:
                result, timings = generate_native_batch(
                    prompts, actor, trainer.processing_class, trainer.generation_config,
                    queue.stop_requested, groups=trainer.args.num_generations,
                    graph=True, pool=pool, compact=trainer.archlab_compact_decode, generator=rng,
                    diagnostics=diagnostics,
                )
            timings["graph_eager_misses"] = pool.eager_misses - eager_before
            timings["graph_startup_seconds"] = prewarm_seconds
            timings["graph_startup_entries"] = startup_entries
            if diagnostics is not None:
                diagnostics.mark("actor", "stream_wait", policy_version=snapshot.version)
            stream.synchronize()
        return result, timings

    queue = AsyncRolloutQueue(generate, generator, initial,
                             trainer.archlab_stop_requested, trainer.archlab_max_policy_lag,
                             diagnostics=diagnostics)
    return queue
