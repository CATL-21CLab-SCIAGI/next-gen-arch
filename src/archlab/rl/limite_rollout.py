"""Record native generation probabilities for upstream GRPO importance ratios."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path


def with_behavior_logprobs(inputs):
    import torch

    behavior, mask = inputs["sampling_per_token_logps"], inputs["completion_mask"]
    if behavior.ndim != 2 or behavior.shape != mask.shape or behavior.device != mask.device:
        raise ValueError("behavior probabilities must match the completion mask")
    if not behavior.is_floating_point() or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("behavior probabilities require floating scores and a binary mask")
    active = behavior[mask.bool()]
    if not bool(torch.isfinite(active).all()) or bool((active > 1e-5).any()):
        raise FloatingPointError("invalid sampled behavior log probabilities")
    # Padding never had a sampled distribution. Give it a neutral denominator
    # so the upstream exp-ratio followed by multiplication by zero cannot turn
    # an ignored NaN/-inf padding score into a nonfinite loss.
    old = behavior.detach().masked_fill(~mask.bool(), 0.0)
    return dict(inputs, old_per_token_logps=old)


class SamplingLogprobs:
    """Keep one distribution, then record only the token actually sampled.

    Transformers appends sampling warpers after user logits processors. The
    qualified native contract has no such warpers; reject a changed contract
    instead of recording probabilities from the wrong distribution.
    """

    def __init__(self, generation_config):
        expected = {"temperature": 1.0, "top_p": 1.0, "top_k": 0}
        if any(getattr(generation_config, key, value) != value for key, value in expected.items()):
            raise ValueError("streamed behavior probabilities require native unwarped sampling")
        if (
            getattr(generation_config, "do_sample", True) is False
            or
            getattr(generation_config, "num_beams", None) not in (None, 1)
            or
            any(getattr(generation_config, key, None) is not None for key in ("min_p", "top_h", "watermarking_config"))
            or getattr(generation_config, "typical_p", None) not in (None, 1.0)
            or any(getattr(generation_config, key, None) not in (None, 0.0) for key in ("epsilon_cutoff", "eta_cutoff"))
            or getattr(generation_config, "renormalize_logits", False)
            or getattr(generation_config, "repetition_penalty", None) not in (None, 1.0)
            or getattr(generation_config, "encoder_repetition_penalty", None) not in (None, 1.0)
            or any(getattr(generation_config, key, None) not in (None, 0) for key in (
                "min_length", "min_new_tokens", "no_repeat_ngram_size", "encoder_no_repeat_ngram_size",
            ))
            or any(getattr(generation_config, key, None) is not None for key in (
                "forced_bos_token_id", "forced_eos_token_id", "bad_words_ids",
                "suppress_tokens", "begin_suppress_tokens", "sequence_bias", "guidance_scale",
            ))
        ):
            raise ValueError("sampling processors would change recorded behavior probabilities")
        self.previous = None
        self.selected = []
        self.max_distribution_bytes = 0

    def _select(self, input_ids):
        if self.previous is not None:
            self.selected.append(self.previous.gather(-1, input_ids[:, -1:]).squeeze(-1))

    def __call__(self, input_ids, scores):
        self._select(input_ids)
        self.previous = scores.float().log_softmax(-1)
        self.max_distribution_bytes = max(
            self.max_distribution_bytes, self.previous.numel() * self.previous.element_size()
        )
        return scores

    def finish(self, sequences):
        import torch

        self._select(sequences)
        if not self.selected:
            raise ValueError("native generation did not sample any tokens")
        result = torch.stack(self.selected, dim=1)
        self.previous = None
        self.selected.clear()
        return result


def generate_native_batch(prompts, model, tokenizer, config, stop_requested, *, groups,
                          graph=False, pool=None, compact=False, generator=None, diagnostics=None):
    import torch

    from archlab.rl.limite_protocol import response_budget

    result = dict(prompt_ids=[], completion_ids=[], logprobs=[], finish_reason=[], completion_budget=[])
    timings = dict(generation_seconds=0.0, selection_seconds=0.0,
                   sampling_score_buffer_bytes=0, graph_capture_seconds=0.0,
                   decode_rows=0, uncompressed_decode_rows=0,
                   graph_pool_hits=0, graph_pool_misses=0)
    device = next(model.parameters()).device
    for start in range(0, len(prompts), groups):
        group = prompts[start:start + groups]
        if len(set(group)) != 1:
            raise ValueError("native rollout expects contiguous prompt groups")
        if diagnostics is not None:
            diagnostics.mark("actor", "tokenize", group_start=start, batch=len(group))
        inputs = tokenizer(group, return_tensors="pt", padding=False).to(device)
        budget = response_budget(config, inputs["input_ids"].shape[1], model.config.max_position_embeddings)
        recorder = SamplingLogprobs(config)
        started = time.perf_counter()
        with torch.no_grad():
            if graph:
                from archlab.rl.limite_generation import graph_generate

                stats = {}
                completions, selected_tensor, lengths, reasons, captured_seconds = graph_generate(
                    model, inputs["input_ids"], config, tokenizer, stop_requested,
                    pool=pool, compact=compact, generator=generator, stats=stats, diagnostics=diagnostics,
                )
                timings["graph_capture_seconds"] += captured_seconds
                for key, value in stats.items():
                    timings[key] += value
                recorder.max_distribution_bytes = inputs["input_ids"].shape[0] * model.config.vocab_size * 4
            else:
                if generator is not None:
                    raise ValueError("asynchronous sampling requires explicit-generator graph decode")
                local_config = deepcopy(config)
                local_config.max_new_tokens = budget
                output = model.generate(
                    **inputs, generation_config=local_config, return_dict_in_generate=True,
                    output_scores=False, logits_processor=[recorder],
                )
                completions = output.sequences[:, inputs["input_ids"].shape[1]:]
                selected_tensor = recorder.finish(output.sequences)
                lengths, reasons = [], []
                for ids in completions.tolist():
                    end = next((i for i, token in enumerate(ids) if token in (151643, 151645)), None)
                    lengths.append(end + 1 if end is not None else len(ids))
                    reasons.append("eos" if end is not None else "length")
        timings["generation_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        if selected_tensor.shape != completions.shape:
            raise RuntimeError("sampled token and recorded probability clocks differ")
        if diagnostics is not None:
            diagnostics.mark("actor", "cpu_transfer", tokens=completions.shape[1], batch=completions.shape[0])
        selected = selected_tensor.cpu().tolist()
        timings["selection_seconds"] += time.perf_counter() - started
        timings["sampling_score_buffer_bytes"] = max(timings["sampling_score_buffer_bytes"], recorder.max_distribution_bytes)
        for prompt_ids, ids, logps, length, reason in zip(
            inputs["input_ids"].tolist(), completions.tolist(), selected, lengths, reasons, strict=True
        ):
            result["prompt_ids"].append(prompt_ids)
            result["completion_ids"].append(ids[:length])
            result["logprobs"].append(logps[:length])
            result["finish_reason"].append(reason)
            result["completion_budget"].append(budget)
    return result, timings


def native_rollout(prompts, trainer):
    from trl.models.utils import unwrap_model_for_generation

    asynchronous = getattr(trainer, "archlab_async_rollout", None)
    training = trainer.model.training
    if asynchronous is not None and training:
        overlap = getattr(trainer, "archlab_overlap_actor_learner", True)
        schedule = getattr(trainer, "archlab_rollout_prefetch_schedule", "before_update")
        if schedule not in ("before_update", "after_update"):
            raise ValueError("unknown rollout prefetch schedule")
        if schedule == "after_update" and overlap is not False:
            raise ValueError("after_update rollout scheduling requires overlap_actor_learner=False")
        rendezvous = getattr(trainer, "archlab_rollout_rendezvous", None)
        if not overlap and trainer.accelerator.num_processes > 1 and rendezvous is None:
            raise RuntimeError("distributed no-overlap rollout requires a host rendezvous")
        result, timings = asynchronous.consume(prompts, trainer.archlab_policy_version())
        if schedule == "before_update":
            asynchronous.prefetch(getattr(trainer, "archlab_next_rollout_prompts", None))
        if not overlap:
            # Local drain alone is insufficient: faster ranks can otherwise
            # launch waiting NCCL kernels while a peer is still decoding.
            # The after-update schedule starts the next batch on demand at its
            # own consume, after the previous optimizer/checkpoint boundary.
            # A restored legacy pending batch is still consumed exactly once.
            timings["rollout_drain_seconds"] = 0.0
            if schedule == "before_update":
                started = time.perf_counter()
                asynchronous.drain()
                timings["rollout_drain_seconds"] = time.perf_counter() - started
            trainer._metrics["train"]["rollout/drain_seconds"].append(timings["rollout_drain_seconds"])
            started = time.perf_counter()
            if rendezvous is not None:
                rendezvous()
            timings["rollout_peer_wait_seconds"] = time.perf_counter() - started
            trainer._metrics["train"]["rollout/peer_wait_seconds"].append(timings["rollout_peer_wait_seconds"])
        trainer._metrics["train"]["rollout/policy_lag"].append(timings["policy_lag"])
        trainer._metrics["train"]["rollout/wait_seconds"].append(timings["rollout_wait_seconds"])
    else:
        if asynchronous is not None:
            # Heldout decode may capture learner graphs. PyTorch capture cannot
            # overlap CUDA work on the actor thread, even with thread_local mode.
            asynchronous.drain()
        with unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator) as model:
            was_training = model.training
            model.eval()
            try:
                pool = None
                if getattr(trainer, "archlab_reuse_decode", False):
                    from archlab.architectures.limite_decode import GraphDecoderPool

                    pool = getattr(trainer, "archlab_decode_pool", None)
                    if pool is None:
                        pool = trainer.archlab_decode_pool = GraphDecoderPool(model)
                result, timings = generate_native_batch(
                    prompts, model, trainer.processing_class, trainer.generation_config,
                    getattr(trainer, "archlab_stop_requested", lambda: False),
                    groups=trainer.args.num_generations,
                    graph=getattr(trainer, "archlab_graph_decode", False), pool=pool,
                    compact=getattr(trainer, "archlab_compact_decode", False),
                )
            finally:
                model.train(was_training)
        rendezvous = getattr(trainer, "archlab_rollout_rendezvous", None)
        if rendezvous is not None:
            # Heldout responses can span the same native context as training.
            # Admit all ranks on the host before any GRPO metric/NCCL gather.
            rendezvous()
    suffix = f"-rank{trainer.accelerator.process_index}" if trainer.accelerator.num_processes > 1 else ""
    ledger = Path(trainer.args.output_dir).parent / f"behavior{suffix}.jsonl"
    with ledger.open("a") as stream:
        stream.write(json.dumps(dict(step=trainer.state.global_step,
                                    phase="train" if training else "eval", **timings, **result)) + "\n")
    return result
