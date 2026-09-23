"""Inference-only, request-local cache for the resident V4.1 RL actor.

The first prompt pass uses the unchanged training model. Subsequent positions
execute its existing FSDP-wrapped blocks with one-token attention and bounded
adapter windows. Cache state is discarded after each rollout and after errors.
This path is opt-in until a real B300 replay qualification admits it.
"""

from __future__ import annotations

from dataclasses import replace
from types import MethodType

import torch
from torch.nn import functional as F


def _append(previous, current, limit=None):
    result = current if previous is None else torch.cat((previous, current), dim=1)
    return result[:, -limit:].contiguous() if limit is not None else result.contiguous()


def _adapter_projections(adapter, streams):
    c = adapter.config
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=streams.is_cuda and streams.dtype == torch.bfloat16
    ):
        read = adapter.read_logits.float().softmax(-1)
        x = adapter.input_norm(
            (streams.float() * read[None, None, :, None]).sum(-2).to(streams.dtype)
        )
        batch, length = streams.shape[:2]
        q = adapter.q_norm(adapter.q(x).reshape(batch, length, c.query_heads, c.head_dim))
        shape = (batch, length, c.kv_heads, c.head_dim)
        if hasattr(adapter, "k1"):
            projected = {
                "k1": adapter.k1_norm(adapter.k1(x).reshape(shape)),
                "k2": adapter.k2_norm(adapter.k2(x).reshape(shape)),
                "v1": adapter.v1(x).reshape(shape),
                "v2": adapter.v2(x).reshape(shape),
            }
        else:
            projected = {
                "k": adapter.k_norm(adapter.k(x).reshape(shape)),
                "v": adapter.v(x).reshape(shape),
            }
    return x, q, projected


def _adapter_step(adapter, cache, streams, *, position, replay_canvas):
    from flash_attn import flash_attn_func

    from archlab.architectures.simplicial_decode import simplicial_decode_attention
    from archlab.automodel.deepseek_v41_rl_cache_shapes import pad_at_position

    if streams.shape[1] != 1:
        raise ValueError("cached adapter expects one token")
    c = adapter.config
    if "streams" in cache:
        from archlab.architectures.deepseek_v41_normal_adapter import V41NormalAttentionAdapter

        history = _append(cache["streams"], streams, c.long_window)
        cache["streams"] = history
        padded = streams.new_zeros(streams.shape[0], replay_canvas, *streams.shape[2:])
        padded[:, position + 1 - history.shape[1] : position + 1] = history
        return V41NormalAttentionAdapter.forward(adapter, padded)[
            :, position : position + 1
        ].contiguous()
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=streams.is_cuda and streams.dtype == torch.bfloat16
    ):
        x, q, projected = _adapter_projections(
            adapter, pad_at_position(streams, position, replay_canvas)
        )
        q = q[:, position : position + 1].contiguous()
        projected = {
            name: value[:, position : position + 1].contiguous()
            for name, value in projected.items()
        }
        for name, value in projected.items():
            window = c.short_window if name.endswith("1") else c.long_window
            cache[name] = _append(cache[name], value, window)
        with torch.autocast(streams.device.type, enabled=False):
            if "k" in cache:
                attended = flash_attn_func(
                    q.contiguous(),
                    cache["k"],
                    cache["v"],
                    dropout_p=0.0,
                    softmax_scale=c.head_dim**-0.5,
                    causal=True,
                    window_size=(c.long_window - 1, 0),
                    deterministic=True,
                )
            else:
                attended = simplicial_decode_attention(
                    q.float(),
                    cache["k1"].float(),
                    cache["k2"].float(),
                    cache["v1"].float(),
                    cache["v2"].float(),
                ).to(q.dtype)
        attended = pad_at_position(attended.flatten(-2), position, replay_canvas)
        attended = (attended.float() * adapter.output_gate(x).float().sigmoid()).to(attended.dtype)
        branch = adapter.output(attended)[:, position : position + 1]
        write = 2 * adapter.write_logits.float().sigmoid()
        return (streams.float() + branch.float().unsqueeze(-2) * write[None, None, :, None]).to(
            streams.dtype
        )


def _index_step(
    indexer, hidden, query_latent, latent, angles, compressed_angles, state, previous_keys,
    position, replay_canvas,
):
    from nemo_automodel.components.models.deepseek_v41.attention import (
        _apply_rope,
        _select_candidate_blocks,
    )
    from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache

    keys = state.index_keys
    if indexer.owns_keys:
        new_key = (
            None
            if latent is None
            else quantize_cache(
                _apply_rope(indexer.k_norm(indexer.wk(latent)), compressed_angles),
                format="mxfp4",
                block_size=32,
            )
        )
        keys = _append(previous_keys, new_key) if new_key is not None else previous_keys
    if keys is None:
        raise ValueError("cached CSA2 index keys are missing")
    original_keys = keys
    keys = F.pad(keys, (0, 0, 0, replay_canvas // state.compression_ratio - keys.shape[1]))
    width = keys.shape[1]
    if width == 0:
        indices = torch.empty(hidden.shape[0], 1, 0, device=hidden.device, dtype=torch.long)
        candidates = (
            torch.empty(hidden.shape[0], 1, 0, device=hidden.device, dtype=torch.bool)
            if indexer.is_candidate_source
            else state.candidates
        )
        return replace(state, index_keys=original_keys, topk_indices=indices, candidates=candidates)
    queries = indexer.wq_b(query_latent).unflatten(-1, (indexer.num_heads, indexer.head_dim))
    queries = quantize_cache(_apply_rope(queries, angles), format="mxfp4", block_size=32)
    weights = indexer.weights_proj(hidden) * (indexer.head_dim**-0.5 * indexer.num_heads**-0.5)
    scores = torch.einsum("bshd,btd->bsht", queries, keys)
    scores = (scores.relu_().mul_(weights.unsqueeze(-1))).sum(dim=2)
    visible = (position + 1) // state.compression_ratio
    allowed = torch.arange(width, device=hidden.device) < visible
    if state.compressed_valid is not None:
        allowed = allowed & F.pad(
            state.compressed_valid, (0, width - state.compressed_valid.shape[1]), value=False
        )
    scores = scores.masked_fill(~allowed.view(hidden.shape[0], 1, width), -torch.inf)
    candidates = state.candidates
    if indexer.is_candidate_source:
        counts = torch.full(
            (hidden.shape[0], 1, 1), visible, device=hidden.device, dtype=torch.long
        )
        candidates = _select_candidate_blocks(
            scores,
            counts,
            topk_blocks=indexer.candidate_topk_blocks,
            block_size=indexer.candidate_block_size,
        )
    elif indexer.uses_candidates:
        if candidates is None or candidates.shape != scores.shape:
            raise ValueError("cached hierarchical CSA2 candidates are missing")
        scores = scores.masked_fill(~candidates, -torch.inf)
    selected = scores.topk(min(indexer.topk, width), dim=-1, sorted=False)
    indices = selected.indices.sort(dim=-1).values
    indices = torch.where(torch.isfinite(scores.gather(-1, indices)), indices, -1)
    return replace(state, index_keys=original_keys, topk_indices=indices, candidates=candidates)


def _attention_step(
    attn, cache, hidden, *, position_ids, state, attention_mask=None, position, replay_canvas
):
    from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sparse_attention
    from nemo_automodel.components.models.deepseek_v41.attention import (
        DeepseekV41AttentionOutput,
        DeepseekV41AttentionState,
        _apply_rope,
    )
    from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache

    batch, length, _ = hidden.shape
    if (
        length != 1
        or attention_mask is not None
        or attn.backend.attn != "tilelang"
        or position_ids.shape != (batch, 1)
        or not torch.equal(position_ids, torch.full_like(position_ids, position))
    ):
        raise ValueError("cached attention requires one unpadded absolute position and TileLang")
    angles = attn.rotary_emb(position_ids)
    query_latent = attn.q_norm(attn.wq_a(hidden))
    query = _apply_rope(
        attn.wq_b(query_latent).unflatten(-1, (attn.num_heads, attn.head_dim)), angles
    )
    new_kv = quantize_cache(
        _apply_rope(attn.kv_norm(attn.wkv(hidden)), angles), format="fp8", block_size=32
    )
    cache["local_kv"] = _append(cache["local_kv"], new_kv, attn.window_size)
    next_state = state
    latent = None
    compressed_angles = None
    if attn.compress_ratio:
        ratio = attn.compress_ratio
        if attn.compressor is not None:
            partial = _append(cache["partial"], hidden)
            if partial.shape[1] == ratio:
                latent = attn.compressor(partial)
                group_start = position - ratio + 1
                group_ids = position_ids.new_full((batch, 1), group_start)
                compressed_angles = attn.rotary_emb(group_ids)
                new_compressed = quantize_cache(
                    _apply_rope(latent, compressed_angles), format="nvfp4", block_size=16
                )
                cache["compressed_kv"] = _append(cache["compressed_kv"], new_compressed)
                cache["partial"] = None
            else:
                cache["partial"] = partial
            width = cache["compressed_kv"].shape[1]
            next_state = DeepseekV41AttentionState(
                compressed_kv=cache["compressed_kv"],
                compressed_valid=torch.ones(batch, width, device=hidden.device, dtype=torch.bool),
                compression_ratio=ratio,
            )
        elif next_state.compressed_kv is None or next_state.compression_ratio != ratio:
            raise ValueError("cached CSA2 consumer has no matching source")
        if attn.indexer is not None:
            next_state = _index_step(
                attn.indexer,
                hidden,
                query_latent,
                latent,
                angles,
                compressed_angles,
                next_state,
                cache["index_keys"],
                position,
                replay_canvas,
            )
            if attn.indexer.owns_keys:
                cache["index_keys"] = next_state.index_keys
        if next_state.topk_indices is None or next_state.compressed_kv is None:
            raise ValueError("cached CSA2 selection is missing")
    local_length = cache["local_kv"].shape[1]
    slots = torch.arange(local_length, device=hidden.device, dtype=torch.long).view(1, 1, -1)
    slots = slots.expand(batch, 1, -1)
    slots = F.pad(slots, (0, min(replay_canvas, attn.window_size) - local_length), value=-1)
    kv = cache["local_kv"]
    if attn.compress_ratio:
        selected = next_state.topk_indices
        slots = torch.cat((slots, torch.where(selected >= 0, selected + local_length, -1)), dim=-1)
        kv = torch.cat((kv, next_state.compressed_kv), dim=1)
    slots = F.pad(slots, (0, -slots.shape[-1] % 64), value=-1)
    attended = dsv4_sparse_attention(
        query,
        kv,
        attn.sinks_param(query),
        slots,
        attn.head_dim**-0.5,
        backend="tilelang",
        reference_rounding=True,
    )
    valid = torch.ones(batch, 1, device=hidden.device, dtype=torch.bool)
    return DeepseekV41AttentionOutput(attn._project_output(attended, angles, valid), next_state)


class V41PolicyCache:
    """One rollout's cached forward path; caller owns eval/no-grad/FSDP residency."""

    def __init__(self, model):
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise TypeError("expected the resident V4.1 causal LM actor")
        self.model = model
        self.layers = []
        for wrapped in model.model.layers.values():
            inner = wrapped
            while hasattr(inner, "_checkpoint_wrapped_module"):
                inner = inner._checkpoint_wrapped_module
            self.layers.append((wrapped, inner, inner.attn))
        self.caches = [{} for _ in self.layers]
        self.position = None
        self.tokens = None
        self.valid = True

    def _check_mode(self):
        if self.model.training or torch.is_grad_enabled():
            raise ValueError("resident KV cache requires eval mode and no_grad")

    def prefill(self, input_ids, attention_mask=None):
        from nemo_automodel.components.models.deepseek_v41.attention import _apply_rope
        from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache

        self._check_mode()
        if (
            not self.valid
            or self.position is not None
            or input_ids.ndim != 2
            or min(input_ids.shape) < 1
        ):
            raise ValueError("cache prefill requires one fresh uniform-length prompt batch")
        prefix_length = input_ids.shape[1]
        if attention_mask is not None:
            if (
                attention_mask.shape != input_ids.shape
                or attention_mask.dtype != torch.bool
                or bool((attention_mask[:, 1:] & ~attention_mask[:, :-1]).any())
            ):
                raise ValueError("cache prefill requires a boolean right-padding mask")
            lengths = attention_mask.sum(-1)
            if not bool((lengths == lengths[0]).all()) or int(lengths[0]) < 1:
                raise ValueError("cache prefill requires equal positive local prompt lengths")
            prefix_length = int(lengths[0])
        hooks = []
        try:
            for index, (_, inner, attn) in enumerate(self.layers):
                cache = self.caches[index]

                def capture_attention(module, args, kwargs, output, *, target=cache):
                    hidden = args[0]
                    angles = module.rotary_emb(kwargs["position_ids"])
                    kv = quantize_cache(
                        _apply_rope(module.kv_norm(module.wkv(hidden)), angles),
                        format="fp8",
                        block_size=32,
                    )
                    target["local_kv"] = (
                        kv[:, max(0, prefix_length - module.window_size) : prefix_length]
                        .contiguous()
                        .clone()
                    )
                    target["partial"] = (
                        hidden[
                            :,
                            prefix_length
                            // module.compress_ratio
                            * module.compress_ratio : prefix_length,
                        ]
                        .contiguous()
                        .clone()
                        if module.compressor is not None
                        else None
                    )
                    target["compressed_kv"] = (
                        output.state.compressed_kv[:, : prefix_length // module.compress_ratio]
                        .contiguous()
                        .clone()
                        if module.compressor is not None
                        else None
                    )
                    target["index_keys"] = (
                        output.state.index_keys[:, : prefix_length // module.compress_ratio]
                        .contiguous()
                        .clone()
                        if module.indexer is not None and module.indexer.owns_keys
                        else None
                    )

                hooks.append(attn.register_forward_hook(capture_attention, with_kwargs=True))
                adapter = getattr(inner.attn_hc, "simplicial_adapter", None)
                if adapter is not None:

                    def capture_adapter(module, args, _output, *, target=cache):
                        if not hasattr(module, "k1"):
                            target["adapter"] = {
                                "streams": args[0][
                                    :, max(0, prefix_length - module.config.long_window) : prefix_length
                                ].contiguous().clone()
                            }
                            return
                        _, _, projected = _adapter_projections(module, args[0])
                        target["adapter"] = {
                            name: value[
                                :,
                                max(
                                    0,
                                    prefix_length
                                    - (
                                        module.config.short_window
                                        if name.endswith("1")
                                        else module.config.long_window
                                    ),
                                ) : prefix_length,
                            ]
                            .contiguous()
                            .clone()
                            for name, value in projected.items()
                        }

                    hooks.append(adapter.register_forward_hook(capture_adapter))
            output = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_hidden_states=True,
                **getattr(self.model, "_archlab_rl_hidden_forward_kwargs", {}),
            )
            if any(
                "local_kv" not in item
                or "adapter" not in item
                and getattr(inner.attn_hc, "simplicial_adapter", None) is not None
                for item, (_, inner, _) in zip(self.caches, self.layers, strict=True)
            ):
                raise RuntimeError("a V4.1 prefill hook did not execute")
            self.tokens = input_ids[:, :prefix_length].clone()
            self.position = prefix_length
            return output.hidden_states[:, prefix_length - 1].clone()
        except BaseException:
            self.valid = False
            raise
        finally:
            for hook in hooks:
                hook.remove()

    def decode(self, token_ids, *, replay_canvas):
        from contextlib import ExitStack

        from nemo_automodel.components.models.deepseek_v41.attention import (
            DeepseekV41AttentionState,
        )
        from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41HyperConnection

        from archlab.automodel.deepseek_v41_rl_cache_shapes import (
            padded_collapse,
            replay_shape_boundaries,
        )

        self._check_mode()
        if (
            not self.valid
            or self.position is None
            or token_ids.shape != (self.tokens.shape[0], 1)
            or token_ids.dtype != torch.long
            or token_ids.device != self.tokens.device
        ):
            raise ValueError("cache decode requires one matching token per row after prefill")
        if type(replay_canvas) is not int or replay_canvas < 1:
            raise ValueError("cache decode requires its explicit full-prefix replay canvas")
        # Finished local ranks still participate in collectives and consume dummy
        # tokens. Their unused states may extend beyond the active ranks' canvas.
        replay_canvas = max(replay_canvas, self.position + 1)
        originals = []
        scope = ExitStack()
        try:
            scope.enter_context(replay_shape_boundaries(
                self.model, self.layers, position=self.position, canvas=replay_canvas
            ))
            tokens = torch.cat((self.tokens, token_ids), dim=1)
            text = self.model.model
            hashes = text.engram_hash(tokens) if text.engram_hash is not None else None
            embedded = text.embed_tokens(token_ids)
            streams = embedded.unsqueeze(2).expand(-1, -1, text.config.hc_mult, -1)
            pre_mix = torch.zeros(
                *token_ids.shape, text.config.hc_mult, device=embedded.device, dtype=torch.float32
            )
            pre_mix[..., 0] = 1
            state = DeepseekV41AttentionState()
            positions = token_ids.new_full(token_ids.shape, self.position)
            for index, (wrapped, inner, attn) in enumerate(self.layers):
                cache = self.caches[index]
                old = attn.__dict__.get("forward")
                originals.append((attn, old))

                def attention_forward(
                    module, hidden, *, position_ids, state, attention_mask=None, target=cache
                ):
                    return _attention_step(
                        module,
                        target,
                        hidden,
                        position_ids=position_ids,
                        state=state,
                        attention_mask=attention_mask,
                        position=self.position,
                        replay_canvas=replay_canvas,
                    )

                attn.forward = MethodType(attention_forward, attn)
                adapter = getattr(inner.attn_hc, "simplicial_adapter", None)
                if adapter is not None:
                    old = adapter.__dict__.get("forward")
                    originals.append((adapter, old))

                    def adapter_forward(module, streams, *, target=cache["adapter"]):
                        return _adapter_step(
                            module, target, streams, position=self.position,
                            replay_canvas=replay_canvas,
                        )

                    adapter.forward = MethodType(adapter_forward, adapter)
                engram = (
                    None if inner.engram is None else hashes[:, -1:, inner.engram.layer_hash_index]
                )
                streams, pre_mix, state = wrapped(
                    streams,
                    pre_mix,
                    state,
                    position_ids=positions,
                    attention_mask=None,
                    image_mask=None,
                    engram_hash_ids=engram,
                )
            hidden = padded_collapse(
                DeepseekV41HyperConnection.collapse, streams, pre_mix,
                position=self.position, canvas=replay_canvas,
            )
            hidden = text.norm(hidden)[:, 0]
            self.tokens = tokens
            self.position += 1
            return hidden
        except BaseException:
            self.valid = False
            raise
        finally:
            for module, original in reversed(originals):
                if original is None:
                    module.__dict__.pop("forward", None)
                else:
                    module.forward = original
            scope.close()


def qualify_resident_cache(
    model,
    prompts,
    *,
    pad_token_id,
    context_limit,
    steps=16,
    tolerance=0.02,
    retain_weights=True,
    reserve_gib=16,
):
    """Compare cached hidden states and the full vocabulary against padded replay.

    This teacher-forced probe crosses compression boundaries without sampling or
    changing parameters. The separate rollout/replay gate must also pass. All
    ranks execute the same number of forwards and agree on the returned verdict.
    """
    from contextlib import ExitStack

    import torch.distributed as dist

    from archlab.automodel.deepseek_v41_live_window import inference_head
    from archlab.rl.weight_residency import retained_fsdp_weights

    if not prompts or len({len(row) for row in prompts}) != 1 or not prompts[0]:
        raise ValueError("cache qualification requires uniform nonempty local prompts")
    if not 0 < steps or len(prompts[0]) + steps > context_limit or not 0 < tolerance <= 0.02:
        raise ValueError("invalid cache qualification budget or replay tolerance")
    modes = [(module, module.training) for module in model.modules()]
    device = model.lm_head.weight.device
    stats = torch.zeros(4, device=device, dtype=torch.float32)
    comparisons = []
    residency = None
    try:
        model.eval()
        with ExitStack() as scope:
            scope.enter_context(torch.no_grad())
            if retain_weights:
                residency = scope.enter_context(
                    retained_fsdp_weights(model, minimum_free_gib=reserve_gib)
                )
            head_context = residency.inference_head if residency else inference_head
            original = torch.tensor(prompts, device=device, dtype=torch.long)
            ids = original.clone()
            cache = V41PolicyCache(model)
            prefill_canvas = context_limit
            prefill_ids = F.pad(ids, (0, prefill_canvas - ids.shape[1]), value=pad_token_id)
            prefill_mask = (
                torch.arange(prefill_canvas, device=device).expand_as(prefill_ids) < ids.shape[1]
            )
            cached = cache.prefill(prefill_ids, attention_mask=prefill_mask)
            for index in range(steps + 1):
                if index:
                    token = original[:, (index - 1) % original.shape[1] :][:, :1]
                    ids = torch.cat((ids, token), dim=1)
                canvas = context_limit
                if index:
                    cached = cache.decode(token, replay_canvas=canvas)
                inputs = F.pad(ids, (0, canvas - ids.shape[1]), value=pad_token_id)
                mask = torch.arange(canvas, device=device).expand_as(inputs) < ids.shape[1]
                all_hidden = model(
                    input_ids=inputs, attention_mask=mask, return_hidden_states=True,
                    **getattr(model, "_archlab_rl_hidden_forward_kwargs", {}),
                ).hidden_states
                reference = all_hidden[:, ids.shape[1] - 1].clone()
                del all_hidden
                finite = cached.isfinite().all() & reference.isfinite().all()
                difference = cached.float() - reference.float()
                absolute = difference.abs().max()
                relative_rms = (
                    difference.square().mean().sqrt()
                    / reference.float().square().mean().sqrt().clamp_min(1e-12)
                )
                with (
                    head_context(model.lm_head) as head,
                    torch.autocast(device.type, enabled=False),
                ):
                    observed = F.linear(cached.float(), head.weight).log_softmax(-1)
                    expected = F.linear(reference.float(), head.weight).log_softmax(-1)
                    finite = finite & observed.isfinite().all() & expected.isfinite().all()
                    probability_error = (observed - expected).abs().max()
                values = torch.stack((absolute, relative_rms, probability_error, (~finite).float()))
                stats = torch.maximum(
                    stats, torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                )
                local_values = values.tolist()
                comparisons.append(
                    {
                        "decode_step": index,
                        "prefix_tokens": ids.shape[1],
                        "replay_canvas": canvas,
                        "finite": not bool(local_values[3]),
                        "hidden_max_abs_error": local_values[0] if not local_values[3] else None,
                        "hidden_relative_rms_error": local_values[1]
                        if not local_values[3]
                        else None,
                        "logprob_max_abs_error": local_values[2] if not local_values[3] else None,
                    }
                )
                del observed, expected, reference, difference
            del cache, cached
        if dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        maximum, relative, logprob, nonfinite = stats.tolist()
        return {
            "passed": not nonfinite and relative <= 1e-3 and logprob <= tolerance,
            "kind": "resident-cache-padded-prefix-equivalence-v1",
            "replay_canvas_policy": "fixed-context-v1",
            "decode_steps": steps,
            "prefix_comparisons": steps + 1,
            "prompt_tokens_local": len(prompts[0]),
            "all_vocabulary_logprobs": True,
            "finite": not bool(nonfinite),
            "hidden_max_abs_error": maximum,
            "hidden_relative_rms_error": relative,
            "hidden_relative_rms_tolerance": 1e-3,
            "logprob_max_abs_error": logprob,
            "logprob_tolerance": tolerance,
            "local_comparisons": comparisons,
            "weight_residency": None if residency is None else residency.receipt,
            "optimizer_updates": 0,
        }
    finally:
        for module, training in modes:
            module.training = training
