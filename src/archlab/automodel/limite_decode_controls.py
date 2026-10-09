"""Independent controls for fixed-cache execution and attention arithmetic.

Graph capture and cache migration must reproduce the same fixed-cache eager
implementation exactly. Cross-backend BF16 differences belong to a separate
behavior-policy ratio check; they are not evidence of a cache indexing error.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


def _pair_equal(actual, expected):
    return len(actual) == len(expected) and all(
        a.shape == b.shape and a.dtype == b.dtype and a.device == b.device and torch.equal(a, b)
        for a, b in zip(actual, expected, strict=True)
    )


def cache_equal(actual, expected):
    """Compare every logical fixed-cache history, recursively including adapters."""
    if (actual.global_layers != expected.global_layers
            or len(actual.buffers) != len(expected.buffers)
            or actual.get_seq_length() != expected.get_seq_length()
            or not torch.equal(actual.position, expected.position)):
        return False
    for actual_pair, expected_pair in zip(actual.buffers, expected.buffers, strict=True):
        if not _pair_equal(actual_pair, expected_pair):
            return False
    if actual.archlab_short.keys() != expected.archlab_short.keys():
        return False
    for index, actual_pair in actual.archlab_short.items():
        if not _pair_equal(actual_pair, expected.archlab_short[index]):
            return False
    has_actual = hasattr(actual, "archlab_native_preludes")
    has_expected = hasattr(expected, "archlab_native_preludes")
    return (has_actual == has_expected and (not has_actual or cache_equal(
        actual.archlab_native_preludes, expected.archlab_native_preludes,
    )))


def cache_selection_equal(source, selected, rows, *, length):
    """Verify migration against the unmigrated tensors, without cache_rows."""
    if (source.global_layers != selected.global_layers
            or len(source.buffers) != len(selected.buffers)
            or selected.get_seq_length() != length
            or not bool((selected.position == length).all())
            or source.archlab_short.keys() != selected.archlab_short.keys()):
        return False
    for output_row, source_row in enumerate(rows.tolist()):
        for index, (origin_pair, target_pair) in enumerate(zip(source.buffers, selected.buffers, strict=True)):
            extent = min(length, origin_pair[0].shape[2])
            global_layer = index in source.global_layers
            window = slice(0, extent) if global_layer else slice(-extent, None)
            for origin, target in zip(origin_pair, target_pair, strict=True):
                if not _pair_equal([target[output_row, :, window]], [origin[source_row, :, window]]):
                    return False
                if global_layer and bool(target[output_row, :, extent:].count_nonzero()):
                    return False
                if not global_layer and bool(target[output_row, :, :-extent].count_nonzero()):
                    return False
        for index, origin_pair in source.archlab_short.items():
            for origin, target in zip(origin_pair, selected.archlab_short[index], strict=True):
                extent = min(length, origin.shape[1])
                if not _pair_equal([target[output_row, -extent:]], [origin[source_row, -extent:]]):
                    return False
                if bool(target[output_row, :-extent].count_nonzero()):
                    return False
    has_source = hasattr(source, "archlab_native_preludes")
    has_selected = hasattr(selected, "archlab_native_preludes")
    return (has_source == has_selected and (not has_source or cache_selection_equal(
        source.archlab_native_preludes, selected.archlab_native_preludes, rows, length=length,
    )))


def fixed_cache_decode_controls(model, ids, capacity, *, steps=8,
                                native_gqa_backend="flash_attn_kvcache",
                                teacher_forcing="sampled", seed=1234):
    """Prove capture and compact-cache migration with identical eager arithmetic.

    Uses one immutable model and checkpoint. Every CUDA graph remains owned by
    its pool throughout the oracle, as in production. Sampling uses a private
    generator and neither a learner optimizer nor global RNG is advanced.
    """
    from archlab.architectures.limite_decode import GraphDecoder, GraphDecoderPool
    from archlab.architectures.limite_decode_state import cache_rows
    from archlab.architectures.limite_gqa import set_native_decode_gqa

    if (ids.ndim != 2 or ids.shape[0] < 2 or ids.shape[1] < 1 or steps < 8
            or ids.shape[1] + steps > capacity):
        raise ValueError("fixed-cache control requires multiple rows and eight valid decode positions")
    if teacher_forcing not in ("sampled", "greedy"):
        raise ValueError("fixed-cache teacher forcing must be sampled or greedy")
    if ids.device.type != "cuda":
        raise ValueError("captured decode admission requires CUDA")
    was_training = model.training
    records, retained_eager = [], []
    generator = torch.Generator(device=ids.device).manual_seed(seed)
    try:
        model.eval()
        set_native_decode_gqa(model, backend=native_gqa_backend)
        with torch.no_grad():
            output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            source = output.past_key_values
            tokens = output.logits[:, -1].argmax(-1, keepdim=True)
            del output
            pool = GraphDecoderPool(model)
            pool.synchronize()
            graph = pool.get(source, tokens, capacity)
            eager = GraphDecoder(model, source, tokens, graph.cache.capacity, capture=False)
            retained_eager.append(eager)
            rows = torch.arange(ids.shape[0], device=ids.device)
            prefix = ids.shape[1]
            for step in range(steps):
                migrated = None
                migration_cache_exact = True
                if step in (2, 4, 6) and rows.numel() > 1:
                    kept = torch.arange(1, len(rows), device=ids.device)
                    previous_graph_cache = graph.cache
                    graph_source = cache_rows(graph.cache, kept, length=prefix + step)
                    eager_source = cache_rows(eager.cache, kept, length=prefix + step)
                    rows = rows[1:]
                    pool.synchronize()
                    graph = pool.get(graph_source, tokens[rows], capacity)
                    eager = GraphDecoder(model, eager_source, tokens[rows], graph.cache.capacity, capture=False)
                    # A fresh object rebuilt from the graph history is an
                    # independent control for pool reset/row migration.
                    migrated = GraphDecoder(model, graph_source, tokens[rows], graph.cache.capacity, capture=False)
                    retained_eager.extend((eager, migrated))
                    migration_cache_exact = (cache_equal(graph.cache, migrated.cache)
                                             and cache_selection_equal(previous_graph_cache, graph.cache,
                                                                       kept, length=prefix + step))
                actual = graph(tokens[rows], prefix + step).clone()
                expected = eager(tokens[rows], prefix + step).clone()
                migration_logits_exact = True
                if migrated is not None:
                    fresh = migrated(tokens[rows], prefix + step).clone()
                    migration_logits_exact = torch.equal(actual, fresh)
                    migration_cache_exact &= cache_equal(graph.cache, migrated.cache)
                finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
                records.append(dict(
                    step=step, active_rows=rows.tolist(), finite_logits=finite,
                    graph_eager_logits_exact=torch.equal(actual, expected),
                    graph_eager_max_abs_logit_error=float((actual.float() - expected.float()).abs().max()),
                    graph_eager_cache_exact=cache_equal(graph.cache, eager.cache),
                    migration_tested=migrated is not None,
                    migration_cache_exact=migration_cache_exact,
                    migration_logits_exact=migration_logits_exact,
                ))
                selected = (torch.multinomial(expected.float().softmax(-1), 1, generator=generator)
                            if teacher_forcing == "sampled" else expected.argmax(-1, keepdim=True))
                tokens = tokens.clone()
                tokens[rows] = selected
            torch.cuda.synchronize(ids.device)
        passed = all(row["finite_logits"] and row["graph_eager_logits_exact"]
                     and row["graph_eager_cache_exact"] and row["migration_cache_exact"]
                     and row["migration_logits_exact"] for row in records)
        return dict(
            passed=passed, backend=native_gqa_backend, steps=steps,
            batch_size=ids.shape[0], prompt_length=ids.shape[1], capacity=capacity,
            teacher_forcing=teacher_forcing, seed=seed, comparisons=records,
            reference="identical fixed-cache backend, weights, active batch and forced token history",
            tolerance="bit-exact logits and all logical backbone/prelude/short KV histories",
            no_optimizer_step=True,
        )
    finally:
        torch.cuda.synchronize(ids.device)
        model.train(was_training)


def decode_reference(q, k, v, *, scaling, short=None):
    """Explicit valid-key softmax oracle; no fixed storage or attention backend."""
    groups = q.shape[2] // k.shape[2]
    q, k, v = [value.float() for value in (q, k, v)]
    k, v = [value.repeat_interleave(groups, dim=2) for value in (k, v)]
    if short is None:
        scores = torch.einsum("bhd,bkhd->bhk", q[:, 0], k) * scaling
        return torch.einsum("bhk,bkhd->bhd", scores.softmax(-1), v)[:, None]
    k1, v1 = [value.float().repeat_interleave(groups, dim=2) for value in short]
    scores = torch.einsum("bhd,bjhd,bkhd->bhjk", q[:, 0], k1, k) * scaling
    probabilities = scores.flatten(-2).softmax(-1).reshape_as(scores)
    return torch.einsum("bhjk,bjhd,bkhd->bhd", probabilities, v1, v)[:, None]


def attention_reduction_controls(model, *, native_gqa_backend="flash_attn_kvcache", seed=83):
    """Check the actual decode primitives against independent FP32 softmax math.

    Exercises full and sliding attention, three valid keys and a full local
    window. Invalid prefix/tail slots contain large sentinels, so an indexing
    or masking error cannot pass by accidentally reading zeros. The relative
    tolerance0.005 is the existing TileLang forward numerical contract.
    """
    from archlab.architectures.limite_gqa import _DecodeInterface
    from archlab.architectures.tilelang_attention import tilelang_decode

    config = model.config
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("decode reduction admission requires CUDA")
    generator = torch.Generator(device=device).manual_seed(seed)
    geometry = dict(batch=3, query_heads=config.num_attention_heads,
                    kv_heads=config.num_key_value_heads, head_dim=config.head_dim)
    scale = float(config.attention_softmax_scale)
    variant = model.model.adapter_config["variant"]
    records = []
    interface = _DecodeInterface(SimpleNamespace(get_interface=lambda *_: None)).get_interface("sdpa", None)
    with torch.no_grad():
        for is_global, active in ((True, 3), (True, 1027), (False, 3), (False, int(config.sliding_window))):
            capacity = config.max_position_embeddings if is_global else int(config.sliding_window)
            start = 0 if is_global else capacity - active
            q = torch.randn(3, 1, geometry["query_heads"], geometry["head_dim"],
                            device=device, dtype=torch.bfloat16, generator=generator)
            key, value = [torch.full((3, capacity, geometry["kv_heads"], geometry["head_dim"]),
                                     100., device=device, dtype=torch.bfloat16) for _ in range(2)]
            for tensor in (key, value):
                tensor[:, start:start + active].copy_(torch.randn(
                    tensor[:, start:start + active].shape, device=device, dtype=tensor.dtype, generator=generator))
            short_count = min(active, 16)
            short = None
            if variant == "simplicial":
                short = tuple(torch.full((3, 16, geometry["kv_heads"], geometry["head_dim"]),
                                         100., device=device, dtype=torch.bfloat16) for _ in range(2))
                for tensor in short:
                    tensor[:, -short_count:].copy_(torch.randn(
                        tensor[:, -short_count:].shape, device=device, dtype=tensor.dtype, generator=generator))
            indices = torch.arange(capacity, device=device)
            mask = ((indices >= start) & (indices < start + active))[None, None, None]
            module = SimpleNamespace(training=False, _archlab_decode_gqa=True,
                                     _archlab_decode_backend=native_gqa_backend, is_global=is_global)
            actual_backbone, _ = interface(module, q.transpose(1, 2), key.transpose(1, 2),
                                           value.transpose(1, 2), mask, scaling=scale)
            actual_prelude = tilelang_decode(
                q, key, value, scaling=scale, short=short,
                lengths=torch.tensor([active, short_count, start], device=device, dtype=torch.int32),
            )
            valid_key, valid_value = key[:, start:start + active], value[:, start:start + active]
            expected_backbone = decode_reference(q, valid_key, valid_value, scaling=scale)
            expected_prelude = decode_reference(
                q, valid_key, valid_value, scaling=scale,
                short=tuple(tensor[:, -short_count:] for tensor in short) if short else None,
            )
            reductions = {}
            for label, actual, expected in (
                ("backbone", actual_backbone, expected_backbone),
                ("prelude", actual_prelude, expected_prelude),
            ):
                difference = actual.float() - expected.float()
                relative = float(difference.norm() / expected.float().norm().clamp_min(1e-6))
                finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
                reductions[label] = dict(relative_l2=relative, max_abs_error=float(difference.abs().max()),
                                         finite=finite, passed=finite and relative < .005)
            records.append(dict(is_global=is_global, active_keys=active, capacity=capacity,
                                leftpad=start, reductions=reductions))
        torch.cuda.synchronize(device)
    return dict(passed=all(value["passed"] for row in records for value in row["reductions"].values()),
                variant=variant, backend=native_gqa_backend, geometry=geometry, seed=seed,
                tolerance=dict(relative_l2=.005), comparisons=records,
                reference="explicit FP32 normal or joint simplicial softmax over valid keys",
                invalid_slots="100-valued sentinel keys and values; excluded explicitly by oracle")
