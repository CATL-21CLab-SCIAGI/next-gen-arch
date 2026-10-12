"""Receipt-only cache and reducer instrumentation for opt-in GPU admission."""
from __future__ import annotations

import torch

from archlab.training.serving import QualificationWorker


class ConversionQualificationWorker(QualificationWorker):
    def archlab_start_cache_audit(self):
        import archlab.training.serving_conversion as bridge

        self._raw_checks, self._histories, self._slots = [], {}, {}
        original, decode = bridge._paged_joint_attention, bridge._decode
        current = {}

        def record(kind, native, actual, expected, length):
            self._raw_checks.append(dict(
                kind=kind, layer=native.layer_idx, length=length,
                relative_l2=float((actual - expected).norm() / expected.norm().clamp_min(1e-12)),
                max_abs=float((actual - expected).abs().max()),
            ))

        def checked_decode(q, k, v, short, *, scaling):
            output = decode(q, k, v, short, scaling=scaling)
            native = current["native"]
            history = self._histories[native.layer_idx]
            length = history[0].shape[1]
            window = length if native.is_global else native.window_span
            expected = decode(q, history[0][:, -window:], history[1][:, -window:],
                              tuple(t[:, -16:] for t in history[2:]), scaling=scaling)
            record("paged_decode_vs_contiguous_history", native, output, expected, length)
            return output

        def observed(native, query, key, value, mask, **kwargs):
            from archlab.architectures.tilelang_attention import tilelang_attention, tilelang_decode

            owners = [native.archlab_conversion_long_attention,
                      native.archlab_conversion_short_attention]
            contexts = [bridge._attention_context(owner) for owner in owners]
            if contexts[0][0] is None:
                return original(native, query, key, value, mask, **kwargs)
            q, k, v = [t.transpose(1, 2).contiguous() for t in (query, key, value)]
            short = native._archlab_conversion_short
            length = int(contexts[0][0].seq_lens[0].item())
            fresh = length == q.shape[1]
            pieces = (k, v, *short)
            self._histories[native.layer_idx] = (tuple(t.clone() for t in pieces) if fresh else
                tuple(torch.cat((old, new), 1) for old, new in
                      zip(self._histories[native.layer_idx], pieces, strict=True)))
            current["native"] = native
            output = original(native, query, key, value, mask, **kwargs)
            for owner, context, tensors in zip(owners, contexts, ((k, v), short), strict=True):
                self._slots[owner.layer_name] = (
                    owner, int(context[3][-1].item()),
                    tuple(t[:, -1].squeeze(0).clone() for t in tensors))
            if fresh:
                window = length if native.is_global else native.window_span
                prefill = tilelang_attention(q, k, v, scaling=native.scaling, short=short,
                    short_window=16, long_window=window, dynamic_length=True)[:, -1:]
                decoded = tilelang_decode(q[:, -1:], k[:, -window:], v[:, -window:],
                    scaling=native.scaling, short=tuple(t[:, -16:] for t in short))
                record("prefill_last_query_vs_decode", native, decoded, prefill, length)
            return output

        bridge._paged_joint_attention, bridge._decode = observed, checked_decode
        return self.archlab_cache_audit()

    def archlab_cache_audit(self):
        model = self.model_runner.model
        owners = []
        for index, layer in enumerate(model.policy.model.layers):
            native = layer.self_attn.native
            for axis in ("long", "short"):
                owner = getattr(native, f"archlab_conversion_{axis}_attention")
                cache = owner.kv_cache
                window = (None if native.is_global else native.window_span) if axis == "long" else 16
                assert owner.sliding_window == window
                assert cache.ndim == 5 and cache.shape[0] == 2
                assert cache.shape[3:] == (native.num_kv_heads, native.head_dim)
                assert cache.dtype == torch.bfloat16
                assert owner.get_attn_backend().get_name() == "FLASH_ATTN"
                assert owner.impl.vllm_flash_attn_version == 2
                owners.append(dict(layer=index, axis=axis, name=owner.layer_name,
                                   window=window, shape=list(cache.shape), stride=list(cache.stride())))
        assert len(owners) == 2 * len(model.policy.model.layers)
        assert len({owner["name"] for owner in owners}) == len(owners)
        live, ranges = [], []
        for name, (owner, slot, expected) in getattr(self, "_slots", {}).items():
            block, offset = divmod(slot, owner.kv_cache.shape[2])
            actual = tuple(owner.kv_cache[axis, block, offset] for axis in (0, 1))
            live.append(dict(name=name, slot=slot, exact_values_preserved=all(
                torch.equal(left, right) for left, right in zip(actual, expected, strict=True))))
            for value in actual:
                for row in value:
                    assert row.stride() == (1,)
                    ranges.append((row.data_ptr(), row.data_ptr() + row.numel() * row.element_size()))
        ranges.sort()
        return dict(owners=owners, raw_checks=list(getattr(self, "_raw_checks", [])), live_slots=live,
                    live_slot_byte_ranges_disjoint=all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:], strict=False)))
