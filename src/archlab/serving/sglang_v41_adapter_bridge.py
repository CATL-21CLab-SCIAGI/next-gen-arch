"""Checked post-attention insertion for an eager, TP8 SGLang V4.1 worker.

No container source is modified. The wrapper replaces methods on selected
decoder instances only. Runtime qualification must establish that the reviewed
boundary executes exactly once per adapted layer; bypassing it fails closed.
"""

from __future__ import annotations

from types import MethodType

import torch

from archlab.architectures.deepseek_v41_incremental import IncrementalV41Adapter
from archlab.serving.sglang_v41_padding import real_token_count


class RequestAdapterCaches:
    def __init__(self, adapter, *, max_slots=32):
        self.adapter, self.max_slots = adapter, max_slots
        self.caches = {}

    def apply(self, streams, *, slots, lengths, positions):
        if len(slots) != len(lengths) or len(set(slots)) != len(slots):
            raise ValueError("request slots must be distinct and match segment lengths")
        if any(type(slot) is not int or not 0 <= slot < self.max_slots for slot in slots):
            raise ValueError("request slot exceeds the qualified cache bound")
        if any(type(length) is not int or length <= 0 for length in lengths):
            raise ValueError("empty/padded request segments are unsupported")
        if sum(lengths) != streams.shape[0] or len(positions) != streams.shape[0]:
            raise ValueError("positions and request lengths do not cover residual streams")
        segments, offset = [], 0
        # Validate the whole batch before advancing any request's state.
        for slot, length in zip(slots, lengths, strict=True):
            start = positions[offset]
            if positions[offset:offset + length] != list(range(start, start + length)):
                raise ValueError("positions must be contiguous within each request")
            existing = self.caches.get(slot)
            if start and (existing is None or existing.position != start):
                raise ValueError("missing adapter prefix: prefix sharing/replay is unsupported")
            segments.append((slot, offset, length, start))
            offset += length
        outputs = []
        for slot, offset, length, start in segments:
            if start == 0:
                self.caches[slot] = IncrementalV41Adapter(self.adapter)
            outputs.append(self.caches[slot](streams[offset:offset + length].unsqueeze(0),
                                            start_position=start).squeeze(0))
        return torch.cat(outputs, dim=0)


def install_adapter_boundary(layer, adapter, *, tp_size, max_slots=32):
    if tp_size != 8:
        raise ValueError("the reviewed eager insertion requires TP8 (TP4 has fused bypasses)")
    if not getattr(layer, "hc_pre_from_prev_sublayer", False):
        raise ValueError("V4.1 carried pre-mix semantics are required")
    if hasattr(layer, "archlab_adapter"):
        raise ValueError("adapter boundary is already installed")
    original_forward = layer.forward_hc_pre_from_prev
    combined_boundary = hasattr(layer, "_hc_post_with_combine")
    original_post = layer._hc_post_with_combine if combined_boundary else layer.hc_post
    state = RequestAdapterCaches(adapter, max_slots=max_slots)
    context = {}
    layer.add_module("archlab_adapter", adapter)

    def apply_live(updated):
        count = context["count"]
        if count == 0:
            return updated
        live = state.apply(updated[:count], slots=context["slots"], lengths=context["lengths"],
                           positions=context["positions"])
        return torch.cat((live, updated[count:]), dim=0) if count < updated.shape[0] else live

    def post(self, x, residual, post, comb, pre, forward_batch, norm=None):
        updated, combined, normalized = original_post(x, residual, post, comb, pre,
                                                      forward_batch, norm=norm)
        if updated.shape[0] == 0 or not context:
            return updated, combined, normalized
        if norm is self.post_attention_layernorm:
            if not context or context["batch"] is not forward_batch:
                raise RuntimeError("adapter insertion has no matching request context")
            context["calls"] += 1
            if context["calls"] != 1:
                raise RuntimeError("adapter attention boundary executed more than once")
            updated = apply_live(updated)
            # Any fused FFN input computed before adaptation is now stale.
            combined, normalized = None, None
        return updated, combined, normalized

    def direct_post(self, x, residual, post, comb):
        updated = original_post(x, residual, post, comb)
        if updated.shape[0] == 0 or not context:
            return updated
        if not context:
            raise RuntimeError("adapter HC expansion has no matching request context")
        context["posts"] += 1
        if context["posts"] == 1:
            context["calls"] += 1
            return apply_live(updated)
        if context["posts"] != 2:
            raise RuntimeError("unexpected extra hyper-connection expansion")
        return updated

    def forward(self, positions, hidden_states, input_ids, forward_batch,
                input_ids_global, prev_pre, **kwargs):
        if context:
            raise RuntimeError("reentrant layer execution is unsupported")
        mode = forward_batch.forward_mode
        count = real_token_count(forward_batch, hidden_states.shape[0])
        slots = forward_batch.req_pool_indices.detach().cpu().tolist()
        if count == 0:
            return original_forward(positions, hidden_states, input_ids, forward_batch,
                                    input_ids_global, prev_pre, **kwargs)
        if mode.is_decode():
            slots = slots[:count]
            lengths = [1] * len(slots)
        elif mode.is_extend_without_speculative():
            lengths = list(forward_batch.extend_seq_lens_cpu)
        else:
            raise ValueError("only non-speculative eager prefill and decode are qualified")
        if sum(lengths) != count:
            raise ValueError("adapter request lengths do not cover real tokens")
        context.update(batch=forward_batch, slots=slots, lengths=lengths, count=count,
                       positions=positions[:count].detach().cpu().tolist(), calls=0, posts=0)
        try:
            result = original_forward(positions, hidden_states, input_ids, forward_batch,
                                      input_ids_global, prev_pre, **kwargs)
            if context["calls"] != 1 or (not combined_boundary and context["posts"] != 2):
                raise RuntimeError("runtime bypassed the required attention adapter boundary")
            return result
        finally:
            context.clear()

    if combined_boundary:
        layer._hc_post_with_combine = MethodType(post, layer)
    else:
        layer.hc_post = MethodType(direct_post, layer)
    layer.forward_hc_pre_from_prev = MethodType(forward, layer)
    return state
