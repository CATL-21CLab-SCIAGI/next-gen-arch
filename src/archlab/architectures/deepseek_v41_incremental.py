"""Request-local cached V4.1 adapter oracle for inference integration.

This preserves trained parameters and the residual insertion equation. It is
not an SGLang model implementation: the engine must supply post-attention
hyper-connection streams and own a separate instance per request and layer.
No prefix sharing, padding, speculative rollback, or training is supported.
"""

from __future__ import annotations

import math

import torch

from archlab.architectures.deepseek_v41_adapter import V41SimplicialAdapter
from archlab.architectures.deepseek_v41_normal_adapter import V41NormalAttentionAdapter


class IncrementalV41Adapter:
    """Bounded K/V state; accept contiguous prefill chunks or single tokens.

    Attention arithmetic is the FP32 equation oracle, not a qualified fused
    GPU kernel. Projections retain the original adapter autocast policy.
    Use ``reset`` between requests; start_position prevents accidental replay.
    """

    def __init__(self, adapter):
        if not isinstance(adapter, (V41SimplicialAdapter, V41NormalAttentionAdapter)):
            raise TypeError("expected a V4.1 adapter")
        self.adapter = adapter
        self.reset()

    def reset(self):
        self.position = 0
        self._cache = {}
        self._signature = None

    @property
    def cache_lengths(self):
        return {name: tensor.shape[1] for name, tensor in self._cache.items()}

    def _append(self, name, tensor, window):
        previous = self._cache.get(name)
        if previous is not None:
            tensor = torch.cat((previous, tensor), dim=1)
        # Clone the retained slice so a long prefill cannot retain old storage.
        tensor = tensor[:, -window:].contiguous().clone()
        self._cache[name] = tensor
        return tensor

    @torch.inference_mode()
    def __call__(self, streams, *, start_position):
        a, c = self.adapter, self.adapter.config
        if a.training:
            raise ValueError("incremental adapter requires eval mode")
        if type(start_position) is not int or start_position != self.position:
            raise ValueError("noncontiguous position; reset cache for a new request")
        if (streams.ndim != 4 or min(streams.shape) < 1
                or streams.shape[-2:] != (c.streams, c.width)):
            raise ValueError("expected unpadded [batch, sequence, streams, width]")
        signature = (streams.shape[0], streams.device, streams.dtype)
        if self._signature is not None and signature != self._signature:
            raise ValueError("batch/device/dtype changed within request")
        if streams.dtype != torch.float32 and not (
            streams.is_cuda and streams.dtype == torch.bfloat16
        ):
            raise ValueError("oracle supports FP32 or CUDA BF16 streams")
        previous = (self.position, self._cache.copy(), self._signature)
        self._signature = signature
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=streams.is_cuda and streams.dtype == torch.bfloat16):
                output = self._chunk(streams)
                self.position += streams.shape[1]
        except Exception:
            # Appends replace tensors, so the old references remain valid.
            self.position, self._cache, self._signature = previous
            raise
        return output

    def _chunk(self, streams):
        a, c = self.adapter, self.adapter.config
        batch, length = streams.shape[:2]
        read = a.read_logits.float().softmax(-1)
        x = a.input_norm((streams.float() * read[None, None, :, None]).sum(-2).to(streams.dtype))
        q = a.q_norm(a.q(x).reshape(batch, length, c.query_heads, c.head_dim))
        shape = (batch, length, c.kv_heads, c.head_dim)
        # Prefill projections and the output linear are batched. Only the
        # bounded-window equation oracle runs per query; decode has length 1.
        if isinstance(a, V41NormalAttentionAdapter):
            projected = {"k": a.k_norm(a.k(x).reshape(shape)), "v": a.v(x).reshape(shape)}
        else:
            projected = {"k1": a.k1_norm(a.k1(x).reshape(shape)),
                         "k2": a.k2_norm(a.k2(x).reshape(shape)),
                         "v1": a.v1(x).reshape(shape), "v2": a.v2(x).reshape(shape)}
        outputs = []
        for index in range(length):
            token = {name: tensor[:, index:index + 1] for name, tensor in projected.items()}
            outputs.append(self._attend(q[:, index], token))
        attended = torch.stack(outputs, dim=1).to(q.dtype).flatten(-2)
        attended = (attended.float() * a.output_gate(x).float().sigmoid()).to(attended.dtype)
        branch = a.output(attended)
        write = 2 * a.write_logits.float().sigmoid()
        return (streams.float() + branch.float().unsqueeze(-2) * write[None, None, :, None]).to(streams.dtype)

    def _attend(self, q, token):
        a, c = self.adapter, self.adapter.config
        repeats = c.query_heads // c.kv_heads
        if isinstance(a, V41NormalAttentionAdapter):
            k = self._append("k", token["k"], c.long_window)
            v = self._append("v", token["v"], c.long_window)
            with torch.autocast(q.device.type, enabled=False):
                k, v = (t.float().repeat_interleave(repeats, dim=2) for t in (k, v))
                scores = torch.einsum("bhd,bkhd->bhk", q.float(), k) / math.sqrt(c.head_dim)
                attended = torch.einsum("bhk,bkhd->bhd", scores.softmax(-1), v)
        else:
            k1 = self._append("k1", token["k1"], c.short_window)
            k2 = self._append("k2", token["k2"], c.long_window)
            v1 = self._append("v1", token["v1"], c.short_window)
            v2 = self._append("v2", token["v2"], c.long_window)
            with torch.autocast(q.device.type, enabled=False):
                k1, k2, v1, v2 = (
                    t.float().repeat_interleave(repeats, dim=2) for t in (k1, k2, v1, v2)
                )
                scores = torch.einsum("bhd,bjhd,bkhd->bhjk", q.float(), k1, k2)
                probabilities = (scores / math.sqrt(c.head_dim)).flatten(-2).softmax(-1)
                probabilities = probabilities.reshape_as(scores)
                attended = torch.einsum("bhjk,bjhd,bkhd->bhd", probabilities, v1, v2)
        return attended
