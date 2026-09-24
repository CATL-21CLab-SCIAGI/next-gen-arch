"""BF16 Engram table prototype for SGLang's TP-only V4.1 integration.

The reviewed upstream EngramEmbedding hardcodes FP8 storage and scales. This
module provides its ordinary TP lookup interface without requantizing trained
BF16 rows. Integration must disable DP/CP, host tables and speculative decoding
until separately qualified. It is not installed into any live engine.
"""

from __future__ import annotations

import torch
from torch import nn


class BF16EngramEmbedding(nn.Module):
    def __init__(self, num_embeddings, dim, *, tp_rank, tp_size, all_reduce, device="cpu"):
        super().__init__()
        if not (0 <= tp_rank < tp_size and num_embeddings > 0 and dim > 0):
            raise ValueError("invalid Engram TP geometry")
        self.num_embeddings, self.dim, self.tp_size = num_embeddings, dim, tp_size
        self.row_start = num_embeddings * tp_rank // tp_size
        self.rows = num_embeddings * (tp_rank + 1) // tp_size - self.row_start
        self.all_reduce = all_reduce
        self.host_table = None
        self.weight = nn.Parameter(torch.empty(self.rows, dim, dtype=torch.bfloat16,
                                               device=device), requires_grad=False)
        self.weight.weight_loader = self._load_rows
        self._loaded = False
        self._pieces = []

    @torch.no_grad()
    def _load_rows(self, param, loaded_weight):
        if (param is not self.weight or loaded_weight.dtype != torch.bfloat16
                or tuple(loaded_weight.shape) != (self.num_embeddings, self.dim)):
            raise ValueError("Engram requires the full BF16 tensor, with no implicit casting")
        if self._pieces or self._loaded:
            raise ValueError("Engram table was already partially or fully loaded")
        param.copy_(loaded_weight[self.row_start:self.row_start + self.rows])
        self._loaded = True

    @torch.no_grad()
    def load_piece(self, row_start, weight):
        if (self._loaded or type(row_start) is not int or row_start < 0
                or weight.ndim != 2 or weight.shape[1] != self.dim
                or weight.dtype != torch.bfloat16 or weight.shape[0] <= 0
                or row_start + weight.shape[0] > self.num_embeddings):
            raise ValueError("invalid BF16 Engram row shard")
        first = max(row_start, self.row_start)
        last = min(row_start + weight.shape[0], self.row_start + self.rows)
        if first >= last:
            return
        lo, hi = first - self.row_start, last - self.row_start
        if any(lo < previous_hi and hi > previous_lo for previous_lo, previous_hi in self._pieces):
            raise ValueError("duplicate or overlapping Engram row shards")
        self.weight[lo:hi].copy_(weight[first - row_start:last - row_start])
        self._pieces.append((lo, hi))

    def finish_load(self, label):
        if not self._loaded:
            end = 0
            for first, last in sorted(self._pieces):
                if first != end:
                    raise ValueError(f"Engram table has a missing row interval: {label}")
                end = last
            if end != self.rows:
                raise ValueError(f"Engram table was not loaded: {label}")
            self._loaded = True

    def owned_rows(self, indices):
        if not self._loaded:
            raise ValueError("Engram table was not loaded")
        if indices.dtype not in (torch.int32, torch.int64) or indices.device != self.weight.device:
            raise ValueError("integer indices must share the table device")
        if self.rows == 0:
            return self.weight.new_zeros((*indices.shape, self.dim))
        local = indices - self.row_start
        owned = (local >= 0) & (local < self.rows)
        values = self.weight[local.masked_fill(~owned, 0).long()]
        return values.masked_fill(~owned.unsqueeze(-1), 0)

    def forward(self, indices, forward_batch=None, *, cp_all_tokens=False):
        if cp_all_tokens:
            raise ValueError("prototype supports TP-only lookup, without context parallelism")
        if self.tp_size == 32:
            from sglang.srt.layers.engram import EngramEmbedding
            # The native gather/reduce/scatter protocol supports idle DP ranks;
            # only the owned-row storage/lookup is replaced with exact BF16.
            return EngramEmbedding._dp_sharded_lookup(self, indices, forward_batch)
        values = self.owned_rows(indices)
        if values.numel() and self.tp_size > 1:
            values = self.all_reduce(values)
        return values

    def _owned_rows(self, indices):
        return self.owned_rows(indices)

    def _empty(self, indices):
        return self.weight.new_empty((*indices.shape, self.dim))
