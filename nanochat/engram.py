"""Engram conditional memory used by controlled Pareto combinations."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import Linear, norm


@lru_cache(maxsize=None)
def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor <= math.isqrt(value):
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def distinct_prime_sizes(start: int, count: int, divisor: int = 1) -> tuple[int, ...]:
    """Return distinct primes at or above ``start`` with a divisible sum."""
    candidate = max(2, int(start))
    sizes: list[int] = []
    while len(sizes) < count:
        if _is_prime(candidate):
            sizes.append(candidate)
        candidate += 1
    while sum(sizes) % divisor:
        replacement = sizes[-1] + 1
        while not _is_prime(replacement):
            replacement += 1
        sizes[-1] = replacement
    return tuple(sizes)


class ShortCausalConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time = x.size(1)
        x_norm = norm(x).transpose(1, 2)
        y = F.conv1d(
            x_norm,
            self.weight.to(dtype=x.dtype),
            padding=(self.kernel_size - 1) * self.dilation,
            dilation=self.dilation,
            groups=x.size(-1),
        )[..., :time]
        return F.silu(y.transpose(1, 2))


class EngramMemory(nn.Module):
    """One Engram injection point with independent suffix n-gram hash tables."""

    def __init__(
        self,
        *,
        hidden_size: int,
        memory_dim: int,
        vocab_size: int,
        vocab_multiplier: int,
        num_heads: int,
        ngram_orders: tuple[int, ...],
        layer_idx: int,
        seed: int,
        kernel_size: int,
        optimizer_world_size: int,
    ):
        super().__init__()
        if memory_dim % (len(ngram_orders) * num_heads):
            raise ValueError("Engram memory width must divide evenly across tables")
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.ngram_orders = tuple(ngram_orders)
        self.max_ngram_order = max(ngram_orders)
        self.head_dim = memory_dim // (len(ngram_orders) * num_heads)
        self.hidden_size = hidden_size
        self.seed = seed

        table_count = len(ngram_orders) * num_heads
        table_sizes = distinct_prime_sizes(
            vocab_size * vocab_multiplier,
            table_count,
            divisor=optimizer_world_size,
        )
        offsets = [0]
        for size in table_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self._table_sizes_python = table_sizes
        self._offsets_python = tuple(offsets)
        self.register_buffer("table_sizes", torch.empty(table_count, dtype=torch.long), persistent=False)
        self.register_buffer("offsets", torch.empty(table_count, dtype=torch.long), persistent=False)
        self.register_buffer("multipliers", torch.empty(self.max_ngram_order, dtype=torch.long), persistent=False)

        self.embedding = nn.Embedding(sum(table_sizes), self.head_dim)
        self.key_proj = Linear(memory_dim, hidden_size, bias=False)
        self.value_proj = Linear(memory_dim, hidden_size, bias=False)
        self.short_conv = ShortCausalConv(hidden_size, kernel_size, self.max_ngram_order)

    @torch.no_grad()
    def init_weights(self) -> None:
        self.table_sizes.copy_(torch.tensor(self._table_sizes_python, device=self.table_sizes.device))
        self.offsets.copy_(torch.tensor(self._offsets_python, device=self.offsets.device))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 10007 * self.layer_idx)
        multipliers = torch.randint(
            1,
            2**31 - 1,
            (self.max_ngram_order,),
            generator=generator,
            dtype=torch.long,
        )
        multipliers |= 1
        self.multipliers.copy_(multipliers.to(self.multipliers.device))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        bound = self.key_proj.in_features**-0.5
        nn.init.uniform_(self.key_proj.weight, -bound, bound)
        nn.init.uniform_(self.value_proj.weight, -bound, bound)
        nn.init.zeros_(self.short_conv.weight)

    def _hash_ids(self, compressed_ids: torch.Tensor, pad_id: torch.Tensor) -> torch.Tensor:
        batch, time = compressed_ids.shape
        shifts = [compressed_ids]
        for distance in range(1, self.max_ngram_order):
            prefix = pad_id.expand(batch, distance)
            shifts.append(torch.cat((prefix, compressed_ids[:, : time - distance]), dim=1))
        hashes = []
        table_idx = 0
        for order in self.ngram_orders:
            mixed = shifts[0] * self.multipliers[0]
            for distance in range(1, order):
                mixed = torch.bitwise_xor(mixed, shifts[distance] * self.multipliers[distance])
            for _ in range(self.num_heads):
                hashes.append(
                    torch.remainder(mixed, self.table_sizes[table_idx]) + self.offsets[table_idx]
                )
                table_idx += 1
        return torch.stack(hashes, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        compressed_ids: torch.Tensor,
        pad_id: torch.Tensor,
    ) -> torch.Tensor:
        hash_ids = self._hash_ids(compressed_ids, pad_id)
        retrieved = self.embedding(hash_ids).flatten(start_dim=-2).to(hidden_states.dtype)
        key = self.key_proj(retrieved)
        value = self.value_proj(retrieved)
        gate_logits = (norm(hidden_states) * norm(key)).sum(dim=-1) / math.sqrt(self.hidden_size)
        gate_logits = gate_logits.abs().clamp_min(1e-6).sqrt() * gate_logits.sign()
        gated_value = torch.sigmoid(gate_logits).unsqueeze(-1) * value
        return gated_value + self.short_conv(gated_value)
