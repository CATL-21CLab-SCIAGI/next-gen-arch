"""Scale Engram capacity and channels together, preserving its prime hash buckets."""

from fractions import Fraction
from math import isqrt


def scaled_engram_geometry(original, width, *, anchor_width=None):
    """Keep historical width scaling unless a fixed lookup budget is declared."""
    if anchor_width is not None and (
        type(anchor_width) is not int or anchor_width <= 0
    ):
        raise ValueError("Engram anchor width must be a positive integer")
    ratio = Fraction(width if anchor_width is None else anchor_width, original["hidden_size"])
    channels = original["engram_head_dim"] * ratio
    # Configure a kernel-aligned channel count, rather than padding tensors
    # during training. Exact fractional channels do not exist at d128/d384.
    channels = max(8, -(-channels.numerator // (channels.denominator * 8)) * 8)
    buckets = max(2, round(original["engram_vocab_size"] * ratio))
    # Same prime-selection order as AutoModel DeepseekV41NgramHash. Counts
    # must cover all prime buckets, not a rounded fraction of the old rows.
    seen, tables = set(), []
    for _ in original["engram_layer_ids"]:
        count = 0
        for _ in range(original["engram_max_ngram_size"] - 1):
            current = buckets - 1
            for _ in range(original["engram_n_heads"]):
                current += 1
                while current in seen or any(
                    current % d == 0 for d in range(2, isqrt(current) + 1)
                ):
                    current += 1
                seen.add(current)
                count += current
        tables.append(count)
    return {
        "engram_head_dim": int(channels),
        "engram_vocab_size": buckets,
        "engram_num_embeddings": tables,
    }
