"""Useful matmul FLOPs for the native Limite normal-prelude comparison.

Count operations, rather than unique parameters: the tied token embedding is
also an output matrix, whereas the value embedding is only a lookup. Attention
counts valid causal/window entries. Activation recomputation, padded tiles,
optimizer operations and communication are excluded from model FLOPs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineFlops:
    linear_per_token: int
    attention_per_token: float
    sequence_length: int

    @property
    def per_token(self) -> float:
        return self.linear_per_token + self.attention_per_token

    def mfu(self, tokens: int, seconds: float, world_size: int, peak: float = 2.25e15) -> float:
        """Whole-step useful MFU against dense BF16 B300 peak per GPU."""
        if tokens < 0 or seconds <= 0 or world_size < 1 or peak <= 0:
            raise ValueError("invalid MFU measurement")
        return tokens * self.per_token / (seconds * world_size * peak)


def causal_pairs(length: int, span: int | None = None) -> int:
    if length < 1 or (span is not None and span < 1):
        raise ValueError("attention length/span must be positive")
    width = min(length, span) if span is not None else length
    return width * (width + 1) // 2 + (length - width) * width


def baseline_flops(config, sequence_length: int, trainable_mode: str = "full") -> BaselineFlops:
    """Analytical ledger for one native attention prelude per native block.

    Frozen backbone matrices still calculate activation gradients between
    trainable preludes, but omit their weight-gradient GEMM (4 rather than 6
    FLOPs per coefficient). Attention activations require all three products
    in either training scope. The publisher calls each MUDD site twice.
    """
    if trainable_mode not in ("full", "adapter"):
        raise ValueError("unsupported trainable mode")
    if getattr(config, "mlp_type", "swiglu") != "swiglu":
        raise ValueError("ledger requires native SwiGLU")
    length = sequence_length
    causal_pairs(length)
    width = config.hidden_size
    layers = config.num_hidden_layers
    heads, kv_heads, head_dim = (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    )
    globals_ = set(config.global_layers)
    if any(index < 0 or index >= layers for index in globals_):
        raise ValueError("invalid global layer indices")
    # QKV plus O; native per-head gate has one channel vector per Q head.
    attention_matrices = layers * width * head_dim * (2 * heads + 2 * kv_heads)
    gates = layers * heads * config.attn_gate_channels
    ve_gates = len(set(config.ve_layers)) * kv_heads * config.ve_gate_channels
    adapter_matrices = attention_matrices + gates + ve_gates
    mlp_matrices = layers * 3 * width * config.intermediate_size
    head_matrix = config.vocab_size * width
    mudd_matrices = 0
    if config.mudd:
        # Attention and optional residual ways share the first-layer matrix.
        ways = 2 if config.mudd_mlp else 1
        mudd_matrices = sum(
            ways * config.mudd_inter * (width + len(config.mudd_tap_idx[str(index)]))
            for index in config.mudd_layers
        )
    backbone_matrices = (
        attention_matrices + gates + ve_gates + mlp_matrices + head_matrix + mudd_matrices
    )
    linear = 6 * adapter_matrices + (6 if trainable_mode == "full" else 4) * backbone_matrices
    global_pairs = causal_pairs(length)
    # Publisher configs normalize the serialized inclusive look-back distance
    # to an actual key span. Portable serialized configs still store distance.
    local_span = (
        config.sliding_window
        if hasattr(config, "_serialized_sliding_window")
        else config.sliding_window + 1
    )
    local_pairs = causal_pairs(length, local_span)
    # Two attention stacks; QK and PV each cost 2*D per valid pair, and
    # their backward matmuls cost twice their forward matmuls.
    pair_count = 2 * (len(globals_) * global_pairs + (layers - len(globals_)) * local_pairs)
    attention = 12 * heads * head_dim * pair_count / length
    return BaselineFlops(linear, attention, length)
