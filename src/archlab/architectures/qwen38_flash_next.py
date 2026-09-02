"""Quarter-shape text model derived from the public Qwen3.8-Flash-Next config.

This is a clean repository-owned implementation for pretraining experiments.  It
does not import or patch Hugging Face, Megatron Core, or Transformer Engine
sources.  Transformer Engine is selected only at the execution boundary so the
same architecture has a small CPU reference path for contract tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Qwen38FlashNextConfig:
    # Text vocabulary is intentionally not divided: token IDs retain their meaning.
    vocab_size: int = 248_320
    sequence_len: int = 2_048
    max_position_embeddings: int = 262_144
    num_hidden_layers: int = 12
    hidden_size: int = 640
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25

    # One QSA layer follows every three GDN layers.
    full_attention_interval: int = 4
    attention_heads: int = 6
    attention_kv_heads: int = 1
    attention_head_dim: int = 64
    indexer_heads: int = 1
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 32
    indexer_budget: int = 512

    linear_qk_heads: int = 4
    linear_v_heads: int = 12
    linear_key_dim: int = 32
    linear_value_dim: int = 32
    linear_conv_kernel: int = 4

    num_experts: int = 128
    num_experts_per_token: int = 3
    moe_intermediate_size: int = 160
    shared_expert_intermediate_size: int = 160
    router_aux_loss_coefficient: float = 0.01
    router_z_loss_coefficient: float = 0.001

    residual_streams: int = 1
    residual_low_rank: int = 80

    ngram_vocab_size: int = 5_000_000
    ngram_orders: tuple[int, ...] = (2, 3)
    ngram_heads_per_order: int = 2
    ngram_embedding_dim: int = 640
    ngram_layer: int = 0

    mtp_layers: int = 1
    mtp_loss_weight: float = 0.1
    tie_word_embeddings: bool = False
    arch_family: str = "qwen38_flash_next_quarter"

    def __post_init__(self) -> None:
        positive = (
            self.vocab_size,
            self.sequence_len,
            self.num_hidden_layers,
            self.hidden_size,
            self.full_attention_interval,
            self.attention_heads,
            self.attention_kv_heads,
            self.attention_head_dim,
            self.indexer_heads,
            self.indexer_kv_heads,
            self.indexer_head_dim,
            self.indexer_budget,
            self.linear_qk_heads,
            self.linear_v_heads,
            self.linear_key_dim,
            self.linear_value_dim,
            self.num_experts,
            self.num_experts_per_token,
            self.moe_intermediate_size,
            self.shared_expert_intermediate_size,
            self.residual_streams,
            self.residual_low_rank,
            self.ngram_vocab_size,
            self.ngram_heads_per_order,
            self.ngram_embedding_dim,
        )
        if min(positive) < 1:
            raise ValueError("Qwen3.8 quarter-shape dimensions must be positive")
        if self.residual_streams != 1:
            raise ValueError("the quarter-shape contract resolves four residual streams to one")
        if self.attention_heads % self.attention_kv_heads:
            raise ValueError("attention KV heads must divide query heads")
        if self.linear_v_heads % self.linear_qk_heads:
            raise ValueError("linear QK heads must divide value heads")
        if self.indexer_heads % self.indexer_kv_heads:
            raise ValueError("indexer KV heads must divide query heads")
        if not 0 < self.num_experts_per_token <= self.num_experts:
            raise ValueError("invalid routed expert count")
        ngram_branches = len(self.ngram_orders) * self.ngram_heads_per_order
        if self.ngram_embedding_dim % ngram_branches:
            raise ValueError("ngram embedding width must divide evenly across hash branches")
        rotary_dim = int(self.attention_head_dim * self.partial_rotary_factor)
        if rotary_dim < 2 or rotary_dim % 2:
            raise ValueError("partial RoPE must retain a positive even dimension")
        if self.mtp_layers != 1:
            raise ValueError("the minimum-one scaling contract retains one MTP layer")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["ngram_orders"] = list(self.ngram_orders)
        return payload


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight.to(x.dtype), eps=self.eps)


class NativeLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.weight.to(x.dtype), bias)


def _linear(in_features: int, out_features: int, *, runtime_backend: str, bias: bool = False):
    if runtime_backend == "native":
        return NativeLinear(in_features, out_features, bias=bias)
    if runtime_backend == "te_fp4":
        # The pinned TE 2.16 kernels require K to be divisible by 32 and the
        # weight's first dimension (N) to be divisible by the 16-value NVFP4
        # block. Keep exact quarter-scaled shapes outside those constraints in
        # BF16 instead of rounding architecture dimensions.
        if in_features % 32 or out_features % 16:
            return NativeLinear(in_features, out_features, bias=bias)
        import transformer_engine.pytorch as te

        return te.Linear(
            in_features,
            out_features,
            bias=bias,
            params_dtype=torch.float32,
        )
    raise ValueError(f"unsupported Qwen3.8 runtime backend: {runtime_backend}")


class NativeGroupedLinear(nn.Module):
    """CPU oracle for Transformer Engine's sorted-token GroupedLinear."""

    def __init__(self, experts: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(experts, out_features, in_features))
        self.out_features = out_features

    def forward(self, inputs: torch.Tensor, m_splits: torch.Tensor) -> torch.Tensor:
        outputs = []
        offset = 0
        for expert, size in enumerate(m_splits.detach().cpu().tolist()):
            stop = offset + int(size)
            if stop > offset:
                outputs.append(F.linear(inputs[offset:stop], self.weight[expert].to(inputs.dtype)))
            offset = stop
        if not outputs:
            return inputs.new_empty((0, self.out_features))
        return torch.cat(outputs, dim=0)


def _raw_grouped_linear(
    experts: int,
    in_features: int,
    out_features: int,
    *,
    runtime_backend: str,
):
    if runtime_backend == "native":
        return NativeGroupedLinear(experts, in_features, out_features)
    if runtime_backend == "te_fp4":
        import transformer_engine.pytorch as te

        return te.GroupedLinear(
            experts,
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.float32,
            single_grouped_weight=True,
        )
    raise ValueError(f"unsupported Qwen3.8 runtime backend: {runtime_backend}")


class ChunkedGroupedLinear(nn.Module):
    """Run sorted experts in bounded groups while preserving their row order."""

    def __init__(
        self,
        experts: int,
        in_features: int,
        out_features: int,
        *,
        runtime_backend: str,
        max_experts_per_group: int,
    ):
        super().__init__()
        if max_experts_per_group < 1:
            raise ValueError("grouped-linear expert limit must be positive")
        self.experts = experts
        self.out_features = out_features
        self.group_sizes = tuple(
            min(max_experts_per_group, experts - offset)
            for offset in range(0, experts, max_experts_per_group)
        )
        self.groups = nn.ModuleList(
            [
                _raw_grouped_linear(
                    group_size,
                    in_features,
                    out_features,
                    runtime_backend=runtime_backend,
                )
                for group_size in self.group_sizes
            ]
        )

    def forward(self, inputs: torch.Tensor, m_splits: torch.Tensor) -> torch.Tensor:
        if m_splits.numel() != self.experts:
            raise ValueError(
                f"expected {self.experts} grouped-linear splits, found {m_splits.numel()}"
            )
        outputs = []
        offset = 0
        split_groups = torch.split(m_splits, self.group_sizes)
        for module, splits in zip(self.groups, split_groups, strict=True):
            rows = int(splits.detach().sum().cpu().item())
            if rows:
                outputs.append(module(inputs[offset : offset + rows], splits))
            else:
                outputs.append(inputs.new_empty((0, self.out_features)))
            offset += rows
        if offset != inputs.size(0):
            raise ValueError("grouped-linear splits do not sum to the input row count")
        return torch.cat(outputs, dim=0)


def _grouped_linear(experts: int, in_features: int, out_features: int, *, runtime_backend: str):
    # Transformer Engine 2.16's grouped Hadamard quantizer accepts at most 64
    # tensors per kernel. Qwen3.8 quarter-shape retains 128 experts, so expose
    # two ordinary 64-GEMM TE modules without altering any expert dimensions.
    if runtime_backend == "te_fp4" and experts > 64:
        return ChunkedGroupedLinear(
            experts,
            in_features,
            out_features,
            runtime_backend=runtime_backend,
            max_experts_per_group=64,
        )
    return _raw_grouped_linear(
        experts,
        in_features,
        out_features,
        runtime_backend=runtime_backend,
    )


def _apply_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    rotary = x[..., :rotary_dim]
    remainder = x[..., rotary_dim:]
    even, odd = rotary[..., 0::2], rotary[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    return torch.cat((rotated.flatten(-2), remainder), dim=-1)


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cuda":
            try:
                from fla.modules.convolution import causal_conv1d
            except ImportError as error:  # pragma: no cover - container probe
                raise RuntimeError(
                    "Qwen3.8 GDN requires the validated fla-core CUDA kernel"
                ) from error
            output, _state = causal_conv1d(
                x=x,
                weight=self.weight.to(x.dtype),
                activation="silu",
                backend="triton",
            )
            return output
        output = F.conv1d(
            x.transpose(1, 2),
            self.weight.to(x.dtype).unsqueeze(1),
            padding=self.kernel_size - 1,
            groups=x.size(-1),
        )[..., : x.size(1)]
        return F.silu(output.transpose(1, 2))


def _gated_delta_reference(q, k, v, decay, beta):
    q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
    v, decay, beta = v.float(), decay.float(), beta.float()
    state = q.new_zeros(q.size(0), q.size(2), q.size(3), v.size(3))
    outputs = []
    for token in range(q.size(1)):
        state = state * decay[:, token].exp().unsqueeze(-1).unsqueeze(-1)
        prediction = torch.einsum("bhk,bhkv->bhv", k[:, token], state)
        update = beta[:, token].unsqueeze(-1) * (v[:, token] - prediction)
        state = state + k[:, token].unsqueeze(-1) * update.unsqueeze(-2)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, token]))
    return torch.stack(outputs, dim=1).to(q.dtype)


class GatedDeltaAttention(nn.Module):
    def __init__(self, config: Qwen38FlashNextConfig, *, runtime_backend: str):
        super().__init__()
        self.q_heads = config.linear_qk_heads
        self.v_heads = config.linear_v_heads
        self.key_dim = config.linear_key_dim
        self.value_dim = config.linear_value_dim
        q_width = self.q_heads * self.key_dim
        v_width = self.v_heads * self.value_dim
        self.qkvz = _linear(
            config.hidden_size,
            2 * q_width + 2 * v_width,
            runtime_backend=runtime_backend,
        )
        self.ba = _linear(
            config.hidden_size,
            2 * self.v_heads,
            runtime_backend=runtime_backend,
        )
        self.conv = CausalDepthwiseConv1d(2 * q_width + v_width, config.linear_conv_kernel)
        self.a_log = nn.Parameter(torch.zeros(self.v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.full((self.v_heads,), -4.0, dtype=torch.float32))
        self.output_norm = RMSNorm(self.value_dim, config.rms_norm_eps)
        self.out = _linear(v_width, config.hidden_size, runtime_backend=runtime_backend)

    def _kernel(self, q, k, v, decay, beta):
        if q.device.type == "cuda":
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            except ImportError as error:  # pragma: no cover - container probe
                raise RuntimeError("Qwen3.8 GDN requires fla-core==0.4.0") from error
            output, _state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=decay,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
            )
            return output
        return _gated_delta_reference(q, k, v, decay, beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q_width = self.q_heads * self.key_dim
        v_width = self.v_heads * self.value_dim
        q, k, v, z = torch.split(
            self.qkvz(x),
            (q_width, q_width, v_width, v_width),
            dim=-1,
        )
        qkv = self.conv(torch.cat((q, k, v), dim=-1))
        q, k, v = torch.split(qkv, (q_width, q_width, v_width), dim=-1)
        beta_logits, decay_logits = self.ba(x).chunk(2, dim=-1)
        beta = beta_logits.sigmoid()
        decay = -self.a_log.exp().view(1, 1, -1) * F.softplus(
            decay_logits.float() + self.dt_bias.view(1, 1, -1)
        )
        q = q.view(batch, seq_len, self.q_heads, self.key_dim)
        k = k.view(batch, seq_len, self.q_heads, self.key_dim)
        v = v.view(batch, seq_len, self.v_heads, self.value_dim)
        repeat = self.v_heads // self.q_heads
        q = q.repeat_interleave(repeat, dim=2)
        k = k.repeat_interleave(repeat, dim=2)
        output = self._kernel(q, k, v, decay, beta)
        output = self.output_norm(output)
        output = output * F.silu(z.view(batch, seq_len, self.v_heads, self.value_dim))
        return self.out(output.flatten(-2))


class QwenSparseAttention(nn.Module):
    """QSA reference: learned top-k token indexer plus GQA and 25% RoPE."""

    def __init__(self, config: Qwen38FlashNextConfig, *, runtime_backend: str):
        super().__init__()
        self.q_heads = config.attention_heads
        self.kv_heads = config.attention_kv_heads
        self.head_dim = config.attention_head_dim
        self.index_heads = config.indexer_heads
        self.index_kv_heads = config.indexer_kv_heads
        self.index_dim = config.indexer_head_dim
        self.budget = config.indexer_budget
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        self.q = _linear(
            config.hidden_size,
            self.q_heads * self.head_dim,
            runtime_backend=runtime_backend,
        )
        self.k = _linear(
            config.hidden_size,
            self.kv_heads * self.head_dim,
            runtime_backend=runtime_backend,
        )
        self.v = _linear(
            config.hidden_size,
            self.kv_heads * self.head_dim,
            runtime_backend=runtime_backend,
        )
        self.index_q = _linear(
            config.hidden_size,
            self.index_heads * self.index_dim,
            runtime_backend=runtime_backend,
        )
        self.index_k = _linear(
            config.hidden_size,
            self.index_kv_heads * self.index_dim,
            runtime_backend=runtime_backend,
        )
        self.out = _linear(
            self.q_heads * self.head_dim,
            config.hidden_size,
            runtime_backend=runtime_backend,
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q(x).view(batch, seq_len, self.q_heads, self.head_dim)
        k = self.k(x).view(batch, seq_len, self.kv_heads, self.head_dim)
        v = self.v(x).view(batch, seq_len, self.kv_heads, self.head_dim)
        q = _apply_partial_rope(q, cos, sin, self.rotary_dim)
        k = _apply_partial_rope(k, cos, sin, self.rotary_dim)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        index_q = self.index_q(x).view(batch, seq_len, self.index_heads, self.index_dim)
        index_k = self.index_k(x).view(batch, seq_len, self.index_kv_heads, self.index_dim)
        if self.index_heads != self.index_kv_heads:
            index_k = index_k.repeat_interleave(self.index_heads // self.index_kv_heads, dim=2)
        index_scores = torch.einsum("bthd,bshd->bhts", index_q, index_k)
        index_scores = index_scores * self.index_dim**-0.5

        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        candidates = index_scores.masked_fill(~causal.view(1, 1, seq_len, seq_len), -torch.inf)
        selected = candidates.topk(min(self.budget, seq_len), dim=-1).indices
        sparse_mask = torch.zeros_like(candidates, dtype=torch.bool)
        sparse_mask.scatter_(-1, selected, True)
        sparse_mask &= causal.view(1, 1, seq_len, seq_len)
        sparse_mask = sparse_mask.any(dim=1, keepdim=True)

        if self.q_heads != self.kv_heads:
            repeat = self.q_heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)
        logits = torch.einsum("bthd,bshd->bhts", q, k) * self.head_dim**-0.5
        # The selected index score is a differentiable routing bias; the top-k
        # boundary itself intentionally remains discrete.
        index_bias = index_scores.mean(dim=1, keepdim=True)
        logits = logits + index_bias
        logits = logits.masked_fill(~sparse_mask, -torch.inf)
        probabilities = F.softmax(logits.float(), dim=-1).to(v.dtype)
        output = torch.einsum("bhts,bshd->bthd", probabilities, v)
        return self.out(output.flatten(-2))


def _pad_grouped_tokens(
    inputs: torch.Tensor,
    m_splits: torch.Tensor,
    *,
    multiple: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad each sorted expert segment and return indices of the real rows."""
    if multiple < 1:
        raise ValueError("grouped-token padding multiple must be positive")
    sizes = [int(size) for size in m_splits.detach().cpu().tolist()]
    padded_chunks = []
    real_indices = []
    padded_sizes = []
    source_offset = 0
    padded_offset = 0
    for size in sizes:
        padded_size = ((size + multiple - 1) // multiple) * multiple
        padded_sizes.append(padded_size)
        if size:
            padded_chunks.append(inputs[source_offset : source_offset + size])
            real_indices.append(
                torch.arange(padded_offset, padded_offset + size, device=inputs.device)
            )
        if padded_size > size:
            padded_chunks.append(inputs.new_zeros((padded_size - size, inputs.size(-1))))
        source_offset += size
        padded_offset += padded_size
    if source_offset != inputs.size(0):
        raise ValueError("grouped-token splits do not sum to the input row count")
    padded = (
        torch.cat(padded_chunks, dim=0) if padded_chunks else inputs.new_empty((0, inputs.size(-1)))
    )
    indices = (
        torch.cat(real_indices)
        if real_indices
        else torch.empty(0, dtype=torch.long, device=inputs.device)
    )
    splits = torch.tensor(padded_sizes, dtype=m_splits.dtype, device=m_splits.device)
    return padded, splits, indices


class SparseMoE(nn.Module):
    def __init__(self, config: Qwen38FlashNextConfig, *, runtime_backend: str):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.router_aux_coefficient = config.router_aux_loss_coefficient
        self.router_z_coefficient = config.router_z_loss_coefficient
        # TE 2.16's grouped NVFP4 Hadamard transform requires every expert's
        # M split to be a multiple of 64 (a stricter bound than GEMM's M%16).
        self.grouped_token_multiple = 64 if runtime_backend == "te_fp4" else 1
        self.router = NativeLinear(config.hidden_size, self.num_experts, bias=False)
        self.expert_up = _grouped_linear(
            self.num_experts,
            config.hidden_size,
            2 * config.moe_intermediate_size,
            runtime_backend=runtime_backend,
        )
        self.expert_down = _grouped_linear(
            self.num_experts,
            config.moe_intermediate_size,
            config.hidden_size,
            runtime_backend=runtime_backend,
        )
        self.shared_up = _linear(
            config.hidden_size,
            2 * config.shared_expert_intermediate_size,
            runtime_backend=runtime_backend,
        )
        self.shared_down = _linear(
            config.shared_expert_intermediate_size,
            config.hidden_size,
            runtime_backend=runtime_backend,
        )
        self.shared_gate = NativeLinear(config.hidden_size, 1, bias=False)
        self.last_aux_loss = torch.tensor(0.0)

    @staticmethod
    def _swiglu(value: torch.Tensor) -> torch.Tensor:
        gate, linear = value.chunk(2, dim=-1)
        return F.silu(gate) * linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        router_logits = self.router(flat).float()
        top_weights, top_experts = router_logits.topk(self.top_k, dim=-1)
        top_weights = F.softmax(top_weights, dim=-1).to(x.dtype)

        expanded = flat.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, shape[-1])
        expert_ids = top_experts.reshape(-1)
        order = expert_ids.argsort(stable=True)
        sorted_inputs = expanded[order]
        m_splits = torch.bincount(expert_ids, minlength=self.num_experts).to(torch.int32)
        if self.grouped_token_multiple > 1:
            grouped_inputs, grouped_splits, real_indices = _pad_grouped_tokens(
                sorted_inputs,
                m_splits,
                multiple=self.grouped_token_multiple,
            )
        else:
            grouped_inputs, grouped_splits, real_indices = sorted_inputs, m_splits, None
        routed = self.expert_up(grouped_inputs, grouped_splits)
        routed = self._swiglu(routed)
        routed = self.expert_down(routed, grouped_splits)
        if real_indices is not None:
            routed = routed.index_select(0, real_indices)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        routed = routed[inverse].view(flat.size(0), self.top_k, -1)
        routed = (routed * top_weights.unsqueeze(-1)).sum(dim=1)

        shared = self.shared_down(self._swiglu(self.shared_up(flat)))
        shared = shared * self.shared_gate(flat).sigmoid()

        probabilities = F.softmax(router_logits, dim=-1)
        assignments = F.one_hot(top_experts, self.num_experts).float().mean(dim=(0, 1))
        importance = probabilities.mean(dim=0)
        balance = self.num_experts * (assignments * importance).sum()
        z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
        self.last_aux_loss = (
            self.router_aux_coefficient * balance + self.router_z_coefficient * z_loss
        )
        return (routed + shared).view(shape)


class SingleStreamResidual(nn.Module):
    """Minimum-one form of Qwen's four-stream gated residual connection."""

    def __init__(self, config: Qwen38FlashNextConfig, *, runtime_backend: str):
        super().__init__()
        self.down = _linear(
            config.hidden_size,
            config.residual_low_rank,
            runtime_backend=runtime_backend,
        )
        self.read = _linear(
            config.residual_low_rank,
            config.hidden_size,
            runtime_backend=runtime_backend,
        )
        self.write = _linear(config.residual_low_rank, 1, runtime_backend=runtime_backend)

    def gates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = F.silu(self.down(x))
        return 2.0 * self.read(latent).sigmoid(), 2.0 * self.write(latent).sigmoid()

    def prepare(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        read, write = self.gates(x)
        return x * read, write

    @staticmethod
    def combine(x: torch.Tensor, branch: torch.Tensor, write: torch.Tensor) -> torch.Tensor:
        return x + write * branch


class Qwen38Block(nn.Module):
    def __init__(
        self,
        config: Qwen38FlashNextConfig,
        layer_idx: int,
        *,
        runtime_backend: str,
        force_sparse_attention: bool = False,
    ):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        use_sparse = force_sparse_attention or (layer_idx + 1) % config.full_attention_interval == 0
        self.attention_kind = "qsa" if use_sparse else "gdn"
        self.attention = (
            QwenSparseAttention(config, runtime_backend=runtime_backend)
            if use_sparse
            else GatedDeltaAttention(config, runtime_backend=runtime_backend)
        )
        self.attention_residual = SingleStreamResidual(config, runtime_backend=runtime_backend)
        self.moe = SparseMoE(config, runtime_backend=runtime_backend)
        self.moe_residual = SingleStreamResidual(config, runtime_backend=runtime_backend)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        branch, write = self.attention_residual.prepare(self.input_norm(x))
        if self.attention_kind == "qsa":
            branch = self.attention(branch, cos, sin)
        else:
            branch = self.attention(branch)
        x = self.attention_residual.combine(x, branch, write)
        branch, write = self.moe_residual.prepare(self.moe_norm(x))
        branch = self.moe(branch)
        return self.moe_residual.combine(x, branch, write)


class NGramPLE(nn.Module):
    """Hashed bigram/trigram PLE with quartered vocab, heads, and width."""

    def __init__(self, config: Qwen38FlashNextConfig):
        super().__init__()
        self.vocab_size = config.ngram_vocab_size
        self.orders = config.ngram_orders
        self.heads = config.ngram_heads_per_order
        self.branch_dim = config.ngram_embedding_dim // (len(self.orders) * self.heads)
        self.tables = nn.ModuleList(
            [
                nn.Embedding(self.vocab_size, self.branch_dim)
                for _ in range(len(self.orders) * self.heads)
            ]
        )
        self.output_norm = RMSNorm(config.ngram_embedding_dim, config.rms_norm_eps)

    def _hash(self, token_ids: torch.Tensor, order: int, head: int) -> torch.Tensor:
        value = torch.zeros_like(token_ids)
        prime = 1_000_003 + 97 * head + 53 * order
        for offset in range(order):
            shifted = F.pad(token_ids[:, : token_ids.size(1) - offset], (offset, 0))
            value = (value * prime + shifted + 1 + 17 * head) % self.vocab_size
        return value

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        outputs = []
        branch = 0
        for order in self.orders:
            for head in range(self.heads):
                outputs.append(self.tables[branch](self._hash(token_ids, order, head)))
                branch += 1
        return self.output_norm(torch.cat(outputs, dim=-1))


class Qwen38FlashNext(nn.Module):
    def __init__(self, config: Qwen38FlashNextConfig, *, runtime_backend: str = "native"):
        super().__init__()
        self.config = config
        self.runtime_backend = runtime_backend
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.ngram = NGramPLE(config)
        self.layers = nn.ModuleList(
            [
                Qwen38Block(config, layer, runtime_backend=runtime_backend)
                for layer in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = _linear(
            config.hidden_size,
            config.vocab_size,
            runtime_backend=runtime_backend,
        )
        self.mtp_block = Qwen38Block(
            config,
            config.num_hidden_layers,
            runtime_backend=runtime_backend,
            force_sparse_attention=True,
        )
        rotary_dim = int(config.attention_head_dim * config.partial_rotary_factor)
        positions = torch.arange(config.sequence_len, dtype=torch.float32)
        channels = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
        inverse_frequency = 1.0 / (config.rope_theta ** (channels / rotary_dim))
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    @torch.no_grad()
    def init_weights(self, std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, RMSNorm):
                if not module.weight.is_meta:
                    module.weight.fill_(1.0)
            if isinstance(module, nn.Embedding):
                if not module.weight.is_meta:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
        for name, parameter in self.named_parameters():
            if parameter.is_meta or parameter.ndim < 2 or "embedding" in name or ".tables." in name:
                continue
            nn.init.normal_(parameter, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, GatedDeltaAttention):
                if not module.a_log.is_meta:
                    module.a_log.zero_()
                    module.dt_bias.fill_(-4.0)

    def get_device(self) -> torch.device:
        return self.token_embedding.weight.device

    def _router_aux_loss(self) -> torch.Tensor:
        losses = [layer.moe.last_aux_loss for layer in self.layers]
        losses.append(self.mtp_block.moe.last_aux_loss)
        return torch.stack([loss.to(self.get_device()) for loss in losses]).mean()

    def num_scaling_params(self) -> dict[str, int]:
        ngram = sum(parameter.numel() for parameter in self.ngram.parameters())
        embeddings = self.token_embedding.weight.numel() + self.lm_head.weight.numel()
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "embeddings_and_head": embeddings,
            "ngram_ple": ngram,
            "transformer": total - embeddings - ngram,
            "total": total,
        }

    def estimate_flops(self) -> float:
        # Dense parameter-equivalent estimate; routed active compute is reported separately.
        return 6.0 * float(self.num_scaling_params()["total"])

    def estimate_executed_flops(self) -> float:
        total = self.num_scaling_params()["total"]
        expert_total = 0
        expert_active = 0
        for block in (*self.layers, self.mtp_block):
            routed = sum(parameter.numel() for parameter in block.moe.expert_up.parameters())
            routed += sum(parameter.numel() for parameter in block.moe.expert_down.parameters())
            expert_total += routed
            expert_active += routed * self.config.num_experts_per_token // self.config.num_experts
        active = total - expert_total + expert_active
        return 6.0 * float(active)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        loss_reduction: str = "mean",
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.rope_cos.size(0):
            raise ValueError("sequence exceeds the configured rotary cache")
        seq_len = input_ids.size(1)
        compute_dtype = self.rope_cos.dtype
        if self.get_device().type == "cuda":
            compute_dtype = torch.bfloat16
        x = self.token_embedding(input_ids).to(compute_dtype)
        ngram = self.ngram(input_ids).to(compute_dtype)
        cos = self.rope_cos[:seq_len].to(device=x.device, dtype=x.dtype).view(1, seq_len, 1, -1)
        sin = self.rope_sin[:seq_len].to(device=x.device, dtype=x.dtype).view(1, seq_len, 1, -1)
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == self.config.ngram_layer:
                x = x + ngram
            x = layer(x, cos, sin)
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)[..., : self.config.vocab_size].float()
        if labels is None:
            return logits

        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-1,
            reduction="none",
        ).view_as(labels)
        if seq_len > 1:
            mtp_input = hidden[:, :-1] + self.token_embedding(input_ids[:, 1:]).to(hidden.dtype)
            mtp_hidden = self.mtp_block(mtp_input, cos[:, :-1], sin[:, :-1])
            mtp_logits = self.lm_head(self.final_norm(mtp_hidden)).float()
            mtp_losses = F.cross_entropy(
                mtp_logits.reshape(-1, mtp_logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-1,
                reduction="none",
            ).view(labels.size(0), seq_len - 1)
            losses[:, :-1] = losses[:, :-1] + self.config.mtp_loss_weight * mtp_losses
        valid = labels.ne(-1)
        losses = losses + valid.to(losses.dtype) * self._router_aux_loss()
        if loss_reduction == "none":
            return losses
        if loss_reduction == "sum":
            return losses.sum()
        if loss_reduction == "mean":
            return losses.sum() / valid.sum().clamp_min(1)
        raise ValueError(f"unsupported loss reduction: {loss_reduction}")
