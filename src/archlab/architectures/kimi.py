"""Controlled Kimi token-mixer and attention-residual architectures.

The surrounding GPT stays identical to the base architecture. KDA replaces only
the configured token mixers, while AttnRes changes only depth-wise residual reads. CUDA uses the KDA
operator from the supported fla-core 0.4 releases; the PyTorch recurrence is a correctness reference
and a CPU test path, never a CUDA fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from archlab.architectures.base import (
    GPT,
    MLP,
    Block,
    GPTConfig,
    HeadSplitLinear,
    Linear,
    has_ve,
    init_projection_uniform_,
    language_model_loss,
    norm,
)

KDA_PATTERN = "KKKG"
KDA_CHUNK_SIZE = 64
KDA_CONV_SIZE = 4
FLA_CORE_VERSION = "0.4.0"
SUPPORTED_FLA_CORE_VERSIONS = (FLA_CORE_VERSION, "0.4.2")
FLA_CORE_WHEEL_SHA256 = "5396f36a9838c99f9e45c70e88e2e0b26688f719d07d2ddd61be16d29327f4ea"


@dataclass
class KimiKDAConfig(GPTConfig):
    arch_family: str = "kimi_kda"
    kda_pattern: str = KDA_PATTERN
    kda_chunk_size: int = KDA_CHUNK_SIZE
    kda_conv_size: int = KDA_CONV_SIZE
    kda_rope_policy: str = "global_only"
    kda_variant: str = "kimi_linear"
    kda_force_final_global: bool = True

    def __post_init__(self) -> None:
        if self.kda_variant not in {"kimi_linear", "kimi_k3", "solar_negative"}:
            raise ValueError(f"unknown kda_variant={self.kda_variant!r}")
        if self.kda_rope_policy not in {"global_only", "none"}:
            raise ValueError("kda_rope_policy must be global_only or none")


def is_kda_layer(
    layer_idx: int,
    n_layer: int,
    pattern: str = KDA_PATTERN,
    force_final_global: bool = True,
) -> bool:
    """Return whether a zero-based layer is KDA; the final layer is always global."""
    if force_final_global and layer_idx == n_layer - 1:
        return False
    char = pattern[layer_idx % len(pattern)].upper()
    if char not in {"K", "G"}:
        raise ValueError(f"Invalid KDA pattern {pattern!r}; use K and G only")
    return char == "K"


def kda_layer_map(
    n_layer: int, pattern: str = KDA_PATTERN, force_final_global: bool = True
) -> dict[str, list[int]]:
    """Return one-based KDA/global layer lists for manifests and tests."""
    kda = [i + 1 for i in range(n_layer) if is_kda_layer(i, n_layer, pattern, force_final_global)]
    global_attn = [
        i + 1 for i in range(n_layer) if not is_kda_layer(i, n_layer, pattern, force_final_global)
    ]
    return {"kda": kda, "global": global_attn}


class LinearWithBias(nn.Linear):
    """Nanochat-style fp32 master weights with an optional cast bias."""

    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)


class ShortConvolution(nn.Module):
    """Depthwise causal conv with a 2-D weight so it can use the Muon matrix group."""

    def __init__(self, hidden_size: int, kernel_size: int = KDA_CONV_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(hidden_size, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(dtype=x.dtype)
        if x.device.type == "cuda":
            try:
                from fla.modules.convolution import causal_conv1d
            except ImportError as exc:  # pragma: no cover - exercised by remote preflight
                raise RuntimeError(
                    f"CUDA KDA requires fla-core=={FLA_CORE_VERSION}; no fallback is allowed"
                ) from exc
            y, _ = causal_conv1d(x=x, weight=weight, activation="silu", backend="triton")
            return y
        # Correctness-only CPU path for unit tests.
        y = F.conv1d(
            x.transpose(1, 2),
            weight[:, None, :],
            padding=self.kernel_size - 1,
            groups=self.hidden_size,
        )[..., : x.size(1)]
        return F.silu(y.transpose(1, 2))


def kda_gate_reference(
    raw: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor, head_dim: int
):
    """Reference for fla.ops.kda.gate.kda_gate_ref."""
    raw = raw + dt_bias.to(raw.dtype)
    raw = raw.view(*raw.shape[:-1], -1, head_dim)
    return -a_log.float().exp().view(*([1] * (raw.ndim - 2)), -1, 1) * F.softplus(raw.float())


def _call_fused_kda_gate(
    fused_kda_gate,
    version: str,
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    n_head: int,
    head_dim: int,
) -> torch.Tensor:
    """Bridge the fla-core 0.4.0 and 0.4.2 KDA gate APIs without changing math."""
    if version == "0.4.0":
        return fused_kda_gate(raw_gate, a_log, head_dim, g_bias=dt_bias)
    if version == "0.4.2":
        shaped = raw_gate.view(*raw_gate.shape[:-1], n_head, head_dim)
        return fused_kda_gate(
            shaped,
            a_log,
            dt_bias=dt_bias,
            output_dtype=raw_gate.dtype,
        )
    supported = ", ".join(SUPPORTED_FLA_CORE_VERSIONS)
    raise RuntimeError(f"Expected fla-core in ({supported}), found {version}")


def kda_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Straightforward Eq. 1 recurrence used for CPU tests and CUDA parity checks."""
    q = F.normalize(q.float(), p=2, dim=-1)
    k = F.normalize(k.float(), p=2, dim=-1)
    v, g, beta = v.float(), g.float(), beta.float()
    batch, _, heads, key_dim = q.shape
    value_dim = v.size(-1)
    state = q.new_zeros(batch, heads, key_dim, value_dim)
    outputs = []
    for t in range(q.size(1)):
        kt, vt, bt = k[:, t], v[:, t], beta[:, t]
        state = state * g[:, t].exp().unsqueeze(-1)
        prediction = torch.einsum("bhk,bhkv->bhv", kt, state)
        update = bt.unsqueeze(-1) * (vt - prediction)
        state = state + kt.unsqueeze(-1) * update.unsqueeze(-2)
        # fla-core's chunk_kda applies its default 1/sqrt(K) query scale
        # after Q/K L2 normalization.
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, t]) * key_dim**-0.5)
    return torch.stack(outputs, dim=1)


class KimiDeltaAttention(nn.Module):
    def __init__(self, config: KimiKDAConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.runtime = config.runtime
        self.head_dim = self.n_embd // self.n_head
        self.chunk_size = config.kda_chunk_size
        self.conv_size = config.kda_conv_size
        self.variant = config.kda_variant
        if self.head_dim <= 0:
            raise ValueError("KDA head dimensions must be positive")
        if self.n_kv_head != self.n_head:
            raise ValueError("Controlled KDA does not support GQA")
        if self.chunk_size != 64:
            raise ValueError("fla-core 0.4.0 KDA has a fixed chunk size of 64")

        if config.per_head_muon:
            self.q_proj = HeadSplitLinear(self.n_embd, self.n_head, self.head_dim)
            self.k_proj = HeadSplitLinear(self.n_embd, self.n_head, self.head_dim)
            self.v_proj = HeadSplitLinear(self.n_embd, self.n_head, self.head_dim)
        else:
            self.q_proj = Linear(self.n_embd, self.n_embd, bias=False)
            self.k_proj = Linear(self.n_embd, self.n_embd, bias=False)
            self.v_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.q_conv = ShortConvolution(self.n_embd, self.conv_size)
        self.k_conv = ShortConvolution(self.n_embd, self.conv_size)
        self.v_conv = ShortConvolution(self.n_embd, self.conv_size)

        self.f_down = Linear(self.n_embd, self.head_dim, bias=False)
        self.f_up = Linear(self.head_dim, self.n_embd, bias=False)
        self.beta_proj = Linear(self.n_embd, self.n_head, bias=False)
        self.a_log = nn.Parameter(torch.empty(self.n_head, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.n_embd, dtype=torch.float32))

        if self.variant in {"kimi_k3", "solar_negative"}:
            self.g_proj = LinearWithBias(self.n_embd, self.n_embd, bias=True)
        else:
            self.g_down = Linear(self.n_embd, self.head_dim, bias=False)
            self.g_up = LinearWithBias(self.head_dim, self.n_embd, bias=True)
        self.output_norm_weight = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32))
        self.o_proj = Linear(self.n_embd, self.n_embd, bias=False)

        self.ve_gate_channels = min(12, self.n_embd)
        self.ve_gate = (
            Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def _gate(self, raw_gate: torch.Tensor) -> torch.Tensor:
        if self.variant == "kimi_k3":
            shaped = (raw_gate + self.dt_bias.to(raw_gate.dtype)).view(
                *raw_gate.shape[:-1], self.n_head, self.head_dim
            )
            scale = self.a_log.float().exp().view(1, 1, self.n_head, 1)
            return (-5.0 * torch.sigmoid(scale * shaped.float())).to(raw_gate.dtype)
        if raw_gate.device.type == "cuda":
            try:
                import fla
                from fla.ops.kda.gate import fused_kda_gate
            except ImportError as exc:  # pragma: no cover - exercised by remote preflight
                raise RuntimeError(
                    "CUDA KDA requires a supported fla-core 0.4 release; no fallback is allowed"
                ) from exc
            version = getattr(fla, "__version__", FLA_CORE_VERSION)
            return _call_fused_kda_gate(
                fused_kda_gate,
                version,
                raw_gate,
                self.a_log,
                self.dt_bias,
                self.n_head,
                self.head_dim,
            )
        return kda_gate_reference(raw_gate, self.a_log, self.dt_bias, self.head_dim)

    def _mix(self, q, k, v, g, beta):
        if q.device.type == "cuda":
            try:
                import fla
                from fla.ops.kda import chunk_kda
            except ImportError as exc:  # pragma: no cover - exercised by remote preflight
                raise RuntimeError(
                    "CUDA KDA requires a supported fla-core 0.4 release; no fallback is allowed"
                ) from exc
            version = getattr(fla, "__version__", FLA_CORE_VERSION)
            if version not in SUPPORTED_FLA_CORE_VERSIONS:
                supported = ", ".join(SUPPORTED_FLA_CORE_VERSIONS)
                raise RuntimeError(f"Expected fla-core in ({supported}), found {version}")
            kernel_head_dim = max(16, self.head_dim)
            padding = kernel_head_dim - self.head_dim
            kernel_q = F.pad(q, (0, padding)) if padding else q
            kernel_k = F.pad(k, (0, padding)) if padding else k
            kernel_v = F.pad(v, (0, padding)) if padding else v
            kernel_g = F.pad(g, (0, padding)) if padding else g
            out, _ = chunk_kda(
                q=kernel_q,
                k=kernel_k,
                v=kernel_v,
                g=kernel_g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
            )
            return out[..., : self.head_dim]
        return kda_recurrent_reference(q, k, v, g, beta).to(q.dtype)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError("KDA KV-cache inference is outside this training ablation")
        batch, seq_len, _ = x.shape
        q = self.q_conv(self.q_proj(x)).view(batch, seq_len, self.n_head, self.head_dim)
        k = self.k_conv(self.k_proj(x)).view(batch, seq_len, self.n_head, self.head_dim)
        v = self.v_conv(self.v_proj(x)).view(batch, seq_len, self.n_head, self.head_dim)

        if ve is not None:
            ve = ve.view(batch, seq_len, self.n_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        raw_gate = self.f_up(self.f_down(x))
        g = self._gate(raw_gate)
        beta = torch.sigmoid(self.beta_proj(x))
        if self.variant == "solar_negative":
            beta = 2.0 * beta
        y = self._mix(q, k, v, g, beta)

        if hasattr(self, "g_proj"):
            output_gate = self.g_proj(x).view(batch, seq_len, self.n_head, self.head_dim)
        else:
            output_gate = self.g_up(self.g_down(x)).view(batch, seq_len, self.n_head, self.head_dim)
        y = F.rms_norm(
            y,
            (self.head_dim,),
            self.output_norm_weight.to(dtype=y.dtype),
            eps=1e-5,
        )
        y = y * torch.sigmoid(output_gate)
        return self.o_proj(y.contiguous().view(batch, seq_len, self.n_embd))


class KimiKDABlock(nn.Module):
    def __init__(self, config: KimiKDAConfig, layer_idx: int):
        super().__init__()
        self.attn = KimiDeltaAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class NoPEGatedAttention(nn.Module):
    """Full-rank elementwise-gated global attention without RoPE."""

    def __init__(self, config: KimiKDAConfig, layer_idx: int):
        super().__init__()
        self.runtime = config.runtime
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        if config.per_head_muon:
            self.c_q = HeadSplitLinear(self.n_embd, self.n_head, self.head_dim)
            self.c_k = HeadSplitLinear(self.n_embd, self.n_head, self.head_dim)
            self.c_v = HeadSplitLinear(self.n_embd, self.n_head, self.head_dim)
        else:
            self.c_q = Linear(self.n_embd, self.n_embd, bias=False)
            self.c_k = Linear(self.n_embd, self.n_embd, bias=False)
            self.c_v = Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.output_gate = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = min(12, self.n_embd)
        self.ve_gate = (
            Linear(self.ve_gate_channels, self.n_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError("NoPE gated global cache inference is outside this campaign")
        batch, time, _ = x.shape
        q = self.c_q(x).view(batch, time, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch, time, self.n_head, self.head_dim)
        v = self.c_v(x).view(batch, time, self.n_head, self.head_dim)
        if ve is not None:
            ve = ve.view_as(v)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        q, k = norm(q) * 1.2, norm(k) * 1.2
        y = self.runtime.attention.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(batch, time, self.n_embd)
        y = (y.float() * torch.sigmoid(self.output_gate(x).float())).to(x.dtype)
        return self.c_proj(y)


class NoPEGatedBlock(nn.Module):
    def __init__(self, config: KimiKDAConfig, layer_idx: int):
        super().__init__()
        self.attn = NoPEGatedAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        return x + self.mlp(norm(x))


class KimiKDA(GPT):
    def __init__(self, config: KimiKDAConfig, pad_vocab_size_to: int = 64):
        nn.Module.__init__(self)
        self.config = config
        if config.kda_variant == "kimi_linear" and config.kda_rope_policy != "global_only":
            raise ValueError("legacy controlled KDA requires RoPE on global layers only")
        if config.kda_variant != "kimi_linear" and config.kda_rope_policy != "none":
            raise ValueError("K3/Solar variants require NoPE global attention")
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            config.runtime.log(
                f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency"
            )
        blocks = []
        for layer_idx in range(config.n_layer):
            if is_kda_layer(
                layer_idx, config.n_layer, config.kda_pattern, config.kda_force_final_global
            ):
                block_cls = KimiKDABlock
            else:
                block_cls = NoPEGatedBlock if config.kda_rope_policy == "none" else Block
            blocks.append(block_cls(config, layer_idx))
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList(blocks),
            }
        )
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate_channels = min(24, config.n_embd)
        self.smear_gate = Linear(self.smear_gate_channels, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        kv_dim = config.n_kv_head * (config.n_embd // config.n_head)
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(padded_vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        # KimiKDA reuses GPT.forward without constructing GPT's optional
        # Engram memories.  Preserve that shared-forward interface explicitly.
        self.engrams = nn.ModuleDict()
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, self.config.n_embd // self.config.n_head
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            if isinstance(block.attn, KimiDeltaAttention):
                attn = block.attn
                for projection in (attn.q_proj, attn.k_proj, attn.v_proj):
                    init_projection_uniform_(projection, -s, s)
                torch.nn.init.zeros_(attn.o_proj.weight)
            else:
                attn = block.attn
                init_projection_uniform_(attn.c_q, -s, s)
                init_projection_uniform_(attn.c_k, -s, s)
                init_projection_uniform_(attn.c_v, -s, s)
                torch.nn.init.zeros_(attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        for i in range(self.config.n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(self.config.n_layer - 1, 1))
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(self.config.n_layer - 1, 1))
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        self.smear_lambda.zero_()
        self.backout_lambda.fill_(0.2)

        # KDA-private convolutions, gates, decay, and optional global output
        # gates live on a separate stream. The main stream above intentionally
        # mirrors GPT.init_weights layer-for-layer so all shape-compatible
        # shared tensors match the paired nanochat baseline exactly.
        device = self.transformer.wte.weight.device
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        kda_seed = torch.initial_seed() ^ 0x4BDA
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(kda_seed)
            for block in self.transformer.h:
                attn = block.attn
                if isinstance(attn, KimiDeltaAttention):
                    conv_bound = self.config.kda_conv_size**-0.5
                    for conv in (attn.q_conv, attn.k_conv, attn.v_conv):
                        torch.nn.init.uniform_(conv.weight, -conv_bound, conv_bound)
                    gate_projections = (
                        (attn.g_proj,) if hasattr(attn, "g_proj") else (attn.g_down, attn.g_up)
                    )
                    for projection in (attn.f_down, attn.f_up, attn.beta_proj, *gate_projections):
                        torch.nn.init.kaiming_uniform_(projection.weight, a=math.sqrt(5))
                    gate_with_bias = attn.g_proj if hasattr(attn, "g_proj") else attn.g_up
                    fan_in = gate_with_bias.weight.size(1)
                    torch.nn.init.uniform_(gate_with_bias.bias, -(fan_in**-0.5), fan_in**-0.5)
                    if attn.variant == "kimi_k3":
                        attn.a_log.zero_()
                    else:
                        attn.a_log.uniform_(1, 16).log_()
                    attn.dt_bias.zero_()
                    attn.output_norm_weight.fill_(1.0)
                elif hasattr(attn, "output_gate"):
                    torch.nn.init.uniform_(attn.output_gate.weight, -s, s)
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, self.config.n_embd // self.config.n_head
        )
        self.cos, self.sin = cos, sin
        compute_dtype = self.config.runtime.compute_dtype
        if compute_dtype != torch.float16:
            self.transformer.wte.to(dtype=compute_dtype)
            for ve in self.value_embeds.values():
                ve.to(dtype=compute_dtype)

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds = sum(ve.weight.numel() for ve in self.value_embeds.values())
        excluded = (
            self.transformer.wte.weight.numel()
            + value_embeds
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.smear_gate.weight.numel()
            + self.smear_lambda.numel()
            + self.backout_lambda.numel()
        )
        excluded += sum(
            p.numel() for block in self.transformer.h for p in block.attn.parameters() if p.ndim < 2
        )
        matmul_and_conv = 6 * (nparams - excluded)
        head_dim = self.config.n_embd // self.config.n_head
        seq_len = self.config.sequence_len
        internal = 0
        for layer_idx in range(self.config.n_layer):
            if is_kda_layer(
                layer_idx,
                self.config.n_layer,
                self.config.kda_pattern,
                self.config.kda_force_final_global,
            ):
                internal += self.config.n_head * (
                    6 * head_dim**2
                    + 3 * self.config.kda_chunk_size * head_dim
                    + self.config.kda_chunk_size**2
                )
            else:
                internal += 12 * self.config.n_head * head_dim * seq_len
        return matmul_and_conv + internal

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = (
            self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.smear_gate.weight.numel()
            + self.smear_lambda.numel()
            + self.backout_lambda.numel()
        )
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters())
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def get_architecture_state(self) -> dict:
        return {
            "family": "kimi_kda",
            "variant": self.config.kda_variant,
            "pattern": self.config.kda_pattern,
            "layer_map": kda_layer_map(
                self.config.n_layer,
                self.config.kda_pattern,
                self.config.kda_force_final_global,
            ),
            "force_final_global": self.config.kda_force_final_global,
            "rope_policy": self.config.kda_rope_policy,
            "decay": "lower_bounded_gmin_-5"
            if self.config.kda_variant == "kimi_k3"
            else "softplus",
            "negative_eigenvalues": self.config.kda_variant == "solar_negative",
            "output_gate": "full_rank" if self.config.kda_variant != "kimi_linear" else "low_rank",
        }


# Kimi K3 block-attention residual architecture.
@dataclass
class KimiAttnResConfig(GPTConfig):
    arch_family: str = "kimi_attnres"
    # K3 uses 12 transformer layers/block at depth 93, i.e. about eight
    # blocks. Two layers/block preserves that granularity at d14.
    attn_res_block_size: int = 2
    attn_res_recompute: bool = True
    attn_res_variant: str = "kimi_k3_block_attnres"
    # H=1 is the original Kimi read. H=8 is Multi-Head Attention Residuals
    # (MHAR), which reshapes the same width-sized query without adding params.
    attn_res_heads: int = 1


class AttnResRead(nn.Module):
    """Single-pseudo-query softmax attention over residual sources."""

    def __init__(self, width: int, heads: int = 1, eps: float = 1e-6):
        super().__init__()
        if heads <= 0 or width % heads:
            raise ValueError("AttnRes width must be divisible by a positive head count")
        self.heads = int(heads)
        self.query = nn.Parameter(torch.zeros(width))
        self.norm_weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, *sources: torch.Tensor) -> torch.Tensor:
        if not sources:
            raise ValueError("AttnRes requires at least one residual source")
        shape = sources[0].shape
        if any(source.shape != shape for source in sources):
            raise ValueError("all AttnRes sources must share a shape")

        # Match K3's released reference: scores and aggregation are FP32, then
        # cast back to the residual-stream dtype. The source-wise loop avoids a
        # large [sources, batch, time, width] FP32 materialization.
        score_weight = self.norm_weight.float() * self.query.float()
        if self.heads > 1:
            head_dim = shape[-1] // self.heads
            score_weight = score_weight.view(self.heads, head_dim)
            scores = []
            values = []
            for source in sources:
                value = source.float()
                inv_rms = torch.rsqrt(value.square().mean(dim=-1) + self.eps)
                normalized = value * inv_rms.unsqueeze(-1)
                normalized = normalized.view(*shape[:-1], self.heads, head_dim)
                scores.append((normalized * score_weight).sum(dim=-1))
                values.append(value.view(*shape[:-1], self.heads, head_dim))
            probabilities = torch.softmax(torch.stack(scores, dim=-2), dim=-2)
            output = torch.zeros_like(values[0])
            for source_idx, value in enumerate(values):
                output = output + probabilities[..., source_idx, :, None] * value
            return output.reshape(shape).to(sources[0].dtype)

        scores = []
        for source in sources:
            value = source.float()
            inv_rms = torch.rsqrt(value.square().mean(dim=-1) + self.eps)
            scores.append((value * inv_rms.unsqueeze(-1) * score_weight).sum(dim=-1))
        probabilities = torch.softmax(torch.stack(scores, dim=-1), dim=-1)

        output = torch.zeros_like(sources[0], dtype=torch.float32)
        for source_idx, source in enumerate(sources):
            output = output + probabilities[..., source_idx, None] * source.float()
        return output.to(sources[0].dtype)


class KimiAttnRes(GPT):
    """nanochat d14 with the Kimi K3 Block AttnRes residual operator."""

    def __init__(self, config: KimiAttnResConfig, pad_vocab_size_to: int = 64):
        if config.attn_res_block_size <= 0:
            raise ValueError("attn_res_block_size must be positive")
        if config.attn_res_variant not in {"kimi_k3_block_attnres", "multi_head_attnres"}:
            raise ValueError("unsupported attention-residual variant")
        if config.attn_res_heads <= 0 or config.n_embd % config.attn_res_heads:
            raise ValueError("attention-residual heads must divide model width")
        if config.attn_res_variant == "kimi_k3_block_attnres" and config.attn_res_heads != 1:
            raise ValueError("Kimi K3 Block AttnRes uses one routing head")
        if config.attn_res_variant == "multi_head_attnres" and config.attn_res_heads <= 1:
            raise ValueError("MHAR requires more than one routing head")
        super().__init__(config, pad_vocab_size_to=pad_vocab_size_to)

        # These nanochat residual controls are alternative depth-mixing rules.
        # Keep non-persistent buffers so GPT.init_weights consumes the identical
        # shared RNG stream, but remove them from the trainable/state contract.
        del self.resid_lambdas
        del self.x0_lambdas
        del self.backout_lambda
        self.register_buffer("resid_lambdas", torch.empty(config.n_layer), persistent=False)
        self.register_buffer("x0_lambdas", torch.empty(config.n_layer), persistent=False)
        self.register_buffer("backout_lambda", torch.empty(1), persistent=False)

        self.attention_residual_reads = nn.ModuleList(
            [AttnResRead(config.n_embd, heads=config.attn_res_heads) for _ in range(config.n_layer)]
        )
        self.mlp_residual_reads = nn.ModuleList(
            [AttnResRead(config.n_embd, heads=config.attn_res_heads) for _ in range(config.n_layer)]
        )
        self.output_residual_read = AttnResRead(config.n_embd, heads=config.attn_res_heads)

    @torch.no_grad()
    def init_weights(self):
        # Shared tensors consume exactly the baseline initialization stream.
        GPT.init_weights(self)
        self.backout_lambda.zero_()
        for read in self._attn_res_reads():
            read.query.zero_()
            read.norm_weight.fill_(1.0)

    def _attn_res_reads(self):
        yield from self.attention_residual_reads
        yield from self.mlp_residual_reads
        yield self.output_residual_read

    def _apply_read(self, read: AttnResRead, sources: list[torch.Tensor]) -> torch.Tensor:
        if self.training and self.config.attn_res_recompute and torch.is_grad_enabled():
            return checkpoint(read, *sources, use_reentrant=False)
        return read(*sources)

    def residual_source_counts(self) -> dict[str, list[int] | int]:
        """Return the exact static source map used by this depth/block layout."""
        completed = 0
        pre_attention = []
        pre_mlp = []
        for layer_idx in range(self.config.n_layer):
            pre_attention.append(completed + 1)
            if layer_idx % self.config.attn_res_block_size == 0:
                completed += 1
            pre_mlp.append(completed + 1)
        return {
            "pre_attention": pre_attention,
            "pre_mlp": pre_mlp,
            "final": completed + 1,
        }

    def get_architecture_state(self) -> dict:
        counts = self.residual_source_counts()
        return {
            "family": "kimi_attnres",
            "variant": self.config.attn_res_variant,
            "routing_heads": self.config.attn_res_heads,
            "routing_head_dim": self.config.n_embd // self.config.attn_res_heads,
            "block_size_transformer_layers": self.config.attn_res_block_size,
            "completed_transformer_blocks": (
                self.config.n_layer + self.config.attn_res_block_size - 1
            )
            // self.config.attn_res_block_size,
            "final_source_count_including_embedding": counts["final"],
            "pre_attention_source_counts": counts["pre_attention"],
            "pre_mlp_source_counts": counts["pre_mlp"],
            "pseudo_query_initialization": "zeros",
            "score_normalization": "learned_rmsnorm_fp32",
            "aggregation_dtype": "float32",
            "activation_recomputation": self.config.attn_res_recompute,
            "standard_resid_lambdas": "replaced",
            "x0_lambdas": "replaced_by_embedding_source",
            "backout": "replaced_by_final_attnres_read",
        }

    def _attn_res_parameter_count(self) -> int:
        return sum(p.numel() for read in self._attn_res_reads() for p in read.parameters())

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        smear = self.smear_gate.weight.numel() + self.smear_lambda.numel()
        attn_res = self._attn_res_parameter_count()
        total = wte + value_embeds + lm_head + transformer_matrices + smear + attn_res
        assert total == sum(p.numel() for p in self.parameters()), "parameter count mismatch"
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "smear": smear,
            "attn_res": attn_res,
            "total": total,
        }

    def estimate_flops(self):
        # Forward+backward convention matches GPT: 6 FLOPs per matrix weight,
        # plus 12*h*q*context for attention. Each AttnRes source contributes
        # two d-dimensional dot-like reductions (score and weighted read), or
        # 12*d FLOPs including backward; norm/softmax elementwise FLOPs omitted.
        matrix_params = sum(p.numel() for p in self.transformer.h.parameters())
        matrix_params += sum(p.numel() for p in self.lm_head.parameters())
        h = self.config.n_head
        q = self.config.n_embd // h
        seq_len = self.config.sequence_len
        attention_flops = 0
        for window_size in self.window_sizes:
            effective_seq = min(window_size[0], seq_len) if window_size[0] >= 0 else seq_len
            attention_flops += 12 * h * q * effective_seq
        counts = self.residual_source_counts()
        source_reads = sum(counts["pre_attention"]) + sum(counts["pre_mlp"]) + counts["final"]
        attn_res_flops = 12 * self.config.n_embd * source_reads
        return 6 * matrix_params + attention_flops + attn_res_flops

    def estimate_executed_flops(self):
        return self.estimate_flops()

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        _, time = idx.size()
        if time > self.cos.size(1):
            raise ValueError(f"sequence length {time} exceeds rotary cache {self.cos.size(1)}")
        if idx.device != self.cos.device:
            raise ValueError("rotary embeddings and tokens are on different devices")
        compute_dtype = self.config.runtime.compute_dtype
        if self.cos.dtype != compute_dtype:
            raise ValueError(f"rotary embeddings must use {compute_dtype}, got {self.cos.dtype}")
        offset = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, offset : offset + time], self.sin[:, offset : offset + time]

        hidden = norm(self.transformer.wte(idx).to(compute_dtype))
        if kv_cache is None:
            if time <= 1:
                raise ValueError("training forward requires more than one token")
            gate = self.smear_lambda.to(hidden.dtype) * torch.sigmoid(
                self.smear_gate(hidden[:, 1:, : self.smear_gate_channels])
            )
            hidden = torch.cat([hidden[:, :1], hidden[:, 1:] + gate * hidden[:, :-1]], dim=1)
        else:
            previous = kv_cache.prev_embedding
            kv_cache.prev_embedding = hidden[:, -1:, :]
            if time > 1:
                gate = self.smear_lambda.to(hidden.dtype) * torch.sigmoid(
                    self.smear_gate(hidden[:, 1:, : self.smear_gate_channels])
                )
                hidden = torch.cat([hidden[:, :1], hidden[:, 1:] + gate * hidden[:, :-1]], dim=1)
            elif previous is not None:
                gate = self.smear_lambda.to(hidden.dtype) * torch.sigmoid(
                    self.smear_gate(hidden[:, :, : self.smear_gate_channels])
                )
                hidden = hidden + gate * previous

        completed_blocks: list[torch.Tensor] = []
        partial_block = hidden
        for layer_idx, block in enumerate(self.transformer.h):
            layer_input = self._apply_read(
                self.attention_residual_reads[layer_idx],
                completed_blocks + [partial_block],
            )
            if layer_idx % self.config.attn_res_block_size == 0:
                completed_blocks.append(partial_block)
                partial_block = None

            value_embedding = (
                self.value_embeds[str(layer_idx)](idx).to(layer_input.dtype)
                if str(layer_idx) in self.value_embeds
                else None
            )
            attention_output = block.attn(
                norm(layer_input), value_embedding, cos_sin, self.window_sizes[layer_idx], kv_cache
            )
            partial_block = (
                attention_output if partial_block is None else partial_block + attention_output
            )

            mlp_input = self._apply_read(
                self.mlp_residual_reads[layer_idx],
                completed_blocks + [partial_block],
            )
            partial_block = partial_block + block.mlp(norm(mlp_input))

        hidden = self._apply_read(self.output_residual_read, completed_blocks + [partial_block])
        hidden = norm(hidden)
        raw_logits = self.lm_head(hidden)[..., : self.config.vocab_size]
        logits = raw_logits.float()
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)
        if targets is None:
            return logits
        return language_model_loss(
            logits,
            raw_logits,
            targets,
            reduction=loss_reduction,
            z_loss_weight=self.config.z_loss_weight,
            training=self.training,
        )
