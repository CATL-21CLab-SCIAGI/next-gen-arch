"""FOG-style transformer family for controlled low-precision comparisons."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from archlab.architectures.base import ArchitectureRuntime, apply_rotary_emb


@dataclass
class FOGConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "L"
    arch_family: str = "fog"
    fog_variant: str = "flash"
    runtime: ArchitectureRuntime = field(
        default_factory=ArchitectureRuntime,
        repr=False,
        compare=False,
    )


class NativeLinear(nn.Linear):
    """Mirror nanochat's explicit mixed-precision linear."""

    def forward(self, x):
        return F.linear(
            x,
            self.weight.to(dtype=x.dtype),
            None if self.bias is None else self.bias.to(dtype=x.dtype),
        )


class GainRMSNorm(nn.Module):
    def __init__(self, dim: int, init_gain: float):
        super().__init__()
        self.weight = nn.Parameter(torch.full((dim,), init_gain))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight.to(dtype=x.dtype))


def frozen_rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


def kurtosis_metric(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    xf = x.float().reshape(-1)
    x2 = xf.square()
    return xf.pow(4).mean() / x2.var(unbiased=False).clamp_min(eps)


class QuantMonitor:
    def __init__(self):
        self.enabled = False
        self.records: dict[str, list[float]] = defaultdict(list)

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if not enabled:
            self.records.clear()

    def record(self, name: str, x: torch.Tensor):
        if not self.enabled:
            return
        x_float = x.float()
        self.records[f"{name}/kurtosis"].append(float(kurtosis_metric(x_float).item()))
        self.records[f"{name}/amax"].append(float(x_float.abs().max().item()))
        self.records[f"{name}/nonfinite_frac"].append(
            float((~torch.isfinite(x_float)).float().mean().item())
        )

    def consume(self) -> dict[str, float]:
        out = {key: sum(values) / len(values) for key, values in self.records.items() if values}
        self.records.clear()
        return out


def _build_linear(in_features: int, out_features: int, *, runtime_backend: str):
    if runtime_backend == "native":
        return NativeLinear(in_features, out_features, bias=False)
    if runtime_backend.startswith("te_"):
        import transformer_engine.pytorch as te

        return te.Linear(in_features, out_features, bias=False, params_dtype=torch.float32)
    raise ValueError(f"Unsupported runtime backend: {runtime_backend}")


class FogSelfAttention(nn.Module):
    def __init__(self, config: FOGConfig, layer_idx: int, *, runtime_backend: str):
        super().__init__()
        self.layer_idx = layer_idx
        self.variant = config.fog_variant
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.sequence_len = config.sequence_len
        self.runtime = config.runtime
        self.runtime_backend = runtime_backend
        assert self.n_head == self.n_kv_head, "FOG runtime currently assumes n_head == n_kv_head"
        self.c_q = _build_linear(
            self.n_embd, self.n_head * self.head_dim, runtime_backend=runtime_backend
        )
        self.c_k = _build_linear(
            self.n_embd, self.n_kv_head * self.head_dim, runtime_backend=runtime_backend
        )
        self.c_v = _build_linear(
            self.n_embd, self.n_kv_head * self.head_dim, runtime_backend=runtime_backend
        )
        self.c_proj = _build_linear(self.n_embd, self.n_embd, runtime_backend=runtime_backend)
        self.te_attention = None
        if runtime_backend.startswith("te_"):
            import transformer_engine.pytorch as te

            self.te_attention = te.DotProductAttention(
                num_attention_heads=self.n_head,
                kv_channels=self.head_dim,
                attention_dropout=0.0,
                attn_mask_type="causal",
                qkv_format="bshd",
            )

    def _regularize_qk(self, q: torch.Tensor, k: torch.Tensor):
        if self.variant == "opt":
            return frozen_rms_norm(q), frozen_rms_norm(k)
        if self.variant == "flash":
            return torch.tanh(q), torch.tanh(k)
        raise ValueError(f"Unsupported fog_variant: {self.variant}")

    def _use_te_attention(self, kv_cache, window_size) -> bool:
        if self.te_attention is None or kv_cache is not None:
            return False
        left_window = window_size[0]
        return left_window < 0 or left_window >= self.sequence_len

    def forward(self, x, cos_sin, window_size, kv_cache, quant_monitor: QuantMonitor | None = None):
        bsz, seq_len, _ = x.size()
        q = self.c_q(x).view(bsz, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(bsz, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(bsz, seq_len, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = self._regularize_qk(q, k)
        if quant_monitor is not None:
            qkv = torch.cat(
                [
                    q.reshape(bsz, seq_len, -1),
                    k.reshape(bsz, seq_len, -1),
                    v.reshape(bsz, seq_len, -1),
                ],
                dim=-1,
            )
            quant_monitor.record("quant/qkv", qkv)
        if (
            self.te_attention is not None
            and kv_cache is None
            and not self._use_te_attention(kv_cache, window_size)
        ):
            raise RuntimeError(
                "FOG Transformer Engine training path requires full-context attention (window_pattern=L). "
                f"Layer {self.layer_idx} received window_size={window_size}."
            )

        if self._use_te_attention(kv_cache, window_size):
            y = self.te_attention(q, k, v)
        elif kv_cache is None:
            y = self.runtime.attention.flash_attn_func(
                q, k, v, causal=True, window_size=window_size
            )
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = self.runtime.attention.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(seq_len)
        y = y.contiguous().view(bsz, seq_len, -1)
        return self.c_proj(y)


class FogMLP(nn.Module):
    def __init__(self, config: FOGConfig, *, runtime_backend: str):
        super().__init__()
        self.c_fc = _build_linear(config.n_embd, 4 * config.n_embd, runtime_backend=runtime_backend)
        self.c_proj = _build_linear(
            4 * config.n_embd, config.n_embd, runtime_backend=runtime_backend
        )

    def forward(self, x, quant_monitor: QuantMonitor | None = None):
        x = self.c_fc(x)
        x = F.gelu(x)
        if quant_monitor is not None:
            quant_monitor.record("quant/ffn_inner", x)
        x = self.c_proj(x)
        return x


class FogBlock(nn.Module):
    def __init__(self, config: FOGConfig, layer_idx: int, *, runtime_backend: str):
        super().__init__()
        init_gain = 1.0 / math.sqrt(config.n_layer)
        self.attn = FogSelfAttention(config, layer_idx, runtime_backend=runtime_backend)
        self.attn_postnorm = GainRMSNorm(config.n_embd, init_gain=init_gain)
        self.mlp = FogMLP(config, runtime_backend=runtime_backend)
        self.mlp_postnorm = GainRMSNorm(config.n_embd, init_gain=init_gain)

    def forward(self, x, cos_sin, window_size, kv_cache, quant_monitor: QuantMonitor | None = None):
        x = x + self.attn_postnorm(
            self.attn(x, cos_sin, window_size, kv_cache, quant_monitor=quant_monitor)
        )
        x = x + self.mlp_postnorm(self.mlp(x, quant_monitor=quant_monitor))
        if quant_monitor is not None:
            quant_monitor.record("quant/block_output", x)
        return x


class FOG(nn.Module):
    def __init__(
        self, config: FOGConfig, runtime_backend: str = "native", pad_vocab_size_to: int = 64
    ):
        super().__init__()
        self.config = config
        self.runtime_backend = runtime_backend
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            config.runtime.log(
                f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency"
            )
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [
                        FogBlock(config, layer_idx, runtime_backend=runtime_backend)
                        for layer_idx in range(config.n_layer)
                    ]
                ),
            }
        )
        self.lm_head = _build_linear(
            config.n_embd, padded_vocab_size, runtime_backend=runtime_backend
        )
        self.init_std = config.n_embd**-0.5
        self.input_scale = 1.0 / self.init_std
        self.quant_monitor = QuantMonitor()
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, config.n_embd // config.n_head
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        sigma = self.init_std
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=sigma)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=sigma)
        for block in self.transformer.h:
            for module in (
                block.attn.c_q,
                block.attn.c_k,
                block.attn.c_v,
                block.attn.c_proj,
                block.mlp.c_fc,
                block.mlp.c_proj,
            ):
                torch.nn.init.normal_(module.weight, mean=0.0, std=sigma)
            torch.nn.init.constant_(
                block.attn_postnorm.weight, 1.0 / math.sqrt(self.config.n_layer)
            )
            torch.nn.init.constant_(block.mlp_postnorm.weight, 1.0 / math.sqrt(self.config.n_layer))
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.config.runtime.compute_dtype != torch.float16:
            self.transformer.wte.to(dtype=self.config.runtime.compute_dtype)

    def set_quant_monitor_enabled(self, enabled: bool):
        self.quant_monitor.set_enabled(enabled)

    def consume_quant_metrics(self) -> dict[str, float]:
        return self.quant_monitor.consume()

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        compute_dtype = self.config.runtime.compute_dtype
        cos, sin = cos.to(compute_dtype), sin.to(compute_dtype)
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), (
            f"Invalid window_pattern: {pattern}. Use only S and L."
        )
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [
            char_to_window[pattern[layer_idx % len(pattern)]] for layer_idx in range(config.n_layer)
        ]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        norm_params = sum(
            block.attn_postnorm.weight.numel() + block.mlp_postnorm.weight.numel()
            for block in self.transformer.h
        )
        nparams_exclude = self.transformer.wte.weight.numel() + norm_params
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = 0
        norms = 0
        for block in self.transformer.h:
            transformer_matrices += sum(p.numel() for p in block.attn.parameters())
            transformer_matrices += sum(p.numel() for p in block.mlp.parameters())
            norms += block.attn_postnorm.weight.numel() + block.mlp_postnorm.weight.numel()
        total = wte + lm_head + transformer_matrices + norms
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            "wte": wte,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "norms": norms,
            "total": total,
        }

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        bsz, seq_len = idx.size()
        assert seq_len <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: {seq_len} > {self.cos.size(1)}"
        )
        assert idx.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        )
        compute_dtype = self.config.runtime.compute_dtype
        assert self.cos.dtype == compute_dtype, (
            f"Rotary embeddings must be in {compute_dtype}, got {self.cos.dtype}"
        )
        start = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, start : start + seq_len], self.sin[:, start : start + seq_len]
        x = self.transformer.wte(idx).to(compute_dtype)
        x = x * self.input_scale
        for i, block in enumerate(self.transformer.h):
            x = block(x, cos_sin, self.window_sizes[i], kv_cache, quant_monitor=self.quant_monitor)
        logits = self.lm_head(x)[..., : self.config.vocab_size]
        logits = logits.float()
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()
