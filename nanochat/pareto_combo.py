"""Composable, initialization-aligned d14 Pareto-search models.

The search deliberately exposes only components that already won or approached
the deterministic d14 frontier in isolated three-seed controls.  Shared tensors
retain the paired nanochat initialization stream; component-only tensors use a
private deterministic stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, get_dist_info, print0
from nanochat.engram import EngramMemory
from nanochat.frontier_pool import (
    FrontierPoolGPT,
    MotifMHCConnection,
    QwenGatedDeltaAttention,
)
from nanochat.gpt import GPT, GPTConfig, Linear, has_ve, norm
from nanochat.optim import DistMuonAdamW, MuonAdamW
from nanochat.sota_pool import SotaAttention, XIELU


PARETO_COMPONENTS = frozenset(
    {"qwen_gdn", "exclusive_attention", "xielu", "motif_mhc_anneal", "engram"}
)


def canonical_components(value: str) -> tuple[str, ...]:
    components = tuple(sorted(part.strip() for part in value.split(",") if part.strip()))
    if len(components) < 2:
        raise ValueError("pareto_combo requires at least two components")
    if len(components) != len(set(components)):
        raise ValueError("pareto_components contains a duplicate")
    unknown = set(components) - PARETO_COMPONENTS
    if unknown:
        raise ValueError(f"unknown pareto components: {sorted(unknown)}")
    return components


@dataclass
class ParetoComboConfig(GPTConfig):
    arch_family: str = "pareto_combo"
    pareto_components: str = "qwen_gdn,xielu"
    frontier_extra_lr: float = 0.005
    sconv_kernel_size: int = 4
    mhc_num_streams: int = 4
    mhc_sinkhorn_iterations: int = 20
    mhc_anneal_steps: int = 1907
    engram_layers: tuple[int, ...] = (1, 6)
    engram_ngram_orders: tuple[int, ...] = (2, 3)
    engram_num_heads: int = 8
    engram_dim: int = 0
    engram_vocab_multiplier: int = 5
    engram_kernel_size: int = 4
    engram_seed: int = 0

    def __post_init__(self) -> None:
        components = canonical_components(self.pareto_components)
        self.pareto_components = ",".join(components)
        if self.frontier_extra_lr <= 0 or self.sconv_kernel_size <= 0:
            raise ValueError("invalid Pareto-combo optimizer or convolution setting")
        if self.mhc_num_streams <= 1 or self.mhc_sinkhorn_iterations <= 0 or self.mhc_anneal_steps <= 0:
            raise ValueError("invalid Pareto-combo mHC setting")
        if "engram" in components:
            if "motif_mhc_anneal" in components:
                raise ValueError("Engram and mHC stream replication are not a supported controlled pair")
            if not self.engram_layers or any(not 0 <= layer < self.n_layer for layer in self.engram_layers):
                raise ValueError("invalid Engram injection layers")
            if len(set(self.engram_layers)) != len(self.engram_layers):
                raise ValueError("Engram injection layers must be unique")
            if not self.engram_ngram_orders or min(self.engram_ngram_orders) < 1:
                raise ValueError("invalid Engram n-gram orders")

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.pareto_components.split(","))

    @property
    def sota_variant(self) -> str:
        return "exclusive_attention" if "exclusive_attention" in self.components else "baseline"

    @property
    def frontier_variant(self) -> str:
        # FrontierPoolGPT's generic accounting understands Qwen GDN.  The
        # baseline-like sentinel selects its standard dense estimate otherwise.
        return "qwen_gdn" if "qwen_gdn" in self.components else "inkling_lr2_weight_decay"


class ParetoMLP(nn.Module):
    def __init__(self, config: ParetoComboConfig):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.activation = XIELU() if "xielu" in config.components else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.c_fc(x)
        hidden = self.activation(hidden) if self.activation is not None else F.relu(hidden).square()
        return self.c_proj(hidden)


class ParetoBlock(nn.Module):
    def __init__(self, config: ParetoComboConfig, layer_idx: int):
        super().__init__()
        self.is_gdn = "qwen_gdn" in config.components and layer_idx % 4 != 3
        self.attn = (
            QwenGatedDeltaAttention(config, layer_idx)
            if self.is_gdn
            else SotaAttention(config, layer_idx)
        )
        self.mlp = ParetoMLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        return x + self.mlp(norm(x))


class ParetoComboGPT(FrontierPoolGPT):
    def __init__(self, config: ParetoComboConfig, pad_vocab_size_to: int = 64):
        GPT.__init__(self, config, pad_vocab_size_to=pad_vocab_size_to)
        self.config = config
        self.transformer.h = nn.ModuleList(
            [ParetoBlock(config, layer_idx) for layer_idx in range(config.n_layer)]
        )
        if "motif_mhc_anneal" in config.components:
            self.mhc_connections = nn.ModuleList(
                [MotifMHCConnection(config) for _ in range(2 * config.n_layer)]
            )
        self.engrams = nn.ModuleDict()
        if "engram" in config.components:
            memory_dim = config.engram_dim or config.n_embd // 2
            for layer_idx in config.engram_layers:
                self.engrams[str(layer_idx)] = EngramMemory(
                    hidden_size=config.n_embd,
                    memory_dim=memory_dim,
                    vocab_size=config.vocab_size,
                    vocab_multiplier=config.engram_vocab_multiplier,
                    num_heads=config.engram_num_heads,
                    ngram_orders=config.engram_ngram_orders,
                    layer_idx=layer_idx,
                    seed=config.engram_seed,
                    kernel_size=config.engram_kernel_size,
                    optimizer_world_size=8,
                )
            self.register_buffer("engram_token_map", torch.empty(config.vocab_size, dtype=torch.long))
            self.register_buffer("engram_pad_id", torch.empty((), dtype=torch.long))
        self._training_step = 0
        self._training_metrics = {}

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        width = self.config.n_embd
        bound = math.sqrt(3.0) * width**-0.5
        device = self.transformer.wte.weight.device

        # Match GPT.init_weights' layer-by-layer RNG consumption.  Replaced GDN
        # layers consume scratch baseline Q/K/V draws so every surviving shared
        # MLP and all later shared tensors remain bit-identical to the baseline.
        for block in self.transformer.h:
            if block.is_gdn:
                scratch = torch.empty((width, width), device=device)
                for _ in range(3):
                    torch.nn.init.uniform_(scratch, -bound, bound)
            else:
                for projection in (block.attn.c_q, block.attn.c_k, block.attn.c_v):
                    torch.nn.init.uniform_(projection.weight, -bound, bound)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -bound * 0.4, bound * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        for layer_idx in range(self.config.n_layer):
            self.resid_lambdas.data[layer_idx] = 1.15 - 0.10 * layer_idx / max(self.config.n_layer - 1, 1)
            self.x0_lambdas.data[layer_idx] = 0.20 - 0.15 * layer_idx / max(self.config.n_layer - 1, 1)
        for value_embed in self.value_embeds.values():
            torch.nn.init.uniform_(value_embed.weight, -bound, bound)
        for block in self.transformer.h:
            if block.is_gdn:
                if has_ve(block.attn.layer_idx, self.config.n_layer):
                    scratch = torch.empty((self.config.n_head, 12), device=device)
                    torch.nn.init.uniform_(scratch, 0.0, 0.02)
            elif block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        self.smear_lambda.zero_()
        self.backout_lambda.fill_(0.2)

        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        private_seed = torch.initial_seed() ^ 0x50434F4D
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(private_seed)
            for block in self.transformer.h:
                if block.is_gdn:
                    attn = block.attn
                    torch.nn.init.uniform_(attn.in_proj_qkvz.weight, -bound, bound)
                    torch.nn.init.uniform_(attn.in_proj_ba.weight, -bound, bound)
                    torch.nn.init.kaiming_uniform_(attn.conv.weight, a=math.sqrt(5))
                    attn.a_log.uniform_(0.01, 16.0).log_()
                    attn.dt_bias.fill_(1.0)
                    attn.output_norm_weight.fill_(1.0)
                    torch.nn.init.zeros_(attn.out_proj.weight)
                    if attn.ve_gate is not None:
                        torch.nn.init.uniform_(attn.ve_gate.weight, 0.0, 0.02)
                if block.mlp.activation is not None:
                    block.mlp.activation.reset_parameters()
            for connection in getattr(self, "mhc_connections", []):
                torch.nn.init.xavier_uniform_(connection.mapping_proj.weight)
                connection.alpha.fill_(0.01)
                connection.bias.zero_()

        if self.engrams:
            with torch.random.fork_rng(devices=devices):
                # Preserve the isolated Engram arm's component-private stream.
                torch.manual_seed(torch.initial_seed() ^ 0xE6A)
                for engram in self.engrams.values():
                    engram.init_weights()
            self.engram_token_map.copy_(
                torch.arange(self.config.vocab_size, device=self.engram_token_map.device)
            )
            self.engram_pad_id.zero_()

        self.cos, self.sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, self.config.n_embd // self.config.n_head
        )
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for value_embed in self.value_embeds.values():
                value_embed.to(dtype=COMPUTE_DTYPE)
            for engram in self.engrams.values():
                engram.embedding.to(dtype=COMPUTE_DTYPE)

    @torch.no_grad()
    def configure_engram_token_map(self, token_map: torch.Tensor, pad_id: int) -> None:
        if not self.engrams:
            return
        if token_map.shape != self.engram_token_map.shape:
            raise ValueError("compressed Engram token map has the wrong shape")
        self.engram_token_map.copy_(
            token_map.to(device=self.engram_token_map.device, dtype=torch.long)
        )
        self.engram_pad_id.copy_(self.engram_token_map[int(pad_id)])

    def _trunk(self, idx, kv_cache=None):
        _, time = idx.shape
        if time > self.cos.size(1):
            raise ValueError("sequence exceeds rotary cache")
        offset = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, offset : offset + time], self.sin[:, offset : offset + time]
        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        if kv_cache is None:
            if time <= 1:
                raise ValueError("training forward requires more than one token")
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            raise NotImplementedError("Pareto-combination KV-cache inference is outside this campaign")
        use_mhc = "motif_mhc_anneal" in self.config.components
        compressed_idx = self.engram_token_map[idx] if self.engrams else None
        if use_mhc:
            x = x.unsqueeze(-2).expand(-1, -1, self.config.mhc_num_streams, -1).contiguous()
        x0,backout=x,None
        for layer_idx,block in enumerate(self.transformer.h):
            x = self.resid_lambdas[layer_idx] * x + self.x0_lambdas[layer_idx] * x0
            if str(layer_idx) in self.engrams:
                x = x + self.engrams[str(layer_idx)](x, compressed_idx, self.engram_pad_id)
            ve = self.value_embeds[str(layer_idx)](idx).to(x.dtype) if str(layer_idx) in self.value_embeds else None
            if use_mhc:
                connection=self.mhc_connections[2*layer_idx]
                branch,post,residual=connection.prepare(x)
                output=block.attn(norm(branch),ve,cos_sin,self.window_sizes[layer_idx],None)
                x=connection.combine(x,output,post,residual)
                connection=self.mhc_connections[2*layer_idx+1]
                branch,post,residual=connection.prepare(x)
                output=block.mlp(norm(branch))
                x=connection.combine(x,output,post,residual)
            else:
                x=block(x,ve,cos_sin,self.window_sizes[layer_idx],None)
            if layer_idx==self.config.n_layer//2:
                backout=x
        if backout is not None:
            x=x-self.backout_lambda.to(x.dtype)*backout
        if use_mhc:
            x=x.mean(dim=-2)
        return norm(x),cos_sin

    def set_training_step(self, step: int) -> None:
        self._training_step = int(step)
        if "motif_mhc_anneal" not in self.config.components:
            return
        progress=min(max(step/self.config.mhc_anneal_steps,0.0),1.0)
        for connection in self.mhc_connections:
            connection.set_post_scale(2.0-progress)

    def estimate_flops(self):
        flops=FrontierPoolGPT.estimate_flops(self)
        if self.engrams:
            # Hash-table lookups are not dense matrix multiplies; GPT's generic
            # accounting otherwise counts the table rows as trainable matmuls.
            flops -= 6 * sum(engram.embedding.weight.numel() for engram in self.engrams.values())
        if "exclusive_attention" in self.config.components:
            dense_layers=sum(not block.is_gdn for block in self.transformer.h)
            flops += 12 * dense_layers * self.config.n_embd
        return flops

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        if not self.engrams:
            return super().setup_optimizer(
                unembedding_lr=unembedding_lr,
                embedding_lr=embedding_lr,
                matrix_lr=matrix_lr,
                weight_decay=weight_decay,
                scalar_lr=scalar_lr,
            )
        ddp, _, _, _ = get_dist_info()
        scale = (self.config.n_embd / 768) ** -0.5
        print0(f"Scaling the LR for AdamW parameters by {scale:.6f}")
        engram_embeddings = [engram.embedding.weight for engram in self.engrams.values()]
        engram_convs = [engram.short_conv.weight for engram in self.engrams.values()]
        excluded = {
            id(self.transformer.wte.weight), id(self.lm_head.weight), id(self.resid_lambdas),
            id(self.x0_lambdas), id(self.smear_gate.weight), id(self.smear_lambda),
            id(self.backout_lambda),
            *(id(parameter) for parameter in self.value_embeds.parameters()),
            *(id(parameter) for parameter in engram_embeddings),
            *(id(parameter) for parameter in engram_convs),
        }
        matrices, architecture = [], []
        for parameter in self.parameters():
            if id(parameter) in excluded:
                continue
            (matrices if parameter.ndim == 2 else architecture).append(parameter)
        controls = [self.resid_lambdas, self.x0_lambdas, self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        groups = [
            dict(kind="adamw", params=[self.lm_head.weight], lr=unembedding_lr * scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=[self.transformer.wte.weight], lr=embedding_lr * scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=list(self.value_embeds.parameters()), lr=embedding_lr * scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=[self.resid_lambdas], lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind="adamw", params=[self.x0_lambdas], lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=controls[2:], lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=engram_embeddings, lr=matrix_lr * 5.0, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=engram_convs, lr=matrix_lr * 5.0, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if architecture:
            groups.append(dict(kind="adamw", params=architecture, lr=self.config.frontier_extra_lr, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        for shape in sorted({parameter.shape for parameter in matrices}):
            groups.append(dict(kind="muon", params=[parameter for parameter in matrices if parameter.shape == shape], lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay))
        grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
        if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != {id(parameter) for parameter in self.parameters()}:
            raise RuntimeError("Pareto Engram optimizer grouping mismatch")
        optimizer = (DistMuonAdamW if ddp else MuonAdamW)(groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def get_architecture_state(self) -> dict:
        return {
            "family":"pareto_combo",
            "components":list(self.config.components),
            "shared_backbone":"deterministic_nanochat_d14",
            "shared_initialization":"bit_identical_where_shape_compatible",
            "variant_only_rng":"forked_seed_xor_0x50434f4d",
            "gdn_layer_pattern":"GGGF" if "qwen_gdn" in self.config.components else None,
            "exclusive_attention_layers":[i+1 for i,b in enumerate(self.transformer.h) if not b.is_gdn] if "exclusive_attention" in self.config.components else [],
            "xielu_layers":self.config.n_layer if "xielu" in self.config.components else 0,
            "mhc_streams":self.config.mhc_num_streams if "motif_mhc_anneal" in self.config.components else 0,
            "engram_layers":list(self.config.engram_layers) if self.engrams else [],
            "engram_ngram_orders":list(self.config.engram_ngram_orders) if self.engrams else [],
            "engram_num_heads":self.config.engram_num_heads if self.engrams else 0,
            "current_step":self._training_step,
        }
