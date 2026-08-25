"""Model-family construction helpers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import fields

import torch

from next_gen_arch.arch.base import GPT, ArchitectureRuntime, GPTConfig
from next_gen_arch.arch.combinations import (
    ComboSearchConfig,
    ComboSearchGPT,
    ParetoComboConfig,
    ParetoComboGPT,
)
from next_gen_arch.arch.dsa import DeepSeekDSA, DeepSeekDSAConfig
from next_gen_arch.arch.fog import FOG, FOGConfig
from next_gen_arch.arch.frontier import FrontierPoolConfig, FrontierPoolGPT
from next_gen_arch.arch.kimi import KimiAttnRes, KimiAttnResConfig, KimiKDA, KimiKDAConfig
from next_gen_arch.arch.sota import SotaPoolConfig, SotaPoolGPT


def training_architecture_runtime() -> ArchitectureRuntime:
    """Bind pure architecture definitions to this trainer's execution ops."""
    from next_gen_arch.training.attention import flash_attn
    from next_gen_arch.training.runtime import COMPUTE_DTYPE, print0

    return ArchitectureRuntime(
        compute_dtype=COMPUTE_DTYPE,
        attention=flash_attn,
        log=print0,
    )


def build_engram_token_map(tokenizer, vocab_size: int) -> tuple[torch.Tensor, int]:
    """Build the fixed tokenizer-normalization map used by Engram arms."""
    mapping: list[int] = []
    key_to_id: dict[str, int] = {}
    whitespace = re.compile(r"[ \t\r\n]+")
    for token_id in range(vocab_size):
        text = tokenizer.id_to_token(token_id)
        if "�" in text:
            key = f"<raw-token-{token_id}>"
        else:
            key = unicodedata.normalize("NFKC", text)
            key = unicodedata.normalize("NFD", key)
            key = "".join(ch for ch in key if unicodedata.category(ch) != "Mn")
            key = whitespace.sub(" ", key.lower())
            key = " " if key == " " else key.strip()
            if not key:
                key = text
        mapping.append(key_to_id.setdefault(key, len(key_to_id)))
    return torch.tensor(mapping, dtype=torch.long), len(key_to_id)


def infer_model_dims(depth: int, aspect_ratio: int, head_dim: int) -> tuple[int, int]:
    """Infer model width/head count from the repo's depth-driven scaling rule."""
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return model_dim, num_heads


def patch_model_config_kwargs(model_config_kwargs: dict) -> dict:
    """Patch missing config keys from older checkpoints and normalize families."""
    arch_family = model_config_kwargs.get("arch_family", "nanochat")
    patched = dict(model_config_kwargs)
    patched.setdefault("runtime", training_architecture_runtime())
    patched["arch_family"] = arch_family
    patched.setdefault("per_head_muon", False)
    if "window_pattern" not in patched:
        patched["window_pattern"] = "L"
    if arch_family == "engram":
        patched.setdefault("engram_layers", (1, 6))
        patched["engram_layers"] = tuple(patched["engram_layers"])
        patched.setdefault("engram_ngram_orders", (2, 3))
        patched["engram_ngram_orders"] = tuple(patched["engram_ngram_orders"])
        patched.setdefault("engram_num_heads", 8)
        patched.setdefault("engram_dim", 0)
        patched.setdefault("engram_vocab_multiplier", 5)
        patched.setdefault("engram_kernel_size", 4)
        patched.setdefault("engram_seed", 0)
    if arch_family == "mhc":
        patched.setdefault("mhc_num_streams", 4)
        patched.setdefault("mhc_init_gating_factor", 0.01)
        patched.setdefault("mhc_sinkhorn_iterations", 20)
    if arch_family == "fog":
        patched.setdefault("fog_variant", "flash")
    if arch_family == "kimi_kda":
        patched.setdefault("kda_pattern", "KKKG")
        patched.setdefault("kda_chunk_size", 64)
        patched.setdefault("kda_conv_size", 4)
        patched.setdefault("kda_rope_policy", "global_only")
        patched.setdefault("kda_variant", "kimi_linear")
        patched.setdefault("kda_force_final_global", True)
    if arch_family == "kimi_attnres":
        patched.setdefault("attn_res_block_size", 2)
        patched.setdefault("attn_res_recompute", True)
        patched.setdefault("attn_res_variant", "kimi_k3_block_attnres")
        patched.setdefault("attn_res_heads", 1)
    if arch_family == "deepseek_dsa":
        patched.setdefault("dsa_top_k", 32)
        patched.setdefault("dsa_index_heads", 4)
        patched.setdefault("dsa_index_head_dim", 128)
        patched.setdefault("dsa_index_rope_dim", 64)
        patched.setdefault("dsa_dense_warmup_steps", 40)
        patched.setdefault("dsa_query_chunk_size", 128)
        patched.setdefault("dsa_backend", "sdpa_masked")
        patched.setdefault("dsa_warmup_indexer_lr", 1e-3)
        patched.setdefault("dsa_sparse_indexer_lr", 7.3e-6)
    if arch_family == "combo_search":
        patched.setdefault("search_mlp", "baseline")
        patched.setdefault("gated_mlp_width", -1)
        patched.setdefault("sparser_l1_coeff", 0.0)
        patched.setdefault("colu_dim", 4)
        patched.setdefault("colu_scaling", "soft")
        patched.setdefault("qat_recipe", "none")
        patched.setdefault("qat_group_size", 128)
        patched.setdefault("qat_start_step", 0)
        patched.setdefault("qat_min_size", 128)
    if arch_family == "sota_pool":
        patched.setdefault("sota_variant", "baseline")
        patched.setdefault("sota_extra_lr", 0.005)
        patched.setdefault("canon_kernel_size", 4)
        patched.setdefault("bov_target_fraction", 1.0 / 3.0)
    if arch_family == "frontier_pool":
        patched.setdefault("frontier_variant", "inkling_relative_attention")
        patched.setdefault("frontier_extra_lr", 0.005)
        patched.setdefault("relative_dim", 16)
        patched.setdefault("relative_extent", 1024)
        patched.setdefault("sconv_kernel_size", 4)
        patched.setdefault("mtp_depth", 3)
        patched.setdefault("mtp_loss_weight", 0.1)
    if arch_family == "pareto_combo":
        patched.setdefault("pareto_components", "qwen_gdn,xielu")
        patched.setdefault("frontier_extra_lr", 0.005)
        patched.setdefault("sconv_kernel_size", 4)
        patched.setdefault("mhc_num_streams", 4)
        patched.setdefault("mhc_sinkhorn_iterations", 20)
        patched.setdefault("mhc_anneal_steps", 1907)
    return patched


def build_model_config(
    *,
    arch_family: str,
    depth: int,
    aspect_ratio: int,
    head_dim: int,
    max_seq_len: int,
    vocab_size: int,
    window_pattern: str,
    fog_variant: str,
    per_head_muon: bool = False,
    kda_pattern: str = "KKKG",
    kda_rope_policy: str = "global_only",
    kda_variant: str = "kimi_linear",
    kda_force_final_global: bool = True,
    dsa_top_k: int = 32,
    dsa_index_heads: int = 4,
    dsa_index_head_dim: int = 128,
    dsa_index_rope_dim: int = 64,
    dsa_dense_warmup_steps: int = 40,
    dsa_query_chunk_size: int = 128,
    dsa_backend: str = "sdpa_masked",
    dsa_warmup_indexer_lr: float = 1e-3,
    dsa_sparse_indexer_lr: float = 7.3e-6,
    attn_res_block_size: int = 2,
    attn_res_recompute: bool = True,
    attn_res_variant: str = "kimi_k3_block_attnres",
    attn_res_heads: int = 1,
    search_mlp: str = "baseline",
    gated_mlp_width: int = -1,
    sparser_l1_coeff: float = 0.0,
    colu_dim: int = 4,
    qat_recipe: str = "none",
    qat_group_size: int = 128,
    qat_start_step: int = 0,
    qat_min_size: int = 128,
    sota_variant: str = "baseline",
    sota_extra_lr: float = 0.005,
    canon_kernel_size: int = 4,
    bov_target_fraction: float = 1.0 / 3.0,
    frontier_variant: str = "inkling_relative_attention",
    frontier_extra_lr: float = 0.005,
    relative_dim: int = 16,
    relative_extent: int = 1024,
    sconv_kernel_size: int = 4,
    mtp_depth: int = 3,
    mtp_loss_weight: float = 0.1,
    pareto_components: str = "qwen_gdn,xielu",
    engram_layers: tuple[int, ...] = (1, 6),
    engram_ngram_orders: tuple[int, ...] = (2, 3),
    engram_num_heads: int = 8,
    engram_dim: int = 0,
    engram_vocab_multiplier: int = 5,
    engram_kernel_size: int = 4,
    engram_seed: int = 0,
    mhc_num_streams: int = 4,
    mhc_init_gating_factor: float = 0.01,
    mhc_sinkhorn_iterations: int = 20,
):
    model_dim, num_heads = infer_model_dims(depth, aspect_ratio, head_dim)
    common_kwargs = dict(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
        per_head_muon=per_head_muon,
        runtime=training_architecture_runtime(),
    )
    if arch_family == "nanochat":
        return GPTConfig(**common_kwargs, arch_family="nanochat")
    if arch_family == "engram":
        return GPTConfig(
            **common_kwargs,
            arch_family="engram",
            engram_layers=tuple(engram_layers),
            engram_ngram_orders=tuple(engram_ngram_orders),
            engram_num_heads=engram_num_heads,
            engram_dim=engram_dim,
            engram_vocab_multiplier=engram_vocab_multiplier,
            engram_kernel_size=engram_kernel_size,
            engram_seed=engram_seed,
        )
    if arch_family == "mhc":
        return GPTConfig(
            **common_kwargs,
            arch_family="mhc",
            mhc_num_streams=mhc_num_streams,
            mhc_init_gating_factor=mhc_init_gating_factor,
            mhc_sinkhorn_iterations=mhc_sinkhorn_iterations,
        )
    if arch_family == "fog":
        return FOGConfig(**common_kwargs, arch_family="fog", fog_variant=fog_variant)
    if arch_family == "kimi_kda":
        return KimiKDAConfig(
            **common_kwargs,
            arch_family="kimi_kda",
            kda_pattern=kda_pattern,
            kda_rope_policy=kda_rope_policy,
            kda_variant=kda_variant,
            kda_force_final_global=kda_force_final_global,
        )
    if arch_family == "kimi_attnres":
        return KimiAttnResConfig(
            **common_kwargs,
            arch_family="kimi_attnres",
            attn_res_block_size=attn_res_block_size,
            attn_res_recompute=attn_res_recompute,
            attn_res_variant=attn_res_variant,
            attn_res_heads=attn_res_heads,
        )
    if arch_family == "deepseek_dsa":
        return DeepSeekDSAConfig(
            **common_kwargs,
            arch_family="deepseek_dsa",
            dsa_top_k=dsa_top_k,
            dsa_index_heads=dsa_index_heads,
            dsa_index_head_dim=dsa_index_head_dim,
            dsa_index_rope_dim=dsa_index_rope_dim,
            dsa_dense_warmup_steps=dsa_dense_warmup_steps,
            dsa_query_chunk_size=dsa_query_chunk_size,
            dsa_backend=dsa_backend,
            dsa_warmup_indexer_lr=dsa_warmup_indexer_lr,
            dsa_sparse_indexer_lr=dsa_sparse_indexer_lr,
        )
    if arch_family == "combo_search":
        return ComboSearchConfig(
            **common_kwargs,
            arch_family="combo_search",
            search_mlp=search_mlp,
            gated_mlp_width=gated_mlp_width,
            sparser_l1_coeff=sparser_l1_coeff,
            colu_dim=colu_dim,
            colu_scaling="soft",
            qat_recipe=qat_recipe,
            qat_group_size=qat_group_size,
            qat_start_step=qat_start_step,
            qat_min_size=qat_min_size,
        )
    if arch_family == "sota_pool":
        return SotaPoolConfig(
            **common_kwargs,
            arch_family="sota_pool",
            sota_variant=sota_variant,
            sota_extra_lr=sota_extra_lr,
            canon_kernel_size=canon_kernel_size,
            bov_target_fraction=bov_target_fraction,
        )
    if arch_family == "frontier_pool":
        return FrontierPoolConfig(
            **common_kwargs,
            arch_family="frontier_pool",
            frontier_variant=frontier_variant,
            frontier_extra_lr=frontier_extra_lr,
            relative_dim=relative_dim,
            relative_extent=relative_extent,
            sconv_kernel_size=sconv_kernel_size,
            mtp_depth=mtp_depth,
            mtp_loss_weight=mtp_loss_weight,
        )
    if arch_family == "pareto_combo":
        return ParetoComboConfig(
            **common_kwargs,
            arch_family="pareto_combo",
            pareto_components=pareto_components,
            frontier_extra_lr=frontier_extra_lr,
            sconv_kernel_size=sconv_kernel_size,
        )
    raise ValueError(f"Unsupported arch_family: {arch_family}")


def instantiate_model(model_config, *, runtime_backend: str = "native"):
    """Build a model instance from a config dataclass."""
    arch_family = getattr(model_config, "arch_family", "nanochat")
    if arch_family in {"nanochat", "engram", "mhc"}:
        if runtime_backend != "native":
            raise ValueError("nanochat GPT variants only support the native runtime backend")
        return GPT(model_config)
    if arch_family == "fog":
        return FOG(model_config, runtime_backend=runtime_backend)
    if arch_family == "kimi_kda":
        if runtime_backend != "native":
            raise ValueError("Kimi KDA only supports the native runtime backend")
        return KimiKDA(model_config)
    if arch_family == "kimi_attnres":
        if runtime_backend != "native":
            raise ValueError("Kimi AttnRes only supports the native runtime backend")
        return KimiAttnRes(model_config)
    if arch_family == "deepseek_dsa":
        if runtime_backend != "native":
            raise ValueError("DeepSeek DSA only supports the native runtime backend")
        return DeepSeekDSA(model_config)
    if arch_family == "combo_search":
        if runtime_backend != "native":
            raise ValueError("Combo search GPT only supports the native runtime backend")
        return ComboSearchGPT(model_config)
    if arch_family == "sota_pool":
        if runtime_backend != "native":
            raise ValueError("SoTA pool GPT only supports the native runtime backend")
        return SotaPoolGPT(model_config)
    if arch_family == "frontier_pool":
        if runtime_backend != "native":
            raise ValueError("Frontier pool GPT only supports the native runtime backend")
        return FrontierPoolGPT(model_config)
    if arch_family == "pareto_combo":
        if runtime_backend != "native":
            raise ValueError("Pareto combo GPT only supports the native runtime backend")
        return ParetoComboGPT(model_config)
    raise ValueError(f"Unsupported arch_family: {arch_family}")


def build_model_from_config_kwargs(model_config_kwargs: dict, *, runtime_backend: str = "native"):
    patched = patch_model_config_kwargs(model_config_kwargs)
    arch_family = patched["arch_family"]
    if arch_family in {"nanochat", "engram", "mhc"}:
        config = GPTConfig(**patched)
    elif arch_family == "fog":
        config = FOGConfig(**patched)
    elif arch_family == "kimi_kda":
        config = KimiKDAConfig(**patched)
    elif arch_family == "kimi_attnres":
        config = KimiAttnResConfig(**patched)
    elif arch_family == "deepseek_dsa":
        config = DeepSeekDSAConfig(**patched)
    elif arch_family == "combo_search":
        config = ComboSearchConfig(**patched)
    elif arch_family == "sota_pool":
        config = SotaPoolConfig(**patched)
    elif arch_family == "frontier_pool":
        config = FrontierPoolConfig(**patched)
    elif arch_family == "pareto_combo":
        config = ParetoComboConfig(**patched)
    else:
        raise ValueError(f"Unsupported arch_family: {arch_family}")
    model = instantiate_model(config, runtime_backend=runtime_backend)
    return model, config


def model_config_to_dict(model_config) -> dict:
    return {
        config_field.name: getattr(model_config, config_field.name)
        for config_field in fields(model_config)
        if config_field.name != "runtime"
    }


def strip_backend_extra_state(model_state_dict: dict) -> dict:
    """Drop backend-only state that native eval models do not need."""
    return {k: v for k, v in model_state_dict.items() if not k.endswith("._extra_state")}


def patch_missing_model_state(model_data: dict, model_config) -> dict:
    """Patch old checkpoint parameter sets for backwards compatibility."""
    patched = dict(model_data)
    arch_family = getattr(model_config, "arch_family", "nanochat")
    if arch_family in {"nanochat", "engram", "mhc"}:
        n_layer = model_config.n_layer
        if "resid_lambdas" not in patched:
            patched["resid_lambdas"] = torch.ones(n_layer)
        if "x0_lambdas" not in patched:
            patched["x0_lambdas"] = torch.zeros(n_layer)
    return patched
