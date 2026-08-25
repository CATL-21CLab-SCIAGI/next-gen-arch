"""
Train model. From root directory of the project, run as:

python -m next_gen_arch.training.base_train

or distributed as:

torchrun --nproc_per_node=8 -m next_gen_arch.training.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m next_gen_arch.training.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
import re
import unicodedata
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from next_gen_arch.arch.base import Linear
from next_gen_arch.training.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from next_gen_arch.training.runtime import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from next_gen_arch.training.tokenizer import get_tokenizer, get_token_bytes
from next_gen_arch.training.checkpoint import save_checkpoint, load_checkpoint
from next_gen_arch.training.loss_eval import evaluate_bpb
from next_gen_arch.training.engine import Engine
from next_gen_arch.training.attention import ATTENTION_BACKEND, ATTENTION_BACKEND_REASON, HAS_FLASH_ATTENTION, describe_attention_backend
from next_gen_arch.training.models import build_model_config, build_model_from_config_kwargs, instantiate_model, model_config_to_dict
from next_gen_arch.training.precision import (
    is_full_context_window_pattern,
    precision_recipe_requires_full_context_window,
    resolve_precision_backend,
)
from next_gen_arch.prompts import load_prompt_texts
from next_gen_arch.training.base_eval import evaluate_core
print_banner()


def comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--seed", type=int, default=42, help="global training seed used for model init and matched comparison runs")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
parser.add_argument("--precision-recipe", type=str, default="bf16", choices=["bf16", "fp8_full", "fp4_blackwell"], help="controlled precision recipe for the FOG family")
parser.add_argument("--stochastic-rounding", type=str, default="auto", choices=["auto", "on", "off"], help="low-precision stochastic-rounding policy when exposed by the backend")
parser.add_argument("--split-accumulator", type=str, default="auto", choices=["auto", "split", "fast"], help="low-precision split-accumulator policy when exposed by the backend")
# Model architecture
parser.add_argument("--arch-family", type=str, default="nanochat", choices=["nanochat", "engram", "mhc", "fog", "kimi_kda", "kimi_attnres", "deepseek_dsa", "combo_search", "sota_pool", "frontier_pool", "pareto_combo"], help="model family to train")
parser.add_argument("--fog-variant", type=str, default="flash", choices=["flash", "opt"], help="FOG attention regularization variant")
parser.add_argument("--kda-pattern", type=str, default="KKKG", help="repeating KDA/global layer pattern")
parser.add_argument("--kda-rope-policy", type=str, default="global_only", choices=["global_only", "none"], help="positional policy in global layers")
parser.add_argument("--kda-variant", type=str, default="kimi_linear", choices=["kimi_linear", "kimi_k3", "solar_negative"], help="KDA recurrence/gating recipe")
parser.add_argument("--kda-force-final-global", action=argparse.BooleanOptionalAction, default=True, help="force the final KDA-hybrid layer to global attention")
parser.add_argument("--attn-res-block-size", type=int, default=2, help="transformer layers per Kimi K3 Block AttnRes block")
parser.add_argument("--attn-res-recompute", action=argparse.BooleanOptionalAction, default=True, help="recompute AttnRes reads in backward to bound activation memory")
parser.add_argument("--attn-res-variant", type=str, default="kimi_k3_block_attnres", choices=["kimi_k3_block_attnres", "multi_head_attnres"], help="single- or multi-head depth-routing rule")
parser.add_argument("--attn-res-heads", type=int, default=1, help="number of feature-subspace routing heads (MHAR uses 8)")
parser.add_argument("--dsa-top-k", type=int, default=32, help="number of causal tokens selected by the DSA indexer")
parser.add_argument("--dsa-index-heads", type=int, default=4, help="number of lightning-indexer query heads")
parser.add_argument("--dsa-index-head-dim", type=int, default=128, help="lightning-indexer query/key dimension")
parser.add_argument("--dsa-index-rope-dim", type=int, default=64, help="leading indexer dimensions receiving non-interleaved RoPE")
parser.add_argument("--dsa-dense-warmup-steps", type=int, default=40, help="joint dense LM/indexer-alignment steps before sparse attention")
parser.add_argument("--dsa-query-chunk-size", type=int, default=128, help="query chunk used by the semantic DSA indexer")
parser.add_argument("--dsa-backend", type=str, default="sdpa_masked", choices=["sdpa_masked"], help="DSA execution backend")
parser.add_argument("--dsa-warmup-indexer-lr", type=float, default=1e-3, help="indexer-only LR during dense alignment")
parser.add_argument("--dsa-sparse-indexer-lr", type=float, default=7.3e-6, help="indexer-only LR during sparse training")
parser.add_argument("--search-mlp", type=str, default="baseline", choices=["baseline", "sparser", "colu"], help="FFN arm for the composable search family")
parser.add_argument("--gated-mlp-width", type=int, default=-1, help="gated FFN width (-1 gives an approximately parameter-matched width)")
parser.add_argument("--sparser-l1-coeff", type=float, default=0.0, help="Sakana-style L1 coefficient on gated FFN hidden activations")
parser.add_argument("--colu-dim", type=int, default=4, help="channels per explicit-axis soft CoLU group")
parser.add_argument("--qat-recipe", type=str, default="none", choices=["none", "8da4w"], help="fake-quantization recipe for the composable search family")
parser.add_argument("--qat-group-size", type=int, default=128, help="signed-int4 weight group size (tail padding is allowed)")
parser.add_argument("--qat-start-step", type=int, default=0, help="optimization step that enables fake quantization during training")
parser.add_argument("--qat-min-size", type=int, default=128, help="skip QAT for a Linear if either matrix dimension is smaller")
parser.add_argument("--sota-variant", type=str, default="baseline", choices=["baseline", "gated_attention", "exclusive_attention", "differential_attention", "xielu", "dynamic_tanh", "peri_ln", "canon_abcd", "bank_of_values"], help="single-change arm in the controlled SoTA pool")
parser.add_argument("--sota-extra-lr", type=float, default=0.005, help="AdamW LR for SoTA-specific non-matrix parameters")
parser.add_argument("--canon-kernel-size", type=int, default=4, help="causal depthwise kernel size for Canon-ABCD")
parser.add_argument("--bov-target-fraction", type=float, default=1.0/3.0, help="fraction of deepest layers using Bank of Values")
parser.add_argument("--frontier-variant", type=str, default="inkling_relative_attention", choices=["inkling_relative_attention", "inkling_sconv_kv", "inkling_sconv_residual", "hybrid_swa_5_1_w512", "inkling_lr2_weight_decay", "partial_rope_25", "zero_centered_rmsnorm", "kimi_situ_glu", "shared_mtp3", "attention_sink", "per_head_muon", "qwen_gdn", "deepseek_csa", "deepseek_hca", "glm_mla_muon_split", "glm_simple_gdn", "motif_gdla", "motif_mhc_anneal"], help="isolated frontier-report component")
parser.add_argument("--frontier-extra-lr", type=float, default=0.005, help="AdamW LR for frontier-specific vectors and biases")
parser.add_argument("--relative-dim", type=int, default=16, help="Inkling relative-state dimension")
parser.add_argument("--relative-extent", type=int, default=1024, help="Inkling learned relative-distance extent")
parser.add_argument("--sconv-kernel-size", type=int, default=4, help="Inkling short-convolution kernel")
parser.add_argument("--mtp-depth", type=int, default=3, help="number of future-token heads using the shared MTP block")
parser.add_argument("--mtp-loss-weight", type=float, default=0.1, help="shared MTP auxiliary-loss coefficient")
parser.add_argument("--pareto-components", type=str, default="qwen_gdn,xielu", help="comma-separated controlled components for the Pareto combination family")
parser.add_argument("--engram-layers", type=comma_separated_ints, default=(1, 6), help="zero-based Engram injection layers")
parser.add_argument("--engram-ngram-orders", type=comma_separated_ints, default=(2, 3), help="suffix n-gram orders used by Engram")
parser.add_argument("--engram-num-heads", type=int, default=8, help="Engram hash heads per n-gram order")
parser.add_argument("--engram-dim", type=int, default=0, help="Engram retrieval width (0 = half model width)")
parser.add_argument("--engram-vocab-multiplier", type=int, default=5, help="Engram table-size multiplier")
parser.add_argument("--engram-kernel-size", type=int, default=4, help="Engram causal convolution kernel")
parser.add_argument("--engram-seed", type=int, default=0, help="Engram hash seed")
parser.add_argument("--mhc-num-streams", type=int, default=4, help="number of mHC residual streams")
parser.add_argument("--mhc-init-gating-factor", type=float, default=0.01, help="initial mHC mapping scale")
parser.add_argument("--mhc-sinkhorn-iterations", type=int, default=20, help="mHC Sinkhorn-Knopp iterations")
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
parser.add_argument("--per-head-muon", action=argparse.BooleanOptionalAction, default=False, help="split attention Q/K/V projections into one Muon matrix per head")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--prompt-file", type=str, default=None, help="portable YAML prompt set used for periodic sampling")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--save-final-checkpoint", action=argparse.BooleanOptionalAction, default=True, help="save the final model and optimizer state")
parser.add_argument("--quant-monitor-every", type=int, default=-1, help="monitor FOG kurtosis / backend quant stats every N steps (-1 = disable)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
parser.add_argument("--max-parameters", type=int, default=-1, help="fail before training if total parameters reach this cap (-1 disables)")
parser.add_argument("--max-training-tokens", type=int, default=-1, help="fail before training if tokens reach this cap (-1 disables)")
parser.add_argument("--finite-check-every", type=int, default=1, help="scan gradients for NaN/Inf every N steps (0 disables gradient scans; loss and validation are always checked)")
args = parser.parse_args()
if args.fp8 and args.arch_family == "fog":
    parser.error("--fp8 is the legacy nanochat path. Use --precision-recipe for --arch-family=fog.")
if args.precision_recipe != "bf16" and args.arch_family != "fog":
    parser.error("--precision-recipe is only supported with --arch-family=fog. Use legacy --fp8 for next_gen_arch.arch.")
if args.arch_family == "deepseek_dsa" and args.window_pattern.upper() != "L":
    parser.error("--arch-family=deepseek_dsa requires --window-pattern=L")
if args.arch_family == "sota_pool" and args.window_pattern.upper() != "L":
    parser.error("--arch-family=sota_pool requires --window-pattern=L for the controlled comparison")
if args.arch_family == "frontier_pool" and args.window_pattern.upper() != "L":
    parser.error("--arch-family=frontier_pool requires --window-pattern=L; hybrid windows are selected by the variant")
if args.arch_family == "pareto_combo" and args.window_pattern.upper() != "L":
    parser.error("--arch-family=pareto_combo requires --window-pattern=L")
if args.arch_family == "engram" and any(layer < 0 or layer >= args.depth for layer in args.engram_layers):
    parser.error(f"--engram-layers={args.engram_layers} must fit depth {args.depth}")
if args.finite_check_every < 0:
    parser.error("--finite-check-every must be non-negative")
if args.arch_family in {"deepseek_dsa", "frontier_pool", "pareto_combo"}:
    # DSA and several frontier components add legitimate matrix shapes. The
    # fused optimizer helper is fullgraph-compiled once per shape, so PyTorch's
    # default limit of eight is too small even though each graph is stable after
    # its first compilation.
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
if (
    args.arch_family == "fog"
    and precision_recipe_requires_full_context_window(args.precision_recipe)
    and not is_full_context_window_pattern(args.window_pattern)
):
    parser.error(
        f"--precision-recipe={args.precision_recipe} requires full-context FOG attention. "
        f"Use --window-pattern L (or an all-L equivalent), got '{args.window_pattern}'."
    )
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type, seed=args.seed)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0


def any_rank_nonfinite(local_flag: torch.Tensor) -> bool:
    """Return one synchronized non-finite decision across all ranks."""
    flag = local_flag.to(device=device, dtype=torch.int32)
    if is_ddp_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def require_finite_scalar(name: str, value, step: int) -> None:
    scalar = torch.as_tensor(value, device=device)
    if any_rank_nonfinite(~torch.isfinite(scalar).all()):
        raise FloatingPointError(f"Non-finite {name} detected at step {step}")


def require_finite_gradients(module: torch.nn.Module, step: int) -> None:
    nonfinite = torch.zeros((), device=device, dtype=torch.bool)
    for parameter in module.parameters():
        if parameter.grad is not None:
            nonfinite.logical_or_(~torch.isfinite(parameter.grad).all())
    if any_rank_nonfinite(nonfinite):
        raise FloatingPointError(f"Non-finite gradient detected at step {step}")


if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
precision_backend = resolve_precision_backend(
    args.precision_recipe,
    device_type=device_type,
    gpu_name=gpu_device_name if device_type == "cuda" else None,
    stochastic_rounding=args.stochastic_rounding,
    split_accumulator=args.split_accumulator,
)
print0(f"Precision recipe: {args.precision_recipe} ({precision_backend.reason})")
print0(f"Precision controls: {precision_backend.describe_controls()}")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
backend_name = describe_attention_backend(ATTENTION_BACKEND)
using_flash_attention = ATTENTION_BACKEND in {"fa3", "fa4"}
print0(f"Attention backend: {backend_name} ({ATTENTION_BACKEND_REASON})")
if using_flash_attention:
    print0(f"✓ Using {backend_name}.")
else:
    print0("!" * 80)
    if HAS_FLASH_ATTENTION and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention is available, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0(f"WARNING: Using PyTorch SDPA fallback ({ATTENTION_BACKEND_REASON})")
    print0("WARNING: Training will be less efficient without a Flash Attention backend")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")


def build_compressed_token_map(tokenizer, vocab_size):
    """Apply the fixed Engram token normalization used by the isolated arm."""
    mapping = []
    key_to_id = {}
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
# -----------------------------------------------------------------------------
# Initialize the Model


def scaled_engram_layers(target_depth: int) -> tuple[int, ...]:
    """Map Engram injections onto a meta-reference depth without changing the run."""
    if args.arch_family != "engram" or target_depth == args.depth:
        return args.engram_layers
    scaled = []
    for layer in args.engram_layers:
        candidate = round((layer + 0.5) * target_depth / args.depth - 0.5)
        candidate = min(max(candidate, 0), target_depth - 1)
        if candidate not in scaled:
            scaled.append(candidate)
    if len(scaled) != len(args.engram_layers):
        raise ValueError(
            f"Cannot preserve {len(args.engram_layers)} Engram injections at depth {target_depth}"
        )
    return tuple(scaled)


def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    config = build_model_config(
        arch_family=args.arch_family,
        depth=depth,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        vocab_size=vocab_size,
        window_pattern=args.window_pattern,
        fog_variant=args.fog_variant,
        per_head_muon=args.per_head_muon,
        kda_pattern=args.kda_pattern,
        kda_rope_policy=args.kda_rope_policy,
        kda_variant=args.kda_variant,
        kda_force_final_global=args.kda_force_final_global,
        dsa_top_k=args.dsa_top_k,
        dsa_index_heads=args.dsa_index_heads,
        dsa_index_head_dim=args.dsa_index_head_dim,
        dsa_index_rope_dim=args.dsa_index_rope_dim,
        dsa_dense_warmup_steps=args.dsa_dense_warmup_steps,
        dsa_query_chunk_size=args.dsa_query_chunk_size,
        dsa_backend=args.dsa_backend,
        dsa_warmup_indexer_lr=args.dsa_warmup_indexer_lr,
        dsa_sparse_indexer_lr=args.dsa_sparse_indexer_lr,
        attn_res_block_size=args.attn_res_block_size,
        attn_res_recompute=args.attn_res_recompute,
        attn_res_variant=args.attn_res_variant,
        attn_res_heads=args.attn_res_heads,
        search_mlp=args.search_mlp,
        gated_mlp_width=args.gated_mlp_width,
        sparser_l1_coeff=args.sparser_l1_coeff,
        colu_dim=args.colu_dim,
        qat_recipe=args.qat_recipe,
        qat_group_size=args.qat_group_size,
        qat_start_step=args.qat_start_step,
        qat_min_size=args.qat_min_size,
        sota_variant=args.sota_variant,
        sota_extra_lr=args.sota_extra_lr,
        canon_kernel_size=args.canon_kernel_size,
        bov_target_fraction=args.bov_target_fraction,
        frontier_variant=args.frontier_variant,
        frontier_extra_lr=args.frontier_extra_lr,
        relative_dim=args.relative_dim,
        relative_extent=args.relative_extent,
        sconv_kernel_size=args.sconv_kernel_size,
        mtp_depth=args.mtp_depth,
        mtp_loss_weight=args.mtp_loss_weight,
        pareto_components=args.pareto_components,
        engram_layers=scaled_engram_layers(depth),
        engram_ngram_orders=args.engram_ngram_orders,
        engram_num_heads=args.engram_num_heads,
        engram_dim=args.engram_dim,
        engram_vocab_multiplier=args.engram_vocab_multiplier,
        engram_kernel_size=args.engram_kernel_size,
        engram_seed=args.engram_seed,
        mhc_num_streams=args.mhc_num_streams,
        mhc_init_gating_factor=args.mhc_init_gating_factor,
        mhc_sinkhorn_iterations=args.mhc_sinkhorn_iterations,
    )
    with torch.device("meta"):
        model_meta = instantiate_model(config, runtime_backend="native")
    return model_meta

# Build the config once so all variants share the exact same architecture.
model_config = build_model_config(
    arch_family=args.arch_family,
    depth=args.depth,
    aspect_ratio=args.aspect_ratio,
    head_dim=args.head_dim,
    max_seq_len=args.max_seq_len,
    vocab_size=vocab_size,
    window_pattern=args.window_pattern,
    fog_variant=args.fog_variant,
    per_head_muon=args.per_head_muon,
    kda_pattern=args.kda_pattern,
    kda_rope_policy=args.kda_rope_policy,
    kda_variant=args.kda_variant,
    kda_force_final_global=args.kda_force_final_global,
    dsa_top_k=args.dsa_top_k,
    dsa_index_heads=args.dsa_index_heads,
    dsa_index_head_dim=args.dsa_index_head_dim,
    dsa_index_rope_dim=args.dsa_index_rope_dim,
    dsa_dense_warmup_steps=args.dsa_dense_warmup_steps,
    dsa_query_chunk_size=args.dsa_query_chunk_size,
    dsa_backend=args.dsa_backend,
    dsa_warmup_indexer_lr=args.dsa_warmup_indexer_lr,
    dsa_sparse_indexer_lr=args.dsa_sparse_indexer_lr,
    attn_res_block_size=args.attn_res_block_size,
    attn_res_recompute=args.attn_res_recompute,
    attn_res_variant=args.attn_res_variant,
    attn_res_heads=args.attn_res_heads,
    search_mlp=args.search_mlp,
    gated_mlp_width=args.gated_mlp_width,
    sparser_l1_coeff=args.sparser_l1_coeff,
    colu_dim=args.colu_dim,
    qat_recipe=args.qat_recipe,
    qat_group_size=args.qat_group_size,
    qat_start_step=args.qat_start_step,
    qat_min_size=args.qat_min_size,
    sota_variant=args.sota_variant,
    sota_extra_lr=args.sota_extra_lr,
    canon_kernel_size=args.canon_kernel_size,
    bov_target_fraction=args.bov_target_fraction,
    frontier_variant=args.frontier_variant,
    frontier_extra_lr=args.frontier_extra_lr,
    relative_dim=args.relative_dim,
    relative_extent=args.relative_extent,
    sconv_kernel_size=args.sconv_kernel_size,
    mtp_depth=args.mtp_depth,
    mtp_loss_weight=args.mtp_loss_weight,
    pareto_components=args.pareto_components,
    engram_layers=args.engram_layers,
    engram_ngram_orders=args.engram_ngram_orders,
    engram_num_heads=args.engram_num_heads,
    engram_dim=args.engram_dim,
    engram_vocab_multiplier=args.engram_vocab_multiplier,
    engram_kernel_size=args.engram_kernel_size,
    engram_seed=args.engram_seed,
    mhc_num_streams=args.mhc_num_streams,
    mhc_init_gating_factor=args.mhc_init_gating_factor,
    mhc_sinkhorn_iterations=args.mhc_sinkhorn_iterations,
)
model_config_kwargs = model_config_to_dict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
if precision_backend.requires_materialized_construction:
    with torch.device(device):
        model = instantiate_model(model_config, runtime_backend=precision_backend.runtime_backend)
    model.init_weights()
else:
    with torch.device("meta"):
        model = instantiate_model(model_config, runtime_backend=precision_backend.runtime_backend)
    model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
    model.init_weights() # 3) All tensors get initialized

if args.arch_family == "engram" or (
    args.arch_family == "pareto_combo" and "engram" in model_config.components
):
    compressed_vocab_size = None
    if master_process:
        compressed_map, compressed_vocab_size = build_compressed_token_map(tokenizer, vocab_size)
        model.configure_engram_token_map(compressed_map, tokenizer.get_bos_token_id())
        print0(
            f"Engram compressed vocab size: {compressed_vocab_size:,} "
            f"({compressed_vocab_size / vocab_size:.2%} of raw)"
        )
    if ddp:
        dist.broadcast(model.engram_token_map, src=0)
        dist.broadcast(model.engram_pad_id, src=0)

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
if args.arch_family == "nanochat":
    default_model_tag = f"d{args.depth}"
elif args.arch_family == "fog":
    default_model_tag = f"fog_{args.fog_variant}_d{args.depth}"
elif args.arch_family == "kimi_kda":
    default_model_tag = f"kimi_kda_d{args.depth}"
elif args.arch_family == "kimi_attnres":
    default_model_tag = (
        f"mhar_h{args.attn_res_heads}_d{args.depth}"
        if args.attn_res_variant == "multi_head_attnres"
        else f"kimi_attnres_d{args.depth}"
    )
elif args.arch_family == "combo_search":
    default_model_tag = f"combo_{args.search_mlp}_{args.qat_recipe}_d{args.depth}"
elif args.arch_family == "sota_pool":
    default_model_tag = f"sota_{args.sota_variant}_d{args.depth}"
elif args.arch_family == "frontier_pool":
    default_model_tag = f"frontier_{args.frontier_variant}_d{args.depth}"
elif args.arch_family == "pareto_combo":
    default_model_tag = f"pareto_{args.pareto_components.replace(',', '_')}_d{args.depth}"
elif args.arch_family == "engram":
    default_model_tag = f"engram_d{args.depth}"
elif args.arch_family == "mhc":
    default_model_tag = f"mhc_d{args.depth}"
else:
    default_model_tag = f"deepseek_dsa_d{args.depth}"
output_dirname = args.model_tag if args.model_tag else default_model_tag
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from next_gen_arch.training.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
if args.arch_family == "combo_search" and args.qat_recipe != "none":
    # Delayed QAT needs distinct eval, dense-train, and fake-quant train graphs
    # for many wrapped Linear sites. PyTorch's default cache of eight frames can
    # otherwise fall back to eager at the phase boundary without changing
    # numerics, making reference throughput needlessly misleading.
    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, 64
    )
    torch._dynamo.config.accumulated_cache_size_limit = max(
        torch._dynamo.config.accumulated_cache_size_limit, 256
    )
if device_type == "mps":
    print0("WARNING: Skipping torch.compile on MPS due to unstable Metal codegen in current PyTorch.")
elif args.arch_family == "fog":
    print0("WARNING: Skipping torch.compile for FOG runs to keep TE integration and quant monitoring predictable.")
else:
    model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
if args.max_parameters > 0 and num_params >= args.max_parameters:
    raise ValueError(f"Parameter cap violated: {num_params:,} >= {args.max_parameters:,}")
num_flops_per_token = model.estimate_flops()
executed_flops_per_token = model.estimate_executed_flops() if hasattr(model, "estimate_executed_flops") else num_flops_per_token
print0(f"Estimated algorithmic FLOPs per token: {num_flops_per_token:e}")
print0(f"Estimated executed FLOPs per token: {executed_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
if args.max_training_tokens > 0 and total_tokens >= args.max_training_tokens:
    raise ValueError(f"Training-token cap violated: {total_tokens:,} >= {args.max_training_tokens:,}")
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    cosine = 0.5 * (1 + math.cos(math.pi * it / num_iterations))
    if args.arch_family == "frontier_pool" and args.frontier_variant == "inkling_lr2_weight_decay":
        # Inkling couples decay strength to eta^2. Normalize by the initial LR
        # so step zero retains the controlled baseline decay coefficient.
        lrm = get_lr_multiplier(it)
        return weight_decay_scaled * cosine * lrm * lrm
    return weight_decay_scaled * cosine

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    if hasattr(orig_model, "set_training_step"):
        orig_model.set_training_step(step)
    flops_so_far = num_flops_per_token * total_batch_size * step
    executed_flops_so_far = executed_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            with precision_backend.eval_context():
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        require_finite_scalar("validation BPB", val_bpb, step)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_executed_flops": executed_flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            with precision_backend.eval_context():
                results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = load_prompt_texts(args.prompt_file)
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                with precision_backend.eval_context():
                    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0, seed=args.seed)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if (last_step and args.save_final_checkpoint) or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "architecture_state": orig_model.get_architecture_state() if hasattr(orig_model, "get_architecture_state") else None,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    monitor_this_step = args.arch_family == "fog" and args.quant_monitor_every > 0 and step % args.quant_monitor_every == 0
    if hasattr(orig_model, "set_quant_monitor_enabled"):
        orig_model.set_quant_monitor_enabled(monitor_this_step)
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    architecture_log_data = {}
    loss_nonfinite = torch.zeros((), device=device, dtype=torch.bool)
    for micro_step in range(grad_accum_steps):
        with precision_backend.training_context():
            loss = model(x, y)
        loss_nonfinite.logical_or_(~torch.isfinite(loss.detach()).all())
        if hasattr(orig_model, "consume_training_metrics"):
            architecture_log_data = orig_model.consume_training_metrics()
            train_loss = architecture_log_data.get("train/lm_loss", loss.detach())
        else:
            train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    if any_rank_nonfinite(loss_nonfinite):
        raise FloatingPointError(f"Non-finite training loss detected at step {step}")
    if args.finite_check_every and step % args.finite_check_every == 0:
        require_finite_gradients(orig_model, step)
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        if group.get("dsa_indexer"):
            group["lr"] = group["dsa_sparse_lr"] if step >= args.dsa_dense_warmup_steps else group["dsa_warmup_lr"]
        else:
            group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    architecture_log_data = {
        key: (value.item() if isinstance(value, torch.Tensor) else value)
        for key, value in architecture_log_data.items()
    }
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    quant_log_data = {}
    if monitor_this_step and hasattr(orig_model, "consume_quant_metrics"):
        quant_log_data.update(orig_model.consume_quant_metrics())
        quant_log_data.update(precision_backend.collect_debug_metrics(orig_model))
    if hasattr(orig_model, "set_quant_monitor_enabled"):
        orig_model.set_quant_monitor_enabled(False)
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = executed_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    architecture_suffix = ""
    if "dsa/indexer_kl" in architecture_log_data:
        architecture_suffix = f" | indexer_kl: {architecture_log_data['dsa/indexer_kl']:.6f} | dsa_phase: {'sparse' if architecture_log_data.get('dsa/sparse_phase') else 'dense_warmup'}"
    elif "sparser/l1_sum_per_token" in architecture_log_data:
        architecture_suffix = f" | ff_l1: {architecture_log_data['sparser/l1_sum_per_token']:.5f} | active: {architecture_log_data['sparser/active_fraction']:.3f}"
    elif "qat/enabled" in architecture_log_data:
        architecture_suffix = f" | qat: {'on' if architecture_log_data['qat/enabled'] else 'off'}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}{architecture_suffix}")
    if quant_log_data:
        preview_keys = [
            "quant/qkv/kurtosis",
            "quant/ffn_inner/kurtosis",
            "quant/block_output/kurtosis",
            "quant/backend_amax",
        ]
        summary_parts = [f"{key.split('/')[-2]}: {quant_log_data[key]:.4f}" for key in preview_keys if key in quant_log_data]
        if summary_parts:
            print0("quant | " + " | ".join(summary_parts))
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_executed_flops": executed_flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        log_data.update(quant_log_data)
        log_data.update(architecture_log_data)
        wandb_run.log(log_data)
    elif quant_log_data or architecture_log_data:
        wandb_run.log({"step": step, **quant_log_data, **architecture_log_data})

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")
completed_steps = max(num_iterations - 10, 0)
avg_step_time = (total_training_time / completed_steps) if completed_steps > 0 else None
avg_tok_per_sec = (int(total_batch_size / avg_step_time) if avg_step_time and avg_step_time > 0 else None)
summary_data = {
    "arch_family": args.arch_family,
    "fog_variant": args.fog_variant if args.arch_family == "fog" else None,
    "precision_recipe": args.precision_recipe,
    "precision_backend": precision_backend.reason,
    "precision_te_recipe": precision_backend.te_recipe_name,
    "precision_stochastic_rounding": precision_backend.stochastic_rounding,
    "precision_split_accumulator": precision_backend.split_accumulator,
    "attention_backend": describe_attention_backend(ATTENTION_BACKEND),
    "architecture_state": orig_model.get_architecture_state() if hasattr(orig_model, "get_architecture_state") else None,
    "algorithmic_flops_per_token": num_flops_per_token,
    "executed_flops_per_token": executed_flops_per_token,
    "seed": args.seed,
    "model_tag": output_dirname,
    "step": num_iterations,
    "num_iterations": num_iterations,
    "total_tokens": total_tokens,
    "total_training_time_s": total_training_time,
    "avg_step_time_s": avg_step_time,
    "avg_tok_per_sec": avg_tok_per_sec,
    "last_step_time_s": dt if "dt" in locals() else None,
    "last_tok_per_sec": tok_per_sec if "tok_per_sec" in locals() else None,
    "val_bpb": val_bpb,
    "min_val_bpb": min_val_bpb if val_bpb is not None else None,
    "core_metric": results.get("core_metric", None),
    "final_lm_loss": train_loss_f if "train_loss_f" in locals() else None,
    "final_architecture_metrics": architecture_log_data if "architecture_log_data" in locals() else {},
}
if master_process:
    summary_path = os.path.join(checkpoint_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print0(f"Wrote training summary to {summary_path}")

# Log to report
from next_gen_arch.training.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Executed FLOPs per token": f"{executed_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "Arch family": args.arch_family,
        "Precision recipe": args.precision_recipe,
        "Precision controls": precision_backend.describe_controls(),
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total executed flops": f"{executed_flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        "Average step time (s)": avg_step_time,
        "Average tok/sec": avg_tok_per_sec,
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
