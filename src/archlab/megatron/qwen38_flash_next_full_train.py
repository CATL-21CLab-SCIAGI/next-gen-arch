"""Megatron-native trainer for the full Qwen3.8-Flash-Next text variant."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from archlab.architectures.qwen38_flash_next_full import (
    SOURCE_CONFIG_SHA256,
    SOURCE_MODEL,
    SOURCE_REVISION,
    TOKENIZER_SHA256,
    DistributedPLE,
    FourStreamGatedResidual,
    GatedDeltaNet,
    Qwen38FlashNextFullConfig,
    parameter_count_contract,
)
from archlab.megatron.backend import validate_runtime

TRAIN_STEPS = 11_921
TOKENS_PER_STEP = 8_388_608
EFFECTIVE_TOKENS = TRAIN_STEPS * TOKENS_PER_STEP
CHECKPOINT_INTERVAL_STEPS = 1_192
CHECKPOINT_WRITER_THREADS = 8
DISTRIBUTED_TIMEOUT_MINUTES = 60
NATIVE_MUON_FP32_MATMUL_PRECISION = "medium"
FULL_MODEL_VARIANT = "full"
QUARTER_DEPTH48_NO_MTP_MODEL_VARIANT = "quarter-depth48-no-mtp"
LOSS_NORMALIZATION = "global-valid-token-mean-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class DPRankTokenBatches:
    """Deterministic raw-int32 batches keyed only by data-parallel rank."""

    def __init__(
        self,
        prefixes: list[Path],
        *,
        batch_size: int,
        sequence_len: int,
        start_batch: int,
        device: torch.device,
        repeat_window_batches: int | None = None,
    ):
        if not prefixes:
            raise ValueError("at least one indexed-data prefix is required")
        self.arrays = [np.memmap(f"{prefix}.bin", mode="r", dtype=np.int32) for prefix in prefixes]
        if any(array.size <= sequence_len for array in self.arrays):
            raise ValueError("each indexed-data part must contain more than one sequence")
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.batch_index = start_batch
        self.start_batch = start_batch
        self.repeat_window_batches = repeat_window_batches
        self.device = device
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future | None = None
        if repeat_window_batches is not None and repeat_window_batches < 1:
            raise ValueError("repeat_window_batches must be positive")
        if device.type == "cuda":
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="dp-token-prefetch"
            )
            self._future = self._executor.submit(self._cpu_batch, self._source_index(start_batch))

    @staticmethod
    def _cyclic_slice(array: np.memmap, start: int, length: int) -> np.ndarray:
        start %= array.size
        if start + length <= array.size:
            return np.asarray(array[start : start + length])
        pieces = [np.asarray(array[start:])]
        remaining = length - pieces[0].size
        while remaining >= array.size:
            pieces.append(np.asarray(array[:]))
            remaining -= array.size
        if remaining:
            pieces.append(np.asarray(array[:remaining]))
        return np.concatenate(pieces)

    def _source_index(self, batch_index: int) -> int:
        if self.repeat_window_batches is None:
            return batch_index
        return self.start_batch + (batch_index - self.start_batch) % self.repeat_window_batches

    def _cpu_batch(self, batch_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        array = self.arrays[batch_index % len(self.arrays)]
        flat = self.batch_size * self.sequence_len
        start = (batch_index // len(self.arrays)) * flat
        window = self._cyclic_slice(array, start, flat + 1)
        tensor = torch.from_numpy(np.array(window, dtype=np.int64, copy=True))
        if self.device.type == "cuda":
            tensor = tensor.pin_memory()
        return (
            tensor[:-1].view(self.batch_size, self.sequence_len),
            tensor[1:].view(self.batch_size, self.sequence_len),
        )

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._future is None:
            tokens, labels = self._cpu_batch(self._source_index(self.batch_index))
        else:
            tokens, labels = self._future.result()
        self.batch_index += 1
        if self._executor is not None:
            self._future = self._executor.submit(
                self._cpu_batch, self._source_index(self.batch_index)
            )
        return {
            "tokens": tokens.to(self.device, non_blocking=True),
            "labels": labels.to(self.device, non_blocking=True),
        }

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def partition_prefixes_for_dp_rank(
    prefixes: list[Path],
    data_parallel_rank: int,
    data_parallel_world_size: int,
    *,
    require_distinct: bool = True,
) -> list[Path]:
    if not prefixes:
        raise ValueError("at least one indexed-data prefix is required")
    if data_parallel_world_size < 1 or not 0 <= data_parallel_rank < data_parallel_world_size:
        raise ValueError("invalid data-parallel rank/world size")
    if require_distinct and len(prefixes) < data_parallel_world_size:
        raise ValueError("training requires at least one indexed-data part per DP rank")
    assigned = prefixes[data_parallel_rank::data_parallel_world_size]
    return assigned or [prefixes[data_parallel_rank % len(prefixes)]]


def _data_prefixes(data_root: Path, split: str) -> list[Path]:
    prefixes = sorted(path.with_suffix("") for path in (data_root / split).glob("part-*.bin"))
    if not prefixes:
        raise FileNotFoundError(f"no {split} part-*.bin files under {data_root}")
    for prefix in prefixes:
        for suffix in (".bin", ".idx", ".json"):
            artifact = Path(f"{prefix}{suffix}")
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                raise FileNotFoundError(f"missing or empty indexed-data artifact: {artifact}")
    return prefixes


def _validated_data_prefixes(data_root: Path) -> tuple[list[Path], list[Path]]:
    data_root = data_root.expanduser().resolve()
    ready_path = data_root / "DATA_READY.json"
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid indexed-data manifest: {ready_path}") from error
    validated = []
    for split, key in (("train", "train_parts"), ("val", "valid_parts")):
        values = ready.get(key)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(x, str) for x in values)
        ):
            raise RuntimeError(f"indexed-data manifest lacks valid {key}")
        declared = []
        for value in values:
            path = Path(value).expanduser()
            declared.append((path if path.is_absolute() else data_root / path).resolve())
        discovered = [prefix.resolve() for prefix in _data_prefixes(data_root, split)]
        if len(set(declared)) != len(declared) or set(declared) != set(discovered):
            raise RuntimeError(f"indexed-data manifest membership changed for {split}")
        validated.append(discovered)
    return validated[0], validated[1]


def shifted_mtp_targets(labels: torch.Tensor, depths: int = 3) -> tuple[torch.Tensor, ...]:
    """CPU-testable statement of the three native-MTP target shifts."""
    targets = []
    for depth in range(1, depths + 1):
        target = torch.full_like(labels, -1)
        if depth < labels.size(1):
            target[:, :-depth] = labels[:, depth:]
        targets.append(target)
    return tuple(targets)


def mtp_weighted_mean(losses: torch.Tensor, scaling: float = 0.1) -> torch.Tensor:
    if losses.size(0) != 3:
        raise ValueError("the supported MTP objective has exactly three depths")
    return scaling * losses.mean(dim=0)


def _native_muon_contract() -> dict[str, Any]:
    return {
        "implementation": "container-owned megatron.core.optimizer TensorParallelMuon",
        "integration": "Megatron --optimizer muon; no adapter or runtime patch",
        "momentum": 0.95,
        "nesterov": True,
        "coefficient": "polar_express",
        "newton_schulz_steps": 8,
        "scale_mode": "spectral",
        "extra_scale_factor": 0.2,
        "fp32_matmul_precision": NATIVE_MUON_FP32_MATMUL_PRECISION,
        "qkv_split": "native query-group Q/K/V split; coarser than Qwen per-head splitting",
        "released_private_optimizer": "Canzona unavailable",
    }


def _megatron_argv(args: argparse.Namespace, config: Qwen38FlashNextFullConfig) -> list[str]:
    tokens_per_step = args.global_batch_size * config.sequence_len
    if args.global_batch_size < args.micro_batch_size:
        raise ValueError("global batch must be at least one microbatch")
    if args.probe_steps:
        train_steps = args.probe_steps
        save_interval = args.probe_save_interval or args.probe_steps
    else:
        if tokens_per_step != TOKENS_PER_STEP:
            raise ValueError(f"production tokens per step must be {TOKENS_PER_STEP}")
        if args.target_train_tokens != EFFECTIVE_TOKENS:
            raise ValueError(f"production target must be exactly {EFFECTIVE_TOKENS} tokens")
        train_steps = TRAIN_STEPS
        save_interval = CHECKPOINT_INTERVAL_STEPS
    argv = [
        f"qwen38-flash-next-{config.arch_family}-bf16",
        "--use-mcore-models",
        "--num-layers",
        str(config.num_hidden_layers),
        "--hidden-size",
        str(config.hidden_size),
        "--ffn-hidden-size",
        str(config.moe_intermediate_size),
        "--num-attention-heads",
        str(config.attention_heads),
        "--num-query-groups",
        str(config.attention_kv_heads),
        "--kv-channels",
        str(config.attention_head_dim),
        "--seq-length",
        str(config.sequence_len),
        "--max-position-embeddings",
        str(config.max_position_embeddings),
        "--position-embedding-type",
        "rope",
        "--rotary-percent",
        str(config.partial_rotary_factor),
        "--rotary-base",
        str(int(config.rope_theta)),
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        str(config.rms_norm_eps),
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--swiglu",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--global-batch-size",
        str(args.global_batch_size),
        "--train-iters",
        str(train_steps),
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "4",
        "--decoder-first-pipeline-num-layers",
        str(config.pipeline_layers[0]),
        "--decoder-last-pipeline-num-layers",
        str(config.pipeline_layers[-1]),
        "--expert-model-parallel-size",
        "8",
        "--context-parallel-size",
        "1",
        "--distributed-backend",
        "nccl",
        "--distributed-timeout-minutes",
        str(DISTRIBUTED_TIMEOUT_MINUTES),
        "--transformer-impl",
        "transformer_engine",
        "--num-experts",
        str(config.num_experts),
        "--moe-router-topk",
        str(config.num_experts_per_token),
        "--moe-ffn-hidden-size",
        str(config.moe_intermediate_size),
        "--moe-shared-expert-intermediate-size",
        str(config.shared_expert_intermediate_size),
        "--moe-shared-expert-gate",
        "--moe-router-load-balancing-type",
        "aux_loss",
        "--moe-aux-loss-coeff",
        str(config.router_aux_loss_coefficient),
        "--moe-token-dispatcher-type",
        "alltoall",
        "--optimizer",
        "muon",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--adam-eps",
        "1e-8",
        "--muon-momentum",
        "0.95",
        "--muon-nesterov",
        "--muon-scale-mode",
        "spectral",
        "--muon-extra-scale-factor",
        "0.2",
        "--muon-fp32-matmul-prec",
        NATIVE_MUON_FP32_MATMUL_PRECISION,
        "--muon-coefficient-type",
        "polar_express",
        "--muon-num-ns-steps",
        "8",
        "--muon-scalar-optimizer",
        "adam",
        "--lr",
        str(args.learning_rate),
        "--min-lr",
        str(args.minimum_learning_rate),
        "--lr-decay-style",
        "cosine",
        "--lr-warmup-fraction",
        str(args.warmup_fraction),
        "--weight-decay",
        str(args.weight_decay),
        "--clip-grad",
        str(args.clip_grad),
        "--bf16",
        "--use-distributed-optimizer",
        "--no-use-layer-wise-param-layout",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        str(config.vocab_size),
        "--dataloader-type",
        "external",
        "--num-workers",
        "0",
        "--eval-interval",
        str(args.eval_interval),
        "--eval-iters",
        str(args.eval_iters),
        "--log-interval",
        str(args.log_interval),
        "--log-throughput",
        "--calculate-per-token-loss",
        "--rerun-mode",
        "disabled",
        "--no-masked-softmax-fusion",
        "--no-bias-gelu-fusion",
        "--no-bias-swiglu-fusion",
        "--no-bias-dropout-fusion",
        "--no-rope-fusion",
        "--save",
        str(args.run_dir / "checkpoints"),
        "--save-interval",
        str(save_interval),
        "--ckpt-format",
        "torch_dist",
        "--dist-ckpt-workers",
        str(CHECKPOINT_WRITER_THREADS),
        "--exit-signal-handler",
        "--tensorboard-dir",
        str(args.run_dir / "tensorboard"),
        "--seed",
        str(args.seed),
    ]
    if config.router_z_loss_coefficient:
        argv.extend(("--moe-z-loss-coeff", str(config.router_z_loss_coefficient)))
    if config.mtp_num_layers:
        argv.extend(
            (
                "--mtp-num-layers",
                str(config.mtp_num_layers),
                "--mtp-use-repeated-layer",
                "--mtp-loss-scaling-factor",
                str(config.mtp_loss_scaling_factor),
            )
        )
    marker = args.run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    if args.resume and marker.is_file():
        argv.extend(("--load", str(args.run_dir / "checkpoints")))
        if args.probe_steps:
            # Probe horizons are intentionally short and may grow between the
            # save and reload gates. Keep the checkpoint's optimizer tensors,
            # but use the resumed probe's native scheduler horizon. Production
            # always retains the fixed 11,921-step contract and never overrides.
            argv.append("--override-opt-param-scheduler")
    return argv


def _tag_native_optimizer_fallbacks(model: torch.nn.Module) -> dict[str, int]:
    counts = {"muon": 0, "adamw": 0, "ple_adam_no_decay": 0}
    for name, parameter in model.named_parameters():
        if ".embedding.tables." in name:
            parameter.is_embedding_or_output_parameter = True
            parameter.archlab_optimizer = "adam"
            parameter.archlab_no_weight_decay = True
            counts["ple_adam_no_decay"] += parameter.numel()
        elif ".router." in name or name.endswith("router.weight"):
            parameter.is_embedding_or_output_parameter = True
            parameter.archlab_optimizer = "adamw"
            counts["adamw"] += parameter.numel()
        elif getattr(parameter, "is_embedding_or_output_parameter", False) or parameter.ndim != 2:
            counts["adamw"] += parameter.numel()
        else:
            parameter.archlab_optimizer = "muon"
            counts["muon"] += parameter.numel()
    counts["all_trainable_parameters"] = sum(p.numel() for p in model.parameters())
    return counts


def _bind_native_moe_layer_number(moe_layer: Any, layer_number: int) -> None:
    """Propagate the global layer number through MCore's public MoE interface."""
    if layer_number < 1:
        raise RuntimeError("native MoE layers require a positive global layer number")
    moe_layer.set_layer_number(layer_number)
    if moe_layer.layer_number != layer_number or moe_layer.router.layer_number != layer_number:
        raise RuntimeError("native MCore did not bind the MoE router layer number")


def _resolve_qwen_layer_number(
    local_layer_number: int, *, is_mtp_layer: bool, backbone_offset: int
) -> int:
    """Keep MTP depth-local numbering; MCore adds the backbone offset when logging it."""
    if local_layer_number < 1 or backbone_offset < 0:
        raise RuntimeError("Qwen layer numbers and backbone offsets must be valid")
    return local_layer_number if is_mtp_layer else local_layer_number + backbone_offset


def _build_model_classes(architecture_config: Qwen38FlashNextFullConfig):
    """Import the container runtime lazily and build its native module spec."""
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec_for_backend
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec, build_module
    from megatron.core.transformer.transformer_block import (
        TransformerBlockSubmodules,
        get_num_layers_to_build,
    )
    from megatron.core.transformer.transformer_layer import (
        BaseTransformerLayer,
        TransformerLayerSubmodules,
        get_transformer_layer_offset,
    )
    from megatron.core.utils import get_pg_rank, get_pg_size

    class SplitSwiGLUExperts(MegatronModule):
        """Native grouped GEMMs with distinct gate/up Muon parameters."""

        def __init__(self, num_local_experts, config, pg_collection=None, name=None):
            super().__init__(config)
            self.tp_group = pg_collection.expt_tp
            common = {
                "config": config,
                "bias": False,
                "skip_bias_add": False,
                "is_expert": True,
                "pg_collection": pg_collection,
            }
            self.gate_proj = TEColumnParallelGroupedLinear(
                num_local_experts,
                config.hidden_size,
                config.moe_ffn_hidden_size,
                init_method=config.init_method,
                tp_comm_buffer_name="fc1_gate",
                name=f"{name}.gate_proj" if name else None,
                **common,
            )
            self.up_proj = TEColumnParallelGroupedLinear(
                num_local_experts,
                config.hidden_size,
                config.moe_ffn_hidden_size,
                init_method=config.init_method,
                tp_comm_buffer_name="fc1_up",
                name=f"{name}.up_proj" if name else None,
                **common,
            )
            self.down_proj = TERowParallelGroupedLinear(
                num_local_experts,
                config.moe_ffn_hidden_size,
                config.hidden_size,
                init_method=config.output_layer_init_method,
                tp_comm_buffer_name="fc2",
                name=f"{name}.down_proj" if name else None,
                **common,
            )

        def forward(self, hidden_states, tokens_per_expert, permuted_probs):
            splits = tokens_per_expert.tolist()
            gate, _ = self.gate_proj(hidden_states, splits)
            up, _ = self.up_proj(hidden_states, splits)
            intermediate = F.silu(gate) * up
            intermediate = intermediate * permuted_probs.unsqueeze(-1).to(intermediate.dtype)
            output, _ = self.down_proj(intermediate, splits)
            return output, None

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            state = {}
            for child_name, child in self.named_children():
                state.update(
                    child.sharded_state_dict(f"{prefix}{child_name}.", sharded_offsets, metadata)
                )
            return state

    class SplitSwiGLUSharedExpert(MegatronModule):
        """Native TP linears for the shared expert, also with split FC1."""

        def __init__(self, config, pg_collection=None, gate=True, name=None):
            super().__init__(config)
            self.tp_group = pg_collection.tp
            width = config.moe_shared_expert_intermediate_size
            common = {
                "config": config,
                "bias": False,
                "skip_bias_add": False,
                "is_expert": False,
                "tp_group": self.tp_group,
            }
            self.gate_proj = TEColumnParallelLinear(
                config.hidden_size,
                width,
                gather_output=False,
                init_method=config.init_method,
                name=f"{name}.gate_proj" if name else None,
                **common,
            )
            self.up_proj = TEColumnParallelLinear(
                config.hidden_size,
                width,
                gather_output=False,
                init_method=config.init_method,
                name=f"{name}.up_proj" if name else None,
                **common,
            )
            row_common = dict(common)
            row_common.pop("skip_bias_add")
            self.down_proj = TERowParallelLinear(
                width,
                config.hidden_size,
                input_is_parallel=True,
                skip_bias_add=False,
                init_method=config.output_layer_init_method,
                name=f"{name}.down_proj" if name else None,
                **row_common,
            )
            self.gate_weight = (
                torch.nn.Parameter(torch.empty(1, config.hidden_size)) if gate else None
            )
            if self.gate_weight is not None:
                config.init_method(self.gate_weight)
                self.gate_weight.is_embedding_or_output_parameter = True

        def forward(self, hidden_states):
            gate, _ = self.gate_proj(hidden_states)
            up, _ = self.up_proj(hidden_states)
            output, _ = self.down_proj(F.silu(gate) * up)
            if self.gate_weight is not None:
                output = output * torch.sigmoid(F.linear(hidden_states, self.gate_weight))
            return output

    class QwenFlashNextLayer(MegatronModule, BaseTransformerLayer):
        def __init__(
            self,
            config,
            submodules,
            layer_number=1,
            pg_collection=None,
            vp_stage=None,
            is_mtp_layer=False,
            **_kwargs,
        ):
            MegatronModule.__init__(self, config)
            self.submodules = submodules
            self.is_mtp_layer = is_mtp_layer
            self.tp_group = pg_collection.tp
            pp_rank = get_pg_rank(pg_collection.pp)
            self.layer_number = _resolve_qwen_layer_number(
                layer_number,
                is_mtp_layer=is_mtp_layer,
                backbone_offset=(
                    0 if is_mtp_layer else get_transformer_layer_offset(config, vp_stage, pp_rank)
                ),
            )
            self.attention_kind = (
                "dense"
                if is_mtp_layer
                or self.layer_number % architecture_config.full_attention_interval == 0
                else "gdn"
            )
            self.attention_residual = FourStreamGatedResidual(architecture_config)
            self.mlp_residual = FourStreamGatedResidual(architecture_config)
            if self.attention_kind == "dense":
                self.attention = build_module(
                    submodules.self_attention,
                    config=config,
                    layer_number=self.layer_number,
                    pg_collection=pg_collection,
                )
            else:
                self.attention = GatedDeltaNet(architecture_config)
            self.mlp = submodules.mlp(
                config=config,
                pg_collection=pg_collection,
                is_mtp_layer=is_mtp_layer,
                name=f"layers.{self.layer_number}.mlp",
            )
            _bind_native_moe_layer_number(self.mlp, self.layer_number)
            self.ple = None
            self._ple_input_ids = None
            if not is_mtp_layer and self.layer_number == architecture_config.ngram_layer + 1:
                self.ple = DistributedPLE(
                    architecture_config,
                    owner_rank=get_pg_rank(pg_collection.ep),
                    owner_world_size=get_pg_size(pg_collection.ep),
                    process_group=pg_collection.ep,
                )
            self.final_mixer = (
                FourStreamGatedResidual(architecture_config, combine=False)
                if is_mtp_layer or self.layer_number == config.num_layers
                else None
            )
            if config.perform_initialization:
                for module in (self.attention_residual, self.mlp_residual, self.ple):
                    if module is None:
                        continue
                    for child in module.modules():
                        if isinstance(child, torch.nn.Linear):
                            config.init_method(child.weight)
                if self.ple is not None:
                    self.ple.embedding.reset_parameters()
                if isinstance(self.attention, GatedDeltaNet):
                    for child in self.attention.modules():
                        if isinstance(child, torch.nn.Linear):
                            config.init_method(child.weight)
                    config.init_method(self.attention.conv1d.weight)

        def set_ple_input_ids(self, input_ids):
            self._ple_input_ids = input_ids

        @staticmethod
        def _add_bias(output):
            value, bias = output
            return value if bias is None else value + bias

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            context=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            inference_context=None,
            packed_seq_params=None,
            sequence_len_offset=None,
            padding_mask=None,
            **_kwargs,
        ):
            if hidden_states.size(-1) == architecture_config.hidden_size:
                hidden_states = hidden_states.repeat(1, 1, architecture_config.residual_streams)
            expected_width = architecture_config.hidden_size * architecture_config.residual_streams
            if hidden_states.size(-1) != expected_width:
                raise RuntimeError(
                    "pipeline tensor does not contain the configured packed GR streams"
                )
            if self.ple is not None:
                if self._ple_input_ids is None:
                    raise RuntimeError("Layer-2 PLE input IDs were not bound by the GPT adapter")
                hidden_states = hidden_states + self.ple(self._ple_input_ids, hidden_states)
            mixed, residual, injection = self.attention_residual(hidden_states)
            if self.attention_kind == "dense":
                branch = self._add_bias(
                    self.attention(
                        hidden_states=mixed,
                        attention_mask=attention_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        inference_context=inference_context,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                    )
                )
            else:
                branch = self.attention(mixed)
            hidden_states = FourStreamGatedResidual.inject(residual, branch, injection)
            mixed, residual, injection = self.mlp_residual(hidden_states)
            branch = self._add_bias(self.mlp(mixed, padding_mask=padding_mask))
            hidden_states = FourStreamGatedResidual.inject(residual, branch, injection)
            if self.final_mixer is not None:
                hidden_states = self.final_mixer(hidden_states)
            return hidden_states, context

    class QwenFlashNextGPT(GPTModel):
        def forward(self, input_ids, *model_args, **model_kwargs):
            if self.pre_process:
                for layer in self.decoder.layers:
                    if getattr(layer, "ple", None) is not None:
                        layer.set_ple_input_ids(input_ids)
            return super().forward(input_ids, *model_args, **model_kwargs)

    backend = TESpecProvider()
    attention_spec = ModuleSpec(
        module=SelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=backend.column_parallel_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            k_layernorm=IdentityOp,
        ),
    )
    moe_builder = partial(
        MoELayer,
        submodules=MoESubmodules(
            experts=SplitSwiGLUExperts,
            shared_experts=SplitSwiGLUSharedExpert,
        ),
    )
    layer_submodules = TransformerLayerSubmodules(
        self_attention=attention_spec,
        mlp=moe_builder,
    )
    layer_spec = ModuleSpec(
        module=QwenFlashNextLayer,
        submodules=layer_submodules,
    )

    def specs_for(config, vp_stage, pp_rank):
        local_layers = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
        block_spec = TransformerBlockSubmodules(
            layer_specs=[layer_spec] * local_layers,
            layer_norm=None,
        )
        mtp_spec = None
        if architecture_config.mtp_num_layers:
            mtp_spec = get_gpt_mtp_block_spec_for_backend(
                config=config,
                spec=block_spec,
                backend=backend,
                vp_stage=vp_stage,
                pp_rank=pp_rank,
            )
        return block_spec, mtp_spec

    return QwenFlashNextGPT, specs_for


def _loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Return Megatron's per-token ABI: summed loss, valid-token count, metrics.

    The native finalizer divides all gradients, including MoE/MTP auxiliary
    gradients, by the global token count. A legacy two-item, pre-averaged
    callback leaves that count at zero and breaks their relative scaling.
    """
    losses = output_tensor.reshape(-1).float()
    mask = loss_mask.reshape(-1).float()
    loss_sum = (losses * mask).sum()
    count = mask.sum(dtype=torch.int64)
    return loss_sum, count, {"lm loss": torch.stack((loss_sum.detach(), count))}


def _assert_pipeline_data_rank_layout(
    *,
    global_rank: int,
    data_parallel_rank: int,
    data_parallel_world_size: int,
    pipeline_global_ranks: tuple[int, ...],
) -> None:
    """Validate PP sample ownership without a collective inside the pipeline schedule."""
    if data_parallel_world_size < 1:
        raise RuntimeError("data-parallel world size must be positive")
    if len(pipeline_global_ranks) != 4 or global_rank not in pipeline_global_ranks:
        raise RuntimeError("the full-model data contract requires a four-rank pipeline group")
    projected_data_ranks = {rank % data_parallel_world_size for rank in pipeline_global_ranks}
    if projected_data_ranks != {data_parallel_rank}:
        raise RuntimeError("pipeline stages do not share one deterministic data rank")


def _forward_step(data_iterator, model, return_schedule_plan: bool = False):
    if return_schedule_plan:
        raise NotImplementedError("the full Qwen adapter does not use schedule plans")
    batch = next(data_iterator)
    tokens, labels = batch["tokens"], batch["labels"]
    positions = torch.arange(tokens.size(1), device=tokens.device).expand_as(tokens)
    loss_mask = labels.ne(-1).float()
    losses = model(
        tokens,
        positions,
        None,
        labels=labels,
        loss_mask=loss_mask,
    )
    return losses, partial(_loss_func, loss_mask)


def _invoke_pretrain(training_module, datasets_provider, model_provider, model_type) -> None:
    parameters = inspect.signature(training_module.pretrain).parameters
    if "cfg_container" in parameters:
        from megatron.training.argument_utils import pretrain_cfg_container_from_args
        from megatron.training.arguments import parse_and_validate_args

        parsed = parse_and_validate_args(args_defaults={"tokenizer_type": "NullTokenizer"})
        container = pretrain_cfg_container_from_args(parsed)
        training_module.pretrain(
            container, datasets_provider, model_provider, model_type, _forward_step
        )
    else:
        training_module.pretrain(
            datasets_provider,
            model_provider,
            model_type,
            _forward_step,
            args_defaults={"tokenizer_type": "NullTokenizer"},
        )


def _execute_checkpoint_request_by_local_rank(
    request: Any,
    *,
    local_rank: int,
    local_world_size: int,
    barrier: Any,
) -> None:
    """Execute one native DCP request at a time on each host.

    MCore's synchronous ``torch_dist`` writer stages every tensor owned by a
    rank to host memory before writing. The full PLE owns enough FP32 Adam
    state that staging all eight local owners concurrently exceeds the DLC
    node's host-memory limit. Planning, sharding, serialization, and
    finalization remain MCore-owned; only the per-host staging order is
    bounded here.
    """
    if local_world_size < 1 or not 0 <= local_rank < local_world_size:
        raise RuntimeError(
            f"invalid local checkpoint topology: rank={local_rank}, world={local_world_size}"
        )

    for writer_rank in range(local_world_size):
        if local_rank == writer_rank:
            call_args = list(request.async_fn_args)
            if request.preload_fn is not None:
                if len(call_args) != 3:
                    raise RuntimeError("native DCP writer changed its request ABI")
                preload = request.preload_fn
                if (
                    not isinstance(preload, partial)
                    or len(preload.args) != 2
                    or preload.args[1] is not True
                    or preload.keywords
                    or "non_blocking" not in inspect.signature(preload.func).parameters
                ):
                    raise RuntimeError("native DCP preload function changed its ABI")
                # The frozen PyTorch host allocator has no Python cache-release API.
                # Its nonblocking D2H path therefore retains every rank's staged
                # tensors in pinned memory until process exit. Use the same native
                # MCore preload implementation synchronously so GC can return the
                # pageable storage before the next local rank takes its turn.
                call_args[1] = preload.func(preload.args[0], non_blocking=False)
            if request.async_fn is not None:
                request.async_fn(*call_args, **request.async_fn_kwargs)
            del call_args
            gc.collect()
        barrier()

    for finalize_fn in request.finalize_fns:
        finalize_fn()


def _install_bounded_torch_dist_staging(training_module: Any) -> None:
    """Inject a native DCP strategy with serialized per-host tensor staging."""
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing.strategies.fully_parallel import (
        FullyParallelSaveStrategyWrapper,
    )
    from megatron.core.dist_checkpointing.strategies.torch import (
        TorchDistSaveShardedStrategy,
    )
    from megatron.training import get_args

    class _BoundedTorchDistSaveShardedStrategy(TorchDistSaveShardedStrategy):
        def save(self, sharded_state_dict, checkpoint_dir):
            request = self.async_save(sharded_state_dict, checkpoint_dir, async_strategy="mcore")
            _execute_checkpoint_request_by_local_rank(
                request,
                local_rank=int(os.environ["LOCAL_RANK"]),
                local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
                barrier=torch.distributed.barrier,
            )
            del request

    original_setup = training_module.setup_model_and_optimizer

    def setup_with_bounded_checkpoint_staging(*setup_args, **setup_kwargs):
        context = setup_kwargs.get("checkpointing_context")
        if context is None and len(setup_args) >= 3:
            context = setup_args[2]
        result = original_setup(*setup_args, **setup_kwargs)
        if context is None:
            raise RuntimeError("Megatron did not provide a checkpointing context")
        parsed = get_args()
        strategy: Any = _BoundedTorchDistSaveShardedStrategy(
            thread_count=parsed.dist_ckpt_workers,
            cpu_shm_mode=bool(getattr(parsed, "async_ckpt_use_cpu_shm", False)),
        )
        if parsed.ckpt_fully_parallel_save:
            strategy = FullyParallelSaveStrategyWrapper(
                strategy,
                parallel_state.get_data_parallel_group(with_context_parallel=True),
                parsed.ckpt_assume_constant_structure,
            )
        context["save_strategy"] = strategy
        return result

    training_module.setup_model_and_optimizer = setup_with_bounded_checkpoint_staging


def _write_contract(args, config) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    tokenizer_hash = _sha256(args.tokenizer / "tokenizer.json")
    config_hash = _sha256(args.tokenizer / "config.json")
    if tokenizer_hash != TOKENIZER_SHA256 or config_hash != SOURCE_CONFIG_SHA256:
        raise RuntimeError("pinned Qwen source/tokenizer hash drift")
    runtime = validate_runtime(require_pretrain=False)
    if args.model_variant == FULL_MODEL_VARIANT:
        model_name = "Qwen3.8-Flash-Next dense-attention owner-sharded-PLE variant"
        variant_differences = [
            "dense global attention at 2K instead of QSA",
            "three MTP depths sharing one physical layer",
            "Megatron-native Muon instead of private Canzona",
            "GPU-owner-sharded PLE instead of unpublished host prefetch",
        ]
    else:
        model_name = "Qwen3.8-Flash-Next quarter-shape depth-48 no-MTP variant"
        variant_differences = [
            "divisible width, head, expert, GR, and PLE shapes quartered",
            "all 48 backbone layers retained with an even PP4 split",
            "MTP module and auxiliary objective disabled",
            "dense global attention at 2K instead of QSA",
            "quarter-shape router stability coefficients use auxiliary 0.01 and z-loss 0.001",
            "Megatron-native Muon instead of private Canzona",
            "GPU-owner-sharded PLE instead of unpublished host prefetch",
        ]
    payload = {
        "model": model_name,
        "source": {
            "model": SOURCE_MODEL,
            "revision": SOURCE_REVISION,
            "config_sha256": SOURCE_CONFIG_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "weights_loaded": False,
            "scope": "from-scratch text-only pretraining",
        },
        "model_config": config.to_dict(),
        "parameter_count": parameter_count_contract(config),
        "variant_differences": variant_differences,
        "parallelism": {"tensor": 1, "pipeline": 4, "expert": 8, "context": 1},
        "pipeline_layers": list(config.pipeline_layers),
        "optimizer": {
            **_native_muon_contract(),
            "peak_lr": args.learning_rate,
            "minimum_lr": args.minimum_learning_rate,
            "warmup_fraction": args.warmup_fraction,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.clip_grad,
            "fc1_layout": "distinct native TE gate/up parameters",
            "ple_tables": "Adam, zero weight decay",
        },
        "checkpointing": {
            "format": "Megatron torch_dist",
            "serialization": "container-owned distributed checkpointing",
            "host_staging": "one local GPU rank per node at a time",
            "files_per_rank": CHECKPOINT_WRITER_THREADS,
            "reason": "bound owner-sharded PLE optimizer staging to host memory",
        },
        "training": {
            "loss_normalization": LOSS_NORMALIZATION,
            "seed": args.seed,
            "sequence_length": config.sequence_len,
            "micro_batch_sequences": args.micro_batch_size,
            "global_batch_sequences": args.global_batch_size,
            "tokens_per_step": args.global_batch_size * config.sequence_len,
            "train_steps": args.probe_steps or TRAIN_STEPS,
            "target_tokens": args.target_train_tokens,
            "effective_tokens": (
                args.probe_steps * args.global_batch_size * config.sequence_len
                if args.probe_steps
                else EFFECTIVE_TOKENS
            ),
            "checkpoint_interval_steps": (
                args.probe_save_interval or args.probe_steps
                if args.probe_steps
                else CHECKPOINT_INTERVAL_STEPS
            ),
        },
        "data": {
            "root": str(args.data_root),
            "manifest_sha256": _sha256(args.data_root / "DATA_READY.json"),
            "sharding": "data-parallel rank; identical across TP/PP/EP/CP ranks",
        },
        "precision": "BF16 model/compute; FP32 optimizer and Muon orthogonalization",
        "source_commit": os.environ.get("NGA_EXPECTED_COMMIT"),
        "implementation_sha256": {
            "architecture": _sha256(Path(inspect.getfile(Qwen38FlashNextFullConfig)).resolve()),
            "trainer": _sha256(Path(__file__).resolve()),
        },
        "runtime": runtime,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "created_at_unix": time.time(),
    }
    contract = args.run_dir / "RUN_CONTRACT.json"
    if contract.exists() and not args.resume:
        raise RuntimeError("run directory already contains a contract")
    if contract.exists():
        previous = json.loads(contract.read_text())
        if previous.get("training", {}).get("loss_normalization") != LOSS_NORMALIZATION:
            raise RuntimeError("loss normalization changed; use a fresh run directory and weights")
    if not contract.exists():
        _atomic_json(contract, payload)
    _atomic_json(args.run_dir / "contracts" / f"attempt-{time.time_ns()}.json", payload)


def _current_iteration() -> int:
    from megatron.training import get_args

    parsed = get_args()
    return int(getattr(parsed, "curr_iteration", getattr(parsed, "iteration", 0)))


def _run(args: argparse.Namespace) -> None:
    if args.model_variant == FULL_MODEL_VARIANT:
        config = Qwen38FlashNextFullConfig(sequence_len=args.sequence_length)
    else:
        config = Qwen38FlashNextFullConfig.quarter_depth48_no_mtp()
    train_prefixes, validation_prefixes = _validated_data_prefixes(args.data_root)
    _write_contract(args, config)
    sys.argv = _megatron_argv(args, config)

    import megatron.training.training as training_module
    from megatron.core import parallel_state
    from megatron.core.enums import ModelType
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.utils import get_pg_rank
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    model_class, specs_for = _build_model_classes(config)
    probe_gradient_state = {"expected": False, "seen": False, "nonfinite": False}

    def model_provider(
        pre_process=True,
        post_process=True,
        vp_stage=None,
        config=None,
        pg_collection=None,
    ):
        transformer_config = config or core_transformer_config_from_args(get_args())
        # The legacy CLI hard-codes this field to false and exposes no positive
        # flag. Dynamic P2P shape exchange is required because inter-stage GR
        # tensors have width 4H while Megatron's language-model width remains H.
        transformer_config.variable_seq_lengths = True
        # Layer 2 alone owns PLE and every fourth layer changes attention type,
        # so checkpoint keys must retain their global layer number.
        transformer_config.hetereogenous_dist_checkpoint = True
        groups = pg_collection or ProcessGroupCollection.use_mpu_process_groups()
        pp_rank = get_pg_rank(groups.pp)
        block_spec, mtp_spec = specs_for(transformer_config, vp_stage, pp_rank)
        model = model_class(
            config=transformer_config,
            transformer_layer_spec=block_spec,
            vocab_size=config_outer.vocab_size,
            max_sequence_length=config_outer.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope",
            rotary_percent=config_outer.partial_rotary_factor,
            rotary_base=int(config_outer.rope_theta),
            mtp_block_spec=mtp_spec,
            pg_collection=groups,
            vp_stage=vp_stage,
        )
        partition = _tag_native_optimizer_fallbacks(model)
        if args.probe_steps and pp_rank == 0:
            early_parameter = next(
                (
                    parameter
                    for name, parameter in model.named_parameters()
                    if name.endswith("attention.in_proj_qkv.weight")
                ),
                None,
            )
            if early_parameter is None:
                raise RuntimeError("the Flash-Next probe could not identify an early GDN parameter")
            probe_gradient_state["expected"] = True

            def record_early_gradient(gradient):
                if not torch.isfinite(gradient).all():
                    probe_gradient_state["nonfinite"] = True
                elif torch.count_nonzero(gradient):
                    probe_gradient_state["seen"] = True
                return gradient

            early_parameter.register_hook(record_early_gradient)
        if torch.distributed.get_rank() == 0:
            _atomic_json(args.run_dir / "OPTIMIZER_PARTITION.json", partition)
        return model

    config_outer = config

    def datasets_provider(_sample_counts):
        dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
        dp_world = parallel_state.get_data_parallel_world_size(with_context_parallel=True)
        _assert_pipeline_data_rank_layout(
            global_rank=torch.distributed.get_rank(),
            data_parallel_rank=dp_rank,
            data_parallel_world_size=dp_world,
            pipeline_global_ranks=tuple(
                torch.distributed.get_process_group_ranks(
                    parallel_state.get_pipeline_model_parallel_group()
                )
            ),
        )
        train = partition_prefixes_for_dp_rank(train_prefixes, dp_rank, dp_world)
        validation = partition_prefixes_for_dp_rank(
            validation_prefixes, dp_rank, dp_world, require_distinct=False
        )
        accumulation = args.global_batch_size // (dp_world * args.micro_batch_size)

        def train_batches():
            yield from DPRankTokenBatches(
                train,
                batch_size=args.micro_batch_size,
                sequence_len=config.sequence_len,
                start_batch=_current_iteration() * accumulation,
                device=torch.device("cuda", torch.cuda.current_device()),
            )

        def validation_batches():
            window = args.eval_iters * accumulation
            yield from DPRankTokenBatches(
                validation,
                batch_size=args.micro_batch_size,
                sequence_len=config.sequence_len,
                start_batch=dp_rank * window,
                repeat_window_batches=window,
                device=torch.device("cuda", torch.cuda.current_device()),
            )

        return train_batches(), validation_batches(), None

    datasets_provider.is_distributed = True
    _install_bounded_torch_dist_staging(training_module)
    _invoke_pretrain(
        training_module,
        datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
    )
    if args.probe_steps:
        gradient_evidence = torch.tensor(
            [
                int(probe_gradient_state["expected"]),
                int(probe_gradient_state["seen"]),
                int(probe_gradient_state["nonfinite"]),
            ],
            device=torch.device("cuda", torch.cuda.current_device()),
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(gradient_evidence)
        expected, seen, nonfinite = gradient_evidence.tolist()
        if expected != 8 or seen != expected or nonfinite:
            raise RuntimeError(
                "Flash-Next probe early-backbone gradient rejection: "
                f"expected={expected}, seen={seen}, nonfinite={nonfinite}"
            )
        if torch.distributed.get_rank() == 0:
            _atomic_json(
                args.run_dir / "PROBE_GRADIENTS.json",
                {
                    "status": "passed",
                    "early_gdn_owner_ranks": expected,
                    "nonzero_gradient_ranks": seen,
                    "nonfinite_gradient_ranks": nonfinite,
                },
            )
    if torch.distributed.get_rank() == 0:
        _atomic_json(
            args.run_dir
            / ("PROBE_COMPLETE.json" if args.probe_steps else "TRAINING_COMPLETE.json"),
            {"iteration": _current_iteration(), "completed_at_unix": time.time()},
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--model-variant",
        choices=(FULL_MODEL_VARIANT, QUARTER_DEPTH48_NO_MTP_MODEL_VARIANT),
        default=FULL_MODEL_VARIANT,
    )
    parser.add_argument("--sequence-length", type=int, default=2_048)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=4_096)
    parser.add_argument("--target-train-tokens", type=int, default=EFFECTIVE_TOKENS)
    parser.add_argument("--learning-rate", type=float, default=1.76e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.76e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=1_192)
    parser.add_argument("--eval-iters", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--probe-save-interval", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (
        min(
            args.sequence_length,
            args.micro_batch_size,
            args.global_batch_size,
            args.target_train_tokens,
            args.eval_interval,
            args.eval_iters,
            args.log_interval,
        )
        < 1
    ):
        raise SystemExit("all integer training controls must be positive")
    if args.sequence_length != 2_048:
        raise SystemExit("the supported training recipe is fixed at 2,048 tokens")
    if args.probe_steps < 0 or args.probe_save_interval < 0:
        raise SystemExit("probe controls must be non-negative")
    for path in (
        args.data_root / "DATA_READY.json",
        args.tokenizer / "tokenizer.json",
        args.tokenizer / "config.json",
    ):
        if not path.is_file():
            raise SystemExit(f"required artifact is missing: {path}")
    ready = json.loads((args.data_root / "DATA_READY.json").read_text())
    if ready.get("tokenizer_sha256") != TOKENIZER_SHA256:
        raise SystemExit("FineWeb-Edu data tokenizer hash does not match Qwen3.8")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "checkpoints").mkdir(exist_ok=True)
    _run(args)


if __name__ == "__main__":
    main()
