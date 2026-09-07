"""Shared, already-qualified construction for the pretrained AutoModel execution boundary."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.qwen3_8_flash_next.config import (
    Qwen3_8_FlashNextConfig, Qwen3_8_FlashNextTextConfig,
)
from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration
from nemo_automodel.components.moe.parallelizer import parallelize_model
from archlab.automodel.loading import (
    audit_checkpoint_keys, rebuild_nonpersistent_buffers,
    poison_weights_before_load, assert_loaded_weights_finite,
)


def emit(event: str, **values):
    print(json.dumps({"event": event, "rank": dist.get_rank(), "time_unix": time.time(), **values},
                     sort_keys=True), flush=True)


def tiny_config(num_experts=8):
    text = Qwen3_8_FlashNextTextConfig(
        vocab_size=1024, hidden_size=256, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        moe_intermediate_size=128, shared_expert_intermediate_size=128,
        num_experts=num_experts, num_experts_per_tok=2, hc_count=4, hc_lowrank=32,
        ple_layer_ids=[], indexer_budget=32, indexer_n_heads=2, indexer_head_dim=32,
        indexer_compress_ratio=4, linear_num_key_heads=4, linear_num_value_heads=8,
        linear_key_head_dim=32, linear_value_head_dim=32, max_position_embeddings=32768,
        dtype="bfloat16", rope_parameters={"rope_type": "default", "rope_theta": 10000000.,
                                           "partial_rotary_factor": .25},
    )
    return Qwen3_8_FlashNextConfig(text_config=text, language_model_only=True)


def build_frozen_base(config, *, tiny: bool, checkpoint: Path | None, ep_size: int, activation_checkpointing: bool):
    device = torch.device("cuda", torch.cuda.current_device())
    precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                     output_dtype=torch.bfloat16)
    setup = DistributedSetup.build(strategy=FSDP2Config(mp_policy=precision),
                                   parallelism_sizes=ParallelismSizes(ep_size=ep_size),
                                   world_size=dist.get_world_size())
    mesh = setup.mesh_context
    emit("parallel_mesh", device_mesh_names=list(mesh.device_mesh.mesh_dim_names),
         device_mesh_ranks=mesh.device_mesh.mesh.tolist(),
         moe_mesh_names=list(mesh.moe_mesh.mesh_dim_names), moe_mesh_ranks=mesh.moe_mesh.mesh.tolist())
    backend = BackendConfig(attn="flex", linear="torch", rms_norm="torch_fp32",
                            experts="torch_mm", dispatcher="deepep", rope_fusion=False,
                            gate_precision="float32", enable_hf_state_dict_adapter=not tiny)
    with torch.device("cpu" if tiny else "meta"):
        model = Qwen3_8_FlashNextForConditionalGeneration(
            config, backend=backend, moe_overrides={"aux_loss_coeff": 0.0})
    if tiny:
        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
        model.to(device)
    else:
        emit("checkpoint_key_audit", **audit_checkpoint_keys(model, checkpoint))
        # The ordinary initialize_weights() path also does this cast. Skipping
        # random initialization must not leave GroupedExperts' FP32 allocation
        # dtype as an accidental frozen master copy of BF16 checkpoint weights.
        cast_model_to_dtype(model, torch.bfloat16, skip_modules=("_fp32_params",))
        rebuild_nonpersistent_buffers(model, device)
    model.requires_grad_(False)
    parallelize_model(model, mesh.device_mesh, mesh.moe_mesh,
                      **mesh.parallelize_axis_kwargs(), activation_checkpointing=activation_checkpointing,
                      reshard_after_forward=True, mp_policy=precision,
                      reapply_trainability=lambda m: m.requires_grad_(False))
    if not tiny:
        # Use the upstream sharded loader, not a handwritten tensor conversion.
        model._skip_init_weights_on_load = True
        Checkpointer.initialize_model_weights(model, device)
        poison_weights_before_load(model)
        checkpointer = CheckpointingConfig(checkpoint_dir="", model_repo_id=str(checkpoint),
                                          save_consolidated=False, dequantize_base_checkpoint=False).build(
            dp_rank=dist.get_rank(), tp_rank=0, pp_rank=0, moe_mesh=mesh.moe_mesh)
        emit("base_load_begin")
        checkpointer.load_base_model(model, device, None, str(checkpoint))
        emit("base_load_end", local_elements=assert_loaded_weights_finite(model))
    model.requires_grad_(False)
    return model, mesh, precision

