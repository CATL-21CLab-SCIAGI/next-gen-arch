"""Miles V4.1 extensions for the matched, fully fine-tuned archlab parents."""

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

# TileLang 0.1.12 moved the target helpers; tile_kernels 1.0 still
# imports the former module name. Alias the unchanged public implementation.
import tilelang.backend.target as tilelang_target
import torch
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.module import mark_keep_in_fp32

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter
from archlab.architectures.deepseek_v41_normal_adapter import V41NormalAttentionAdapter

# The pinned Miles plugin imports the pre-0.1.12 TileLang module name.
sys.modules.setdefault("tilelang.utils.target", tilelang_target)
from miles_plugins.models.deepseek_v41 import deepseek_v41 as native  # noqa: E402
from miles_plugins.models.deepseek_v41.engram import DeepSeekV41Engram, engram_gate  # noqa: E402


class TrainableEngram(DeepSeekV41Engram):
    """BF16 row-sharded table; retain gradients through the embedding lookup."""

    def _load_table(self):
        self.shared_table = False
        self.table = torch.nn.Parameter(torch.empty(
            self.rows_local, self.head_dim, dtype=torch.bfloat16,
            device=torch.cuda.current_device()),
            requires_grad=os.environ.get("ARCHLAB_RL_FREEZE_ENGRAM") != "1")
        set_tensor_model_parallel_attributes(self.table, True, 0, 1)
        self.table.is_embedding_or_output_parameter = True
        self.table.sequence_parallel = False
        self._table_loaded = True

    def lookup(self, ids):
        local = ids - self.row_start
        owned = (local >= 0) & (local < self.rows_avail)
        values = torch.nn.functional.embedding(local.masked_fill(~owned, 0), self.table)
        values = values.masked_fill(~owned.unsqueeze(-1), 0)
        return reduce_from_tensor_model_parallel_region(values, group=self.tp_group)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        result = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        result.update(make_sharded_tensors_for_checkpoint(
            {"table": self.table}, prefix, {"table": 0}, sharded_offsets,
            tp_group=self.tp_group, dp_cp_group=metadata["dp_cp_group"]))
        return result

    def forward(self, hidden_states, input_ids):
        if self.cp_size != 1:
            raise ValueError("BF16 Engram admission requires CP1")
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.tp_group)
        s, b, _ = hidden_states.shape
        if input_ids.shape != (b, s):
            raise ValueError("Engram requires one complete un-packed sequence per batch row")
        ids = self.hash_ids(input_ids).transpose(0, 1)
        kv, _ = self.linear_wkv(self.lookup(ids).flatten(-2))
        x = hidden_states.view(s, b, self.hc_mult, self.dim)
        output = engram_gate(x, kv, self.q_weight, self.k_weight, self.eps,
                            self.clamp_value).reshape_as(hidden_states)
        return scatter_to_sequence_parallel_region(output, group=self.tp_group) if self.sequence_parallel else output


class AdaptedHyperConnection(native.V41HyperConnection):
    def fused_h_res_h_post_bda(self, *args, **kwargs):
        output = super().fused_h_res_h_post_bda(*args, **kwargs)
        if not hasattr(self, "archlab_adapter"):
            return output
        if self.config.sequence_parallel:
            output = gather_from_sequence_parallel_region(
                output, tensor_parallel_output_grad=False, group=self.archlab_tp_group)
        s, b, _ = output.shape
        streams = output.reshape(s, b, self.n, self.hidden_size).transpose(0, 1)
        output = self.archlab_adapter(streams).transpose(0, 1).reshape_as(output)
        return scatter_to_sequence_parallel_region(output, group=self.archlab_tp_group) if self.config.sequence_parallel else output


class AdaptedLayer(native.V41TransformerLayer):
    def __init__(self, config, submodules, layer_number=1, **kwargs):
        super().__init__(config, submodules, layer_number=layer_number, **kwargs)
        if self.layer_number - 1 in config.archlab_metadata["adapter_layers"]:
            variant = config.archlab_metadata["variant"]
            cls = V41NormalAttentionAdapter if variant == "normal" else V41SimplicialAdapter
            with torch.device("cpu"):
                adapter = cls(V41AdapterConfig(width=config.hidden_size,
                                              streams=config.num_residual_streams))
            adapter.to(device=torch.cuda.current_device(), dtype=torch.float32)
            for p in adapter.parameters():
                mark_keep_in_fp32(p)
                p.sequence_parallel = False
            self.self_attention_hyper_connection.add_module("archlab_adapter", adapter)
            self.self_attention_hyper_connection.archlab_tp_group = self.pg_collection.tp
        self.mlp.router.weight.requires_grad_(False)
        config.moe_router_bias_update_rate = 0.0
        torch.backends.cuda.matmul.allow_tf32 = False


def get_spec(args, config, vp_stage=None):
    if config.experimental_attention_variant != "dsv4" or not config.enable_hyper_connections:
        raise ValueError("matched V4.1 requires Miles sparse attention and hyper-connections")
    config.archlab_metadata = json.loads((Path(args.hf_checkpoint) / "config.json").read_text())["archlab"]
    native.DeepSeekV41Engram = TrainableEngram
    spec = native.get_dsv41_spec(args, config, vp_stage)
    for layer in spec.layer_specs:
        layer.module = AdaptedLayer
        layer.submodules.self_attention_hyper_connection = AdaptedHyperConnection
    return spec


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    from copy import copy

    from megatron.training import get_args
    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    args = copy(get_args())
    args.custom_model_provider_path = None
    args.spec = ["archlab.megatron.miles_v41_model", "get_spec"]
    region = nullcontext()
    if os.environ.get("EVERGREENTREE_WEIGHT_CACHE_DIR"):
        from torch_memory_saver import torch_memory_saver
        region = torch_memory_saver.region(tag="evergreentree_model", enable_cpu_backup=False,
                                          enable_disk_backup=False)
    with region:
        model = get_model_provider_func(args)(pre_process, post_process, vp_stage)
        if post_process:
            model.output_layer.weight.data = model.output_layer.weight.data.float()
            mark_keep_in_fp32(model.output_layer.weight)
            original_forward = model.output_layer.forward

            def fp32_head(input_, *args, **kwargs):
                return original_forward(input_.float(), *args, **kwargs)

            model.output_layer.forward = fp32_head
    by_dtype = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        by_dtype[key] = by_dtype.get(key, 0) + parameter.numel() * parameter.element_size()
    print(json.dumps({"event": "EvergreenTree_model_constructed", "rank": torch.distributed.get_rank(),
                      "parameter_bytes": by_dtype,
                      "fp32_gradient_accumulation": args.accumulate_allreduce_grads_in_fp32}), flush=True)
    return model
