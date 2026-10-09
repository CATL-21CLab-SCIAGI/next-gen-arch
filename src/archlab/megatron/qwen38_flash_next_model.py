"""Native Megatron model adapter shared by training, sampling and probes."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

from archlab.architectures.qwen38_flash_next_full import (
    FourStreamGatedResidual,
    GatedDeltaNet,
    Qwen38FlashNextFullConfig,
)
from archlab.megatron.ple_checkpoint import DistributedPLE


def _assert_dp_only_groups(groups) -> dict[str, int]:
    """Reject model/sequence/expert sharding before the first forward pass."""
    sizes = {
        name: getattr(groups, name).size()
        for name in ("tp", "pp", "ep", "cp", "expt_tp", "dp", "expt_dp")
    }
    if any(sizes[name] != 1 for name in ("tp", "pp", "ep", "cp", "expt_tp")):
        raise RuntimeError(f"DP-only execution received model-parallel groups: {sizes}")
    world = torch.distributed.get_world_size()
    if sizes["dp"] != world or sizes["expt_dp"] != world:
        raise RuntimeError(
            f"DP-only execution requires complete replicas on all {world} ranks: {sizes}"
        )
    return sizes


def _tag_native_optimizer_fallbacks(model: torch.nn.Module) -> dict[str, int]:
    counts = {"muon": 0, "adamw": 0, "ple_adam_no_decay": 0}
    for name, parameter in model.named_parameters():
        if ".mlp.experts." in name or ".embedding.tables." in name:
            # Keep the native checkpoint's dense/routed optimizer-group layout.
            # This flag selects expt_dp, NOT EP communication or model sharding.
            # Under DP-only, _assert_dp_only_groups requires expt_dp == DP == WORLD.
            parameter.allreduce = False
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
                    replica_rank=get_pg_rank(pg_collection.expt_dp),
                )
            self.final_mixer = (
                FourStreamGatedResidual(architecture_config, combine=False)
                if is_mtp_layer or self.layer_number == config.num_layers
                else None
            )
            if config.perform_initialization:
                modules_to_initialize = (self.attention_residual, self.mlp_residual, self.ple)
                if architecture_config.zero_centered_gamma:
                    modules_to_initialize += (self.final_mixer,)
                for module in modules_to_initialize:
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
    qkv_projection = backend.column_parallel_linear()
    if architecture_config.attention_output_gate:
        from archlab.megatron.gated_qkv import SplitGatedQKV

        qkv_projection = SplitGatedQKV
    attention_spec = ModuleSpec(
        module=SelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=qkv_projection,
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=None if architecture_config.qk_layernorm else IdentityOp,
            k_layernorm=None if architecture_config.qk_layernorm else IdentityOp,
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


def build_model(
    architecture_config,
    transformer_config,
    groups,
    *,
    pre_process=True,
    post_process=True,
    vp_stage=None,
):
    """Shared native model construction for training and checkpoint sampling."""
    from megatron.core.utils import get_pg_rank

    model_class, specs_for = _build_model_classes(architecture_config)
    transformer_config.variable_seq_lengths = len(architecture_config.pipeline_layers) > 1
    transformer_config.hetereogenous_dist_checkpoint = True
    transformer_config.attention_output_gate = architecture_config.attention_output_gate
    transformer_config.qk_layernorm = architecture_config.qk_layernorm
    transformer_config.layernorm_zero_centered_gamma = architecture_config.zero_centered_gamma
    block_spec, mtp_spec = specs_for(transformer_config, vp_stage, get_pg_rank(groups.pp))
    return model_class(
        config=transformer_config,
        transformer_layer_spec=block_spec,
        vocab_size=architecture_config.vocab_size,
        max_sequence_length=architecture_config.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_percent=architecture_config.partial_rotary_factor,
        rotary_base=int(architecture_config.rope_theta),
        mtp_block_spec=mtp_spec,
        pg_collection=groups,
        vp_stage=vp_stage,
    )
