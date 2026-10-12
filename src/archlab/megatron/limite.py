"""Native Limite provider for verl's upstream NeMo/Megatron engine.

Only data parallelism is supported. Native attention and mixed parameter dtypes
remain owned by Limite; using Megatron here does not imply TE transformer layers.
Import this module through verl's ``model.external_lib`` to register the bridge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.core.enums import ModelType
from megatron.core.transformer.module import MegatronModule

from archlab.training.limite import LimiteConfig, LimiteForCausalLM


class NativeLimiteModule(MegatronModule):
    """Adapt the upstream engine's tensor return contract without replacing layers."""

    supports_chunked_logits = True

    def __init__(self, config):
        super().__init__(config=config)
        self.policy = LimiteForCausalLM(LimiteConfig.from_dict(config.native_config))
        self.pre_process = self.post_process = True
        self.model_type = ModelType.encoder_or_decoder
        self.share_embeddings_and_output_weights = (
            self.policy.get_input_embeddings().weight is self.policy.get_output_embeddings().weight
        )
        if config.activation_checkpointing:
            self.policy.gradient_checkpointing_enable({"use_reentrant": False})

    def set_input_tensor(self, input_tensor):
        if input_tensor is not None and not (
            isinstance(input_tensor, list) and len(input_tensor) == 1 and input_tensor[0] is None
        ):
            raise ValueError("native Limite provider supports data parallelism only")

    def shared_embedding_or_output_weight(self):
        return self.policy.get_input_embeddings().weight

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                packed_seq_params=None, logits_processor=None,
                logits_processor_args=None, **kwargs):
        if packed_seq_params is not None:
            raise ValueError("native Limite requires use_remove_padding=False; packed documents are unsupported")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("native Limite requires one unpacked sequence per microbatch")
        if kwargs:
            raise ValueError(f"unsupported native Limite forward options: {sorted(kwargs)}")
        if logits_processor is not None:
            from archlab.architectures.limite_head import chunked_logits_processor

            publisher = self.policy.policy
            if self.policy.config.conversion:
                attention_mask, position_ids = self.policy.conversion_inputs(
                    input_ids, attention_mask, position_ids,
                )
            hidden = publisher.model(input_ids=input_ids, attention_mask=attention_mask,
                                     position_ids=position_ids, use_cache=False,
                                     return_dict=True).last_hidden_state
            return chunked_logits_processor(publisher._softcapped_logits, hidden,
                                            logits_processor, logits_processor_args or {})
        if logits_processor_args is not None:
            raise ValueError("logits_processor_args requires logits_processor")
        return self.policy(input_ids=input_ids, attention_mask=attention_mask,
                           position_ids=position_ids, use_cache=False).logits


@dataclass
class NativeLimiteProvider(TransformerConfig, ModelProviderMixin[NativeLimiteModule]):
    native_config: dict = field(default_factory=dict, repr=False)
    seq_length: int = 131072
    activation_checkpointing: bool = True

    def finalize(self):
        for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size",
                     "context_parallel_size", "expert_model_parallel_size"):
            if getattr(self, name, 1) != 1:
                raise ValueError(f"native Limite supports {name}=1 only")
        if self.expert_tensor_parallel_size not in (None, 1):
            raise ValueError("native Limite supports expert_tensor_parallel_size=1 only")
        if self.virtual_pipeline_model_parallel_size is not None or self.sequence_parallel:
            raise ValueError("native Limite does not support virtual or sequence parallelism")
        if self.fp8 or self.fp4 or getattr(self, "mtp_num_layers", None):
            raise ValueError("native Limite has no qualified FP8, FP4 or MTP implementation")
        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        if pre_process is False or post_process is False or vp_stage not in (None, 0):
            raise ValueError("native Limite requires a complete model on each data parallel rank")
        return NativeLimiteModule(self)

    def provide_distributed_model(self, *args, **kwargs):
        if kwargs.get("use_megatron_fsdp") or kwargs.get("use_torch_fsdp2"):
            raise ValueError("native Limite currently supports Megatron DDP only")
        ddp_config = kwargs.get("ddp_config", args[0] if args else None)
        if (kwargs.get("wrap_with_ddp", True) and ddp_config is not None
                and not ddp_config.use_distributed_optimizer):
            raise ValueError("native mixed-dtype checkpointing requires use_distributed_optimizer=True")
        # Native FP32 normalization/MUDD parameters must not be rounded to BF16.
        # MCore's optimizer supports native FP32 and BF16 master-copy groups.
        kwargs["mixed_precision_wrapper"] = None
        return super().provide_distributed_model(*args, **kwargs)


@MegatronModelBridge.register_bridge(
    source="ArchlabLimiteForCausalLM", target=NativeLimiteModule,
    provider=NativeLimiteProvider, model_type="archlab_limite",
)
class NativeLimiteBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):
        config = hf_pretrained.config
        return NativeLimiteProvider(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
            kv_channels=config.head_dim,
            ffn_hidden_size=config.intermediate_size,
            seq_length=config.max_position_embeddings,
            native_config=config.to_dict(),
            hidden_dropout=0., attention_dropout=0.,
            gradient_accumulation_fusion=False,
        )

    def mapping_registry(self):
        # The extra prefix belongs only to this engine wrapper. No QKV fusion,
        # transposition, padding, cast or architecture reinterpretation occurs.
        backbone = "model" if self.hf_config.conversion else "model.base"
        return MegatronMappingRegistry(
            # safetensors stores the tied pair once under lm_head, while
            # named_parameters visits the embedding alias first.
            ReplicatedMapping(f"policy.policy.{backbone}.embed_tokens.weight", "policy.lm_head.weight"),
            ReplicatedMapping("policy.**", "**"),
        )

    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        tasks = super().build_conversion_tasks(hf_pretrained, megatron_model)
        # Upstream's generic converter warns and skips unknown/missing weights.
        # This complete DP-only model has no intentional partial conversions.
        if not tasks or any(task is None for task in tasks):
            raise ValueError("native Limite conversion omitted one or more model tensors")
        if hasattr(hf_pretrained, "state"):
            expected = set(hf_pretrained.state.source.get_all_keys())
            mapped = {task.mapping.hf_param for task in tasks}
            if expected != mapped:
                raise ValueError(f"native Limite tensor inventory differs: {sorted(expected ^ mapped)}")
        return tasks

    @classmethod
    def megatron_to_hf_config(cls, provider):
        return dict(provider.native_config)
