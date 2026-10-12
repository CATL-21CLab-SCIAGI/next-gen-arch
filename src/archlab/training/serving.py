"""Narrow vLLM boundary: native normal Limite with a separate cache per prelude.

The package owns paging, scheduling and attention kernels. Publisher projections,
RMSNorm, RoPE, gates, value embeddings, MUDD and sigmoid logits stay unchanged.
Tensor/pipeline parallelism and the simplicial short-axis cache are not admitted.
"""
from __future__ import annotations

from types import MethodType

from torch import nn

from archlab.architectures.limite_bindings import bind_native_forward
from archlab.training.limite import LimiteForCausalLM


class _AttentionInterface:
    def get_interface(self, name, fallback):
        return _paged_attention


def _paged_attention(module, query, key, value, attention_mask, *, scaling, **kwargs):
    batch, heads, tokens, dim = query.shape
    if batch != 1 or scaling != module.scaling:
        raise ValueError("vLLM Limite expects one packed token stream and native scaling")
    q, k, v = (tensor.transpose(1, 2).reshape(tokens, -1) for tensor in (query, key, value))
    output = module.archlab_paged_attention(q, k, v)
    return output.view(1, tokens, heads, dim), None


def bind_normal_attention(policy, attention_factory, *, prefix=""):
    """Bind one independent package cache to each of the 2×N native attentions."""
    if policy.model.adapter_config["variant"] != "normal":
        raise ValueError("simplicial paging requires its own short-axis cache implementation")
    backbone = policy.model
    attentions = []
    for index, (adapter, layer) in enumerate(zip(backbone.adapters, backbone.base.layers, strict=True)):
        for kind, attention in (("prelude", adapter.native), ("backbone", layer.self_attn)):
            attention.archlab_paged_attention = attention_factory(
                attention, prefix=f"{prefix}layers.{index}.{kind}.attention")
            # Always bind the publisher's original function, not the old
            # TileLang reduction. No parent library or global registry patch.
            forward = bind_native_forward(
                type(attention).forward,
                (("ALL_ATTENTION_FUNCTIONS", _AttentionInterface()),),
                "archlab_vllm_normal",
            )
            attention.forward = MethodType(forward, attention)
            attention.config._attn_implementation = "archlab_paged"
            attentions.append(attention)
    # Paged Attention owns masks and cached K/V. Avoid constructing dense masks
    # for a packed stream of independent requests; positions come from vLLM.
    backbone._masks = MethodType(
        lambda self, hidden, positions, mask, cache: dict(full_attention=None, sliding_attention=None),
        backbone,
    )
    return tuple(attentions)


class NormalLimiteForCausalLM(nn.Module):
    """Public vLLM model extension, restricted to one GPU per replica."""

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        from vllm.model_executor.layers.attention import Attention

        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Limite serving currently requires TP=PP=1")
        if vllm_config.quant_config is not None:
            raise ValueError("Limite low-bit serving has not passed numerical qualification")
        config = vllm_config.model_config.hf_config
        wrapper = LimiteForCausalLM(config)
        self.policy = wrapper.policy
        self.config = config

        def factory(attention, *, prefix):
            return Attention(
                num_heads=attention.num_heads, head_size=attention.head_dim,
                scale=attention.scaling, num_kv_heads=attention.num_kv_heads,
                cache_config=vllm_config.cache_config,
                per_layer_sliding_window=attention.window_span, prefix=prefix,
            )

        self._attention_count = len(bind_normal_attention(self.policy, factory, prefix=prefix))

    @property
    def model(self):
        """Expose the real backbone for package introspection without an alias.

        A property avoids registering it twice, which would duplicate checkpoint
        names and break the strict publisher weight-loading contract.
        """
        return self.policy.model

    def embed_input_ids(self, input_ids):
        return self.policy.get_input_embeddings()(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if intermediate_tensors is not None or inputs_embeds is not None or input_ids is None:
            raise ValueError("native Limite requires token IDs for its value embeddings")
        if input_ids.ndim != 1 or positions.ndim != 1:
            raise ValueError("vLLM must supply flat token IDs and positions")
        output = self.policy.model(input_ids=input_ids[None], position_ids=positions[None],
                                   use_cache=False, return_dict=True)
        return output.last_hidden_state.squeeze(0)

    def compute_logits(self, hidden_states):
        return self.policy._softcapped_logits(hidden_states)

    def load_weights(self, weights):
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        parameters = dict(self.named_parameters(remove_duplicate=False))
        loaded = set()
        for name, tensor in weights:
            if name not in parameters:
                raise ValueError("unknown Limite serving weight: " + name)
            parameter = parameters[name]
            if parameter.dtype != tensor.dtype or parameter.shape != tensor.shape:
                raise ValueError("Limite serving weight shape/dtype mismatch: " + name)
            default_weight_loader(parameter, tensor)
            loaded.update(key for key, value in parameters.items() if value is parameter)
        # Rebuild every folded projection after both the first load and updates.
        self.policy.train()
        self.policy.eval()
        return loaded


def register():
    """Call in each serving process through the vLLM general plugin mechanism."""
    from vllm import ModelRegistry

    if "ArchlabLimiteForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("ArchlabLimiteForCausalLM", "archlab.training.serving:NormalLimiteForCausalLM")


class QualificationWorker:
    """Trusted file-based weight reload for the package worker parity test."""

    def archlab_initialize_verl_model(self, vocab_size):
        """Exercise the exact stock rollout initializer on the real worker model."""
        import torch
        from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

        model = self.model_runner.model
        inventory = {name: (id(value), str(value.dtype))
                     for name, value in model.named_parameters(remove_duplicate=False)}
        weight = model.policy.get_output_embeddings().weight
        hidden = torch.linspace(-1., 1., weight.shape[1], device=weight.device,
                                dtype=weight.dtype)[None]
        with torch.inference_mode():
            expected = model.compute_logits(hidden)
        extension = object.__new__(vLLMColocateWorkerExtension)
        extension.model_runner = self.model_runner
        extension.monkey_patch_model(vocab_size=vocab_size, banned_token_ids=None)
        assert inventory == {name: (id(value), str(value.dtype))
                             for name, value in model.named_parameters(remove_duplicate=False)}
        with torch.inference_mode():
            actual = model.compute_logits(hidden)
        torch.testing.assert_close(actual[..., :vocab_size], expected[..., :vocab_size], rtol=0, atol=0)
        assert bool(torch.isneginf(actual[..., vocab_size:]).all())
        return dict(initializer="verl.vLLMColocateWorkerExtension.monkey_patch_model",
                    vocab_size=vocab_size, padded_vocab_size=model.config.vocab_size,
                    native_backbone=model.model is model.policy.model,
                    parameter_inventory_preserved=True, native_logits_preserved=True)

    def archlab_reload_weights(self, path):
        from safetensors.torch import load_file

        model = self.model_runner.model
        loaded = model.load_weights(load_file(path).items())
        return dict(loaded=sorted(loaded), cache_layers=model._attention_count)
