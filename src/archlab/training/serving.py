"""Narrow vLLM boundary for explicit Limite attention experiments.

The package owns paging, scheduling and attention kernels. Publisher projections,
RMSNorm, RoPE, gates, value embeddings, MUDD and sigmoid logits stay unchanged.
Tensor/pipeline parallelism is unsupported. Replacement serving uses the
separately validated short/long package-cache boundary.
"""
from __future__ import annotations

from types import MethodType

from torch import nn, no_grad

from archlab.architectures.limite_bindings import bind_native_forward
from archlab.training.limite import LimiteForCausalLM


class _AttentionInterface:
    def get_interface(self, name, fallback):
        return _paged_attention


@no_grad()
def _refresh_inference_buffers(policy):
    """Refresh publisher folds without invalidating captured tensor addresses."""
    buffers = [(module, name, value) for module in policy.modules()
               for name, value in module.named_buffers(recurse=False, remove_duplicate=False)
               if name.startswith("_inference_")]
    policy.train()
    policy.eval()
    for module, name, original in buffers:
        updated = module.get_buffer(name)
        if (updated is None or updated.shape != original.shape
                or updated.dtype != original.dtype or updated.device != original.device):
            raise ValueError("inference buffer contract changed during weight reload: " + name)
        original.copy_(updated)
        setattr(module, name, original)


def _paged_attention(module, query, key, value, attention_mask, *, scaling, **kwargs):
    batch, heads, tokens, dim = query.shape
    if batch != 1 or scaling != module.scaling:
        raise ValueError("vLLM Limite expects one packed token stream and native scaling")
    q, k, v = (tensor.transpose(1, 2).reshape(tokens, -1) for tensor in (query, key, value))
    output = module.archlab_paged_attention(q, k, v)
    return output.view(1, tokens, heads, dim), None


_GLOBAL_WINDOW = 1 << 30


def _token_layout(metadata, tokens):
    """Per-token sequence index and absolute position from package metadata, without host syncs."""
    import torch

    starts = metadata.query_start_loc
    index = torch.arange(tokens, device=starts.device, dtype=starts.dtype)
    seq = torch.searchsorted(starts[1:].contiguous(), index, right=True)
    seq = seq.clamp_(max=metadata.seq_lens.numel() - 1)
    first = starts[seq]
    position = metadata.seq_lens[seq] - (starts[seq + 1] - first) + (index - first)
    return seq.to(torch.int32), position.to(torch.int32)


def _paged_simplicial_attention(module, query, key, value, attention_mask, *, scaling, **kwargs):
    """Adapter prelude: long K2/V2 and short K1/V1 in two package caches, joint paged reduction."""
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention.attention import (
        get_attention_context,
        unified_kv_cache_update,
    )

    from archlab.architectures.simplicial_paged import simplicial_paged_attention

    batch, heads, tokens, dim = query.shape
    if batch != 1 or scaling != module.scaling or attention_mask is not None:
        raise ValueError("vLLM Limite expects one packed unmasked token stream and native scaling")
    q, k, v = (tensor.transpose(1, 2)[0] for tensor in (query, key, value))  # [T, H, D]
    k1, v1 = (tensor[0] for tensor in module._short_values)
    long_window = _GLOBAL_WINDOW if module.is_global else module.window_span
    short_window = min(module._short_window, long_window)
    # Native reduction contract: FP32 queries pre-scaled so the kernel's D^-1/2 yields `scaling`.
    q = q.float() * (scaling * dim ** 0.5)
    if get_forward_context().attn_metadata is None:
        # Profiling/dummy run: no request cache exists; reduce over the packed tokens directly.
        from archlab.architectures.simplicial_packed import packed_simplicial_attention

        packed_long = tokens if module.is_global else long_window
        out = packed_simplicial_attention(q[None], k1[None].float(), k[None].float(), v1[None].float(),
                                          v[None].float(), min(short_window, packed_long), packed_long)[0]
    else:
        long_owner, short_owner = module.archlab_long_attention, module.archlab_short_attention
        long_metadata, _, long_cache, _ = get_attention_context(long_owner.layer_name)
        short_metadata, _, short_cache, _ = get_attention_context(short_owner.layer_name)
        unified_kv_cache_update(k, v, long_owner.layer_name)
        unified_kv_cache_update(k1, v1, short_owner.layer_name)
        seq, position = _token_layout(long_metadata, tokens)
        out = simplicial_paged_attention(
            q, short_cache, long_cache, short_metadata.block_table, seq, position,
            short_window=short_window, long_window=long_window, scale=dim ** -0.5,
            long_block_table=long_metadata.block_table)
    return out.to(value.dtype).view(1, tokens, heads, dim), None


class _SimplicialInterface:
    def get_interface(self, name, fallback):
        return _paged_simplicial_attention


def bind_normal_attention(policy, attention_factory, *, prefix=""):
    """Bind package caches to the 2×N native attentions; simplicial preludes also get a short-axis cache.

    Normal: every prelude and backbone attention uses one package cache and the package reduction.
    Simplicial: each prelude owns a long cache (native window) and a short cache (K1/V1 window) and
    uses the batched paged joint reduction; backbone attentions stay normal.
    An uninserted publisher binds only its N backbone attentions.
    """
    backbone = policy.model
    inserted = getattr(backbone, "adapter_config", None)
    variant = inserted["variant"] if inserted else None
    if inserted and variant not in ("normal", "simplicial"):
        raise ValueError("unknown Limite adapter variant: " + str(variant))
    if inserted:
        sites = [(("prelude", adapter.native), ("backbone", layer.self_attn))
                 for adapter, layer in zip(backbone.adapters, backbone.base.layers, strict=True)]
    else:
        sites = [(("backbone", layer.self_attn),) for layer in backbone.layers]
    attentions = []
    for index, site in enumerate(sites):
        for kind, attention in site:
            if kind == "prelude" and variant == "simplicial":
                # Allocation margin only; the kernel enforces the exact windows.
                long_capacity = None if attention.is_global else attention.window_span + 16
                attention.archlab_long_attention = attention_factory(
                    attention, prefix=f"{prefix}layers.{index}.prelude.long_attention",
                    window_span=long_capacity)
                attention.archlab_short_attention = attention_factory(
                    attention, prefix=f"{prefix}layers.{index}.prelude.short_attention",
                    window_span=attention._short_window + 16)
                interface, name = _SimplicialInterface(), "archlab_vllm_simplicial"
            else:
                attention.archlab_paged_attention = attention_factory(
                    attention, prefix=f"{prefix}layers.{index}.{kind}.attention")
                interface, name = _AttentionInterface(), "archlab_vllm_normal"
            # Always bind the publisher's original function, not the old
            # TileLang reduction. No parent library or global registry patch.
            forward = bind_native_forward(
                type(attention).forward, (("ALL_ATTENTION_FUNCTIONS", interface),), name)
            attention.forward = MethodType(forward, attention)
            attention.config._attn_implementation = "archlab_paged"
            attentions.append(attention)
    # Paged Attention owns masks and cached K/V. Avoid constructing dense masks
    # for a packed stream of independent requests; positions come from vLLM.
    # An uninserted publisher receives the same empty mapping from forward().
    if inserted:
        backbone._masks = MethodType(
            lambda self, hidden, positions, mask, cache: dict(full_attention=None, sliding_attention=None),
            backbone,
        )
    return tuple(attentions)


class VLLMLimiteForCausalLM(nn.Module):
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
        # vLLM constructs on its CUDA default device. The publisher deliberately
        # creates rotary frequencies on CPU; its _apply rebuilds them on moves.
        wrapper.to(device=next(wrapper.parameters()).device)
        self.policy = wrapper.policy
        self.config = config

        def factory(attention, *, prefix, window_span=...):
            return Attention(
                num_heads=attention.num_heads, head_size=attention.head_dim,
                scale=attention.scaling, num_kv_heads=attention.num_kv_heads,
                cache_config=vllm_config.cache_config,
                per_layer_sliding_window=(attention.window_span if window_span is ... else window_span),
                prefix=prefix,
            )

        if config.conversion:
            from archlab.training.serving_conversion import (
                bind_conversion_attention,
                stock_attention_factory,
                validate_conversion_serving,
            )

            validate_conversion_serving(vllm_config)
            self._attention_count = len(bind_conversion_attention(
                self.policy, stock_attention_factory(vllm_config), prefix=prefix,
            ))
        else:
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
        inserted = getattr(self.policy.model, "adapter_config", None)
        masks = None if inserted else dict(full_attention=None, sliding_attention=None)
        output = self.policy.model(input_ids=input_ids[None], position_ids=positions[None],
                                   attention_mask=masks,
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
        _refresh_inference_buffers(self.policy)
        return loaded


def register():
    """Call in each serving process through the vLLM general plugin mechanism."""
    from vllm import ModelRegistry

    if "ArchlabLimiteForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("ArchlabLimiteForCausalLM", "archlab.training.serving:VLLMLimiteForCausalLM")


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
