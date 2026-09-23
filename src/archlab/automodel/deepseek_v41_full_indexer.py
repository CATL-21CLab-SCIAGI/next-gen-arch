# ruff: noqa: I001  # Preserve the checkpoint-qualified import order.
"""Explicit sampled KL objective for the release's hard-top-k CSA2 indexer.

Selection and inference outputs are unchanged. The indexer learns from detached
compressed-attention probabilities of its current parent attention module.
This is a new, shared full-finetuning contract, not a released training recipe.
"""

from dataclasses import replace
from types import MethodType
import weakref
import torch
from archlab.architectures.deepseek_v41_indexer_objective import attention_indexer_kl


class _InjectAuxiliary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, auxiliary, backward_scale):
        ctx.backward_scale = backward_scale
        return hidden

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient.new_tensor(ctx.backward_scale, dtype=torch.float32), None


def _indexer_forward(
    self, hidden_states, *, query_latent, latent, angles, compressed_angles, state
):
    from nemo_automodel.components.models.deepseek_v41.attention import _apply_rope
    from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache

    result = self._archlab_original_forward(
        hidden_states,
        query_latent=query_latent,
        latent=latent,
        angles=angles,
        compressed_angles=compressed_angles,
        state=state,
    )
    self._archlab_auxiliary = None
    if not self.training or not torch.is_grad_enabled():
        return result
    parent = self._archlab_parent()
    sequence = hidden_states.shape[1]
    ratio = result.compression_ratio
    width = result.index_keys.shape[1]
    if width == 0:
        self._archlab_auxiliary = sum(p.sum() * 0 for p in self.parameters())
        return result
    ids = (
        torch.linspace(
            max(0, ratio - 1),
            sequence - 1,
            min(self._archlab_sample_queries, sequence - ratio + 1),
            device=hidden_states.device,
        )
        .long()
        .unique()
    )
    sampled_angles = angles[:, ids]
    h, ql = hidden_states[:, ids].detach(), query_latent[:, ids].detach()
    keys = result.index_keys.detach()
    if self.owns_keys:
        keys = quantize_cache(
            _apply_rope(self.k_norm(self.wk(latent.detach())), compressed_angles),
            format="mxfp4",
            block_size=32,
        )
    queries = self.wq_b(ql).unflatten(-1, (self.num_heads, self.head_dim))
    queries = quantize_cache(_apply_rope(queries, sampled_angles), format="mxfp4", block_size=32)
    weights = self.weights_proj(h) * (self.head_dim**-0.5 * self.num_heads**-0.5)
    scores = (torch.einsum("bqhd,bkd->bqhk", queries, keys).relu() * weights.unsqueeze(-1)).sum(2)
    allowed = (
        torch.arange(width, device=h.device)[None, None, :] < ((ids + 1) // ratio)[None, :, None]
    )
    allowed = allowed.expand(h.shape[0], -1, -1)
    query_mask = getattr(self, "_archlab_query_valid_mask", None)
    if query_mask is not None:
        allowed = allowed & query_mask[:, ids, None]
    if result.compressed_valid is not None:
        allowed = allowed & result.compressed_valid[:, None]
    if self.uses_candidates and not self.is_candidate_source:
        allowed = allowed & state.candidates[:, ids]
    with torch.no_grad():
        teacher_queries = _apply_rope(
            parent.wq_b(ql).unflatten(-1, (parent.num_heads, parent.head_dim)), sampled_angles
        )
    self._archlab_auxiliary = attention_indexer_kl(
        scores, teacher_queries, result.compressed_kv.detach(), allowed, scale=parent.head_dim**-0.5
    )
    self._archlab_last_kl = float(self._archlab_auxiliary.detach())
    return result


def _attention_forward(self, *args, **kwargs):
    self.indexer._archlab_query_valid_mask = kwargs.get("attention_mask")
    output = self._archlab_original_full_forward(*args, **kwargs)
    auxiliary = self.indexer._archlab_auxiliary
    self.indexer._archlab_auxiliary = None
    if auxiliary is not None:
        output = replace(
            output,
            hidden_states=_InjectAuxiliary.apply(
                output.hidden_states, auxiliary, self.indexer._archlab_auxiliary_backward_scale
            ),
        )
    return output


def install_trainable_indexers(model, *, sample_queries=64):
    from nemo_automodel.components.models.deepseek_v41.attention import DeepseekV41Attention

    indexers = []
    for attention in model.modules():
        if not isinstance(attention, DeepseekV41Attention) or attention.indexer is None:
            continue
        indexer = attention.indexer
        indexer._archlab_parent = weakref.ref(attention)
        from archlab.automodel.deepseek_v41_full_indexer_memory import (
            bind_memory_efficient_selector,
        )

        indexer._archlab_original_forward = bind_memory_efficient_selector(indexer)
        indexer._archlab_sample_queries = sample_queries
        indexer._archlab_auxiliary_backward_scale = 0.0
        indexer._archlab_auxiliary = None
        indexer._archlab_last_kl = None
        indexer.forward = MethodType(_indexer_forward, indexer)
        attention._archlab_original_full_forward = attention.forward
        attention.forward = MethodType(_attention_forward, attention)
        indexers.append(indexer)
    if not indexers:
        raise ValueError("expected CSA2 indexers")
    return indexers


def set_indexer_loss_scale(indexers, *, coefficient, world_size, accumulation):
    for indexer in indexers:
        indexer._archlab_auxiliary_backward_scale = coefficient / (
            len(indexers) * world_size * accumulation
        )
