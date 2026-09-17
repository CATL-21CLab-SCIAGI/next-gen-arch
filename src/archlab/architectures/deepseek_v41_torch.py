"""PyTorch numerical primitives for the frozen V4.1 text backbone.

Sparse gather equations follow NeMo Automodel's ``sparse_attention_torch``;
query checkpointing bounds training intermediates without changing the key
set. The sink participates in the denominator, with a zero value vector.
Unlike the upstream oracle, its logit also enters the stability maximum.
Quantizers follow the pinned native kernel, including FP4 ties-to-even.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def _sparse_chunk(q, kv, sink, indices, scale, native_rounding=False):
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    valid = (indices >= 0) & (indices < kv.shape[1])
    safe = indices.clamp(0, max(0, kv.shape[1] - 1)).long()
    batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
    selected = kv[batch, safe].to(dtype)
    scores = torch.einsum("bshd,bskd->bshk", q.to(dtype), selected) * scale
    scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
    if native_rounding:
        # The released kernel rounds UNNORMALIZED exp scores to BF16 in
        # 64-key blocks, while accumulating the denominator in FP32. Rounding
        # normalized probabilities or retaining FP32 exps changes its forward.
        maximum = scores.new_full((*scores.shape[:-1], 1), -1e30)
        denominator = torch.zeros_like(maximum)
        numerator = q.new_zeros(q.shape, dtype=dtype)
        for start in range(0, scores.shape[-1], 64):
            block = scores[..., start:start + 64]
            updated = torch.maximum(maximum, block.amax(-1, keepdim=True))
            rescale = torch.exp(maximum - updated)
            exponentials = torch.exp(block - updated)
            denominator = denominator * rescale + exponentials.sum(-1, keepdim=True)
            numerator = numerator * rescale + torch.einsum(
                "bshk,bskd->bshd", exponentials.to(torch.bfloat16).to(dtype),
                selected[:, :, start:start + 64],
            )
            maximum = updated
        # Algebraically equivalent sink normalization without overflowing when
        # all keys are masked or the sink dominates. Do not change the exp
        # rounding scale used above to the sink-dependent scale.
        sinks = sink.to(dtype)[None, None, :, None]
        stable_max = torch.maximum(maximum, sinks)
        rescale = torch.exp(maximum - stable_max)
        return (numerator * rescale / (denominator * rescale + torch.exp(sinks - stable_max))).to(q.dtype)
    # Do not deduplicate repeated slots: the native kernel counts each one.
    sinks = sink.to(dtype)[None, None, :, None].expand(*scores.shape[:-1], 1)
    probabilities = torch.cat((scores, sinks), -1).softmax(-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probabilities, selected).to(q.dtype)


def query_chunked_sparse_attention(q, kv, sink, indices, scale, *, query_chunk=32, recompute=True,
                                   native_rounding=False):
    """Attend Q [B,S,H,D] to gathered shared KV [B,K,D] with indices [B,S,T].

    Sink logits [H] supply denominator-only mass. Return [B,S,H,D]. The
    native-rounding option reproduces the released BF16 exp-score rounding;
    autograd through dtype casts is its explicitly chosen straight-through
    approximation, not a derivative of discrete rounding.
    """
    if query_chunk < 1 or q.ndim != 4 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError("invalid sparse attention geometry")
    if q.shape[:2] != indices.shape[:2] or q.shape[0] != kv.shape[0] or q.shape[-1] != kv.shape[-1]:
        raise ValueError("mismatched sparse attention tensors")
    if sink.shape != (q.shape[2],) or kv.shape[1] < 1 or q.shape[1] < 1:
        raise ValueError("empty KV/query or invalid sink")
    outputs = []
    for start in range(0, q.shape[1], query_chunk):
        args = (q[:, start:start + query_chunk], kv, sink, indices[:, start:start + query_chunk], scale,
                native_rounding)
        if recompute and torch.is_grad_enabled() and any(x.requires_grad for x in (q, kv, sink)):
            outputs.append(checkpoint(_sparse_chunk, *args, use_reentrant=False))
        else:
            outputs.append(_sparse_chunk(*args))
    return torch.cat(outputs, dim=1)


class _IdentityGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rounded):
        return rounded

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def rounded_activation(x, *, bits=8, block_size=32, e4m3_scale=False, straight_through=True):
    """Return dequantized native-style activations, with an explicit optional STE.

    No quantized GEMM or TileLang call. This retains activation rounding while
    letting ordinary BF16 linear layers propagate input gradients.
    """
    if bits not in (4, 8) or block_size < 1 or x.shape[-1] % block_size:
        raise ValueError("unsupported activation quantization geometry")
    if bits == 8 and e4m3_scale:
        raise ValueError("native FP8 activations use power-of-two scales")
    with torch.no_grad():
        grouped = x.float().unflatten(-1, (-1, block_size))
        amax = grouped.abs().amax(-1, keepdim=True)
        if bits == 8:
            scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-4) * (1. / 448))))
            values = (grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        else:
            if e4m3_scale:
                scale = (amax.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
            else:
                scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(6 * 2**-126) * (1. / 6))))
            z = (grouped / scale).clamp(-6, 6)
            # E2M1 nonnegative values: 0,.5,1,1.5,2,3,4,6. Every midpoint
            # rounds to the even code, exactly as the native FP4 conversion.
            boundaries = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=x.device)
            codes = torch.bucketize(z.abs().contiguous(), boundaries)
            codes = codes + ((codes % 2 == 1) & (z.abs() == boundaries[codes.clamp_max(6)])).long()
            book = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
            values = torch.copysign(book[codes], z)
        rounded = (values * scale).flatten(-2).to(x.dtype)
    return _IdentityGradient.apply(x, rounded) if straight_through else rounded
