"""Keep DP padding out of stateful V4.1 eager source computations.

Install on project-owned model instances; the container implementation remains
unchanged. Attention/MoE communication continues to see its padded geometry.
"""
from types import MethodType


def real_token_count(batch, padded):
    count = getattr(batch, "num_token_non_padded_cpu", padded)
    if type(count) is not int or not 0 <= count <= padded:
        raise ValueError("invalid real-token count for padded V4.1 batch")
    return count


class _LiveSources:
    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def forward_low_ratio_sources(self, *, x, q_lora, positions, forward_batch, **kwargs):
        # Decode has per-slot padded metadata and native masked cache writes.
        # Eager extend instead constructs request IDs from the real lengths.
        if forward_batch.forward_mode.is_extend():
            count = real_token_count(forward_batch, positions.shape[0])
            if count == 0:
                return
            if sum(forward_batch.extend_seq_lens_cpu) != count:
                raise ValueError("V4.1 extend lengths do not cover real tokens")
            x, q_lora, positions = x[:count], q_lora[:count], positions[:count]
        return self.backend.forward_low_ratio_sources(
            x=x, q_lora=q_lora, positions=positions, forward_batch=forward_batch, **kwargs)


def install_live_source_boundary(attention):
    original = attention._forward_prepare
    original_forward = attention.forward

    def forward(self, x, positions, forward_batch, **kwargs):
        if real_token_count(forward_batch, x.shape[0]) == 0:
            # Attention TP groups share the same local DP batch. They may skip
            # local attention together; the outer layer still runs MoE collectives.
            return x.new_zeros(x.shape)
        return original_forward(x, positions, forward_batch, **kwargs)

    attention.forward = MethodType(forward, attention)

    def prepare(self, x, positions, forward_batch, attn_backend, *args, **kwargs):
        return original(x, positions, forward_batch, _LiveSources(attn_backend), *args, **kwargs)

    attention._forward_prepare = MethodType(prepare, attention)


def install_hash_padding_boundary(hasher):
    """Padded decode slots must write only the hasher's dedicated spare row."""
    from copy import copy

    original = hasher.forward

    def forward(self, input_ids, forward_batch):
        count = real_token_count(forward_batch, input_ids.shape[0])
        if count == 0:
            # Idle DP ranks can have nonempty communication padding. No request
            # owns these rows, so do not enter the native stateful hash kernel.
            return input_ids.new_zeros((input_ids.shape[0], self.primes.shape[0],
                                        self.offsets.shape[1]))
        if forward_batch.forward_mode.is_decode() and count < input_ids.shape[0]:
            forward_batch = copy(forward_batch)
            forward_batch.req_pool_indices = forward_batch.req_pool_indices.clone()
            forward_batch.req_pool_indices[count:] = self.pad_row
        return original(input_ids, forward_batch)

    hasher.forward = MethodType(forward, hasher)
