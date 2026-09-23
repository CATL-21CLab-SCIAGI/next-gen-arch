"""Ordered FP32 sums with a bounded temporary and a permutation backward.

Each nonnegative position must occur exactly once and collectively cover every
input row. This is the expert dispatch permutation, not a general scatter with
repeated source rows. Destination rows may receive one contribution per slot;
slots are added in their supplied order.
"""

import torch


class _OrderedPermutationSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, positions, chunk_rows):
        result = torch.zeros(
            (positions.shape[0], values.shape[1]), device=values.device, dtype=torch.float32
        )
        for slot in range(positions.shape[1]):
            rows = torch.where(positions[:, slot] >= 0)[0]
            for selected in rows.split(chunk_rows):
                source = positions[selected, slot]
                result.index_add_(0, selected, values[source].float())
        ctx.save_for_backward(positions)
        ctx.value_shape, ctx.value_dtype, ctx.chunk_rows = values.shape, values.dtype, chunk_rows
        return result

    @staticmethod
    def backward(ctx, gradient):
        (positions,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None
        # Each dispatched row is used once, so it receives one gradient. Avoid
        # allocating and adding a full zero-filled gradient for every slot/chunk.
        result = torch.empty(ctx.value_shape, device=gradient.device, dtype=ctx.value_dtype)
        for slot in range(positions.shape[1]):
            rows = torch.where(positions[:, slot] >= 0)[0]
            for selected in rows.split(ctx.chunk_rows):
                result.index_copy_(
                    0, positions[selected, slot], gradient[selected].to(ctx.value_dtype)
                )
        return result, None, None


def ordered_permutation_sum(values, positions, *, chunk_rows=8192, validate=True):
    if (
        values.ndim != 2
        or positions.ndim != 2
        or positions.dtype not in (torch.int32, torch.int64)
        or values.device != positions.device
        or not values.is_floating_point()
        or type(chunk_rows) is not int
        or chunk_rows < 1
    ):
        raise ValueError("expected floating rows, integral positions, and a positive chunk size")
    if validate:
        selected = positions[positions >= 0].sort().values
        expected = torch.arange(values.shape[0], device=positions.device, dtype=positions.dtype)
        if bool((positions < -1).any()) or not torch.equal(selected, expected):
            raise ValueError("positions must be a permutation of every source row, with -1 padding")
    return _OrderedPermutationSum.apply(values, positions, chunk_rows)
