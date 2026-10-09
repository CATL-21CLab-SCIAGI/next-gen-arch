"""Stable grouped FP32 sums with a single write per destination row."""

import torch


def ordered_index_sum(destination, indices, values):
    if indices.numel() == 0:
        return
    order = indices.argsort(stable=True)
    ordered = indices[order]
    unique, counts = torch.unique_consecutive(ordered, return_counts=True)
    sums = torch.segment_reduce(values[order], "sum", lengths=counts)
    destination.index_add_(0, unique, sums)
