"""Checkpointed native vocabulary projection without a sequence×vocabulary buffer.

This retains the proven Limite replay chunking mechanism. The publisher owns
its head transform; the execution package owns log probabilities and entropy.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def chunked_logits_processor(head, hidden, logits_processor, logits_processor_args, *, chunk_size=512):
    """Apply an unchanged head and processor to bounded sequence chunks.

    Arguments and outputs use [batch, sequence, ...]. Labels are already shifted
    by the execution package. Non-reentrant checkpointing also differentiates
    head parameters captured by the publisher's bound method.
    """
    if hidden.ndim != 3 or hidden.shape[1] == 0 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("invalid native head chunk geometry")
    names = tuple(logits_processor_args)
    values = tuple(logits_processor_args.values())
    if not values or any(not isinstance(value, torch.Tensor) or value.shape[:2] != hidden.shape[:2]
                         for value in values):
        raise ValueError("native head processor arguments must match [batch, sequence]")

    def part(x, *arguments):
        # Package processors may normalize temperature in place. Keep saved
        # checkpoint inputs unchanged so backward replays the original values.
        kwargs = {name: value.clone() for name, value in zip(names, arguments, strict=True)}
        return logits_processor(head(x), **kwargs)

    chunks = []
    for start in range(0, hidden.shape[1], chunk_size):
        arguments = (hidden[:, start:start + chunk_size],
                     *(value[:, start:start + chunk_size] for value in values))
        result = (checkpoint(part, *arguments, use_reentrant=False)
                  if torch.is_grad_enabled() else part(*arguments))
        if not isinstance(result, dict) or not result:
            raise ValueError("native head processor must return token tensors")
        if any(not isinstance(value, torch.Tensor) or value.shape[:2] != arguments[0].shape[:2]
               for value in result.values()):
            raise ValueError("native head processor output must preserve token geometry")
        chunks.append(result)
    if any(chunk.keys() != chunks[0].keys() for chunk in chunks):
        raise ValueError("native head processor outputs differ between chunks")
    return {name: torch.cat([chunk[name] for chunk in chunks], dim=1) for name in chunks[0]}
