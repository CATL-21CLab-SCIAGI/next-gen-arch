"""Destructive in-place restore qualification at an explicit checkpoint boundary."""
import hashlib
import json
from pathlib import Path

import torch


def tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, f"{prefix}/{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensors(item, f"{prefix}/{index}")


def fingerprint(state):
    """Hash all bytes with bounded host staging, including frozen tables."""
    digest = hashlib.sha256()
    size = 0
    for key, tensor in tensors(state):
        if not tensor.is_contiguous():
            raise ValueError(f"checkpoint verifier requires contiguous storage: {key}")
        digest.update(json.dumps([key, list(tensor.shape), str(tensor.dtype)]).encode())
        for chunk in tensor.detach().reshape(-1).view(torch.uint8).split(16 * 2**20):
            digest.update(chunk.cpu().numpy().tobytes())
            size += chunk.numel()
    return dict(sha256=digest.hexdigest(), bytes=size)


def restore_and_verify(args, iteration, models, optimizer, scheduler):
    import torch.distributed as dist
    from megatron.training.checkpointing import load_checkpoint

    def state():
        return dict(model=[m.state_dict() for m in models], optimizer=optimizer.state_dict())

    before = fingerprint(state())
    completed_steps = optimizer.optimizer.completed_steps
    # Model and optimizer must really be read back from the new checkpoint.
    # No live-state copies are retained to make this test pass accidentally.
    with torch.no_grad():
        for _, tensor in tensors(state()):
            tensor.zero_()
    overrides = dict(load=args.save, ckpt_step=iteration, no_load_optim=False,
                     no_load_rng=False, finetune=False)
    previous = {name: getattr(args, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(args, name, value)
        restored_iteration, _ = load_checkpoint(models, optimizer, scheduler, checkpointing_context={})
    finally:
        for name, value in previous.items():
            setattr(args, name, value)
    after = fingerprint(state())
    if (before != after or restored_iteration != iteration
            or optimizer.optimizer.completed_steps != completed_steps):
        raise ValueError("full model/optimizer checkpoint restore was not exact")
    receipt = dict(rank=dist.get_rank(), world_size=dist.get_world_size(), iteration=iteration,
                   completed_updates=completed_steps, checkpoint_restore=True, **after)
    (Path(args.save).parent / f"checkpoint-restore-rank-{dist.get_rank():02d}.json").write_text(
        json.dumps(receipt, indent=2))
    print("ARCHLAB_CHECKPOINT_RESTORE " + json.dumps(receipt), flush=True)
