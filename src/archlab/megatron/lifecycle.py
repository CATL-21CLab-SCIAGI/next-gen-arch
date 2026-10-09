"""Container API adaptation with an explicitly supplied forward callback."""

from __future__ import annotations

import inspect
import os

import torch


def distributed_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def distributed_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def current_iteration() -> int:
    from megatron.training import get_args

    args = get_args()
    return int(getattr(args, "curr_iteration", getattr(args, "iteration", 0)))


def architecture_from_model(model):
    current = model
    while hasattr(current, "module"):
        current = current.module
    return current.architecture


def invoke_pretrain(training_module, datasets_provider, model_provider, model_type, *, forward_step) -> None:
    parameters = inspect.signature(training_module.pretrain).parameters
    if "cfg_container" in parameters:
        from megatron.training.argument_utils import pretrain_cfg_container_from_args
        from megatron.training.arguments import parse_and_validate_args

        args = parse_and_validate_args(args_defaults={"tokenizer_type": "NullTokenizer"})
        config = pretrain_cfg_container_from_args(args)
        training_module.pretrain(
            config, datasets_provider, model_provider, model_type, forward_step
        )
        return
    training_module.pretrain(
        datasets_provider,
        model_provider,
        model_type,
        forward_step,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
