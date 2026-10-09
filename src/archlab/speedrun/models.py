"""Frozen runtime defaults around the backend-neutral model factory."""

from __future__ import annotations

import archlab.model_factory as model_factory
from archlab.architectures.base import ArchitectureRuntime
from archlab.model_factory import (
    build_engram_token_map as build_engram_token_map,
)
from archlab.model_factory import (
    infer_model_dims as infer_model_dims,
)
from archlab.model_factory import (
    instantiate_model as instantiate_model,
)
from archlab.model_factory import (
    model_config_to_dict as model_config_to_dict,
)
from archlab.model_factory import (
    patch_missing_model_state as patch_missing_model_state,
)
from archlab.model_factory import (
    strip_backend_extra_state as strip_backend_extra_state,
)


def training_architecture_runtime() -> ArchitectureRuntime:
    """Bind pure architecture definitions to this trainer's execution ops."""
    from archlab.speedrun.attention import flash_attn
    from archlab.speedrun.runtime import COMPUTE_DTYPE, print0

    return ArchitectureRuntime(
        compute_dtype=COMPUTE_DTYPE,
        attention=flash_attn,
        log=print0,
    )


def build_model_config(**kwargs):
    return model_factory.build_model_config(**kwargs, runtime=training_architecture_runtime())


def patch_model_config_kwargs(model_config_kwargs):
    return model_factory.patch_model_config_kwargs(model_config_kwargs, runtime=training_architecture_runtime())


def build_model_from_config_kwargs(model_config_kwargs, *, runtime_backend="native"):
    return model_factory.build_model_from_config_kwargs(model_config_kwargs, runtime_backend=runtime_backend, runtime=training_architecture_runtime())
