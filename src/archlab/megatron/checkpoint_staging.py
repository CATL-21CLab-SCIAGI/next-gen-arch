"""Bounded native checkpoint staging; no model or trainer dependency."""

from __future__ import annotations

import gc
import inspect
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch


def _execute_checkpoint_request_by_local_rank(
    request: Any,
    *,
    local_rank: int,
    local_world_size: int,
    barrier: Any,
) -> None:
    """Execute one native DCP request at a time on each host.

    MCore's synchronous ``torch_dist`` writer stages every tensor owned by a
    rank to host memory before writing. The full PLE owns enough FP32 Adam
    state that staging all eight local owners concurrently exceeds the DLC
    node's host-memory limit. Planning, sharding, serialization, and
    finalization remain MCore-owned; only the per-host staging order is
    bounded here.
    """
    if local_world_size < 1 or not 0 <= local_rank < local_world_size:
        raise RuntimeError(
            f"invalid local checkpoint topology: rank={local_rank}, world={local_world_size}"
        )

    for writer_rank in range(local_world_size):
        if local_rank == writer_rank:
            call_args = list(request.async_fn_args)
            if request.preload_fn is not None:
                if len(call_args) != 3:
                    raise RuntimeError("native DCP writer changed its request ABI")
                preload = request.preload_fn
                if (
                    not isinstance(preload, partial)
                    or len(preload.args) != 2
                    or preload.args[1] is not True
                    or preload.keywords
                    or "non_blocking" not in inspect.signature(preload.func).parameters
                ):
                    raise RuntimeError("native DCP preload function changed its ABI")
                # The frozen PyTorch host allocator has no Python cache-release API.
                # Its nonblocking D2H path therefore retains every rank's staged
                # tensors in pinned memory until process exit. Use the same native
                # MCore preload implementation synchronously so GC can return the
                # pageable storage before the next local rank takes its turn.
                call_args[1] = preload.func(preload.args[0], non_blocking=False)
            if request.async_fn is not None:
                request.async_fn(*call_args, **request.async_fn_kwargs)
            del call_args
            gc.collect()
        barrier()

    for finalize_fn in request.finalize_fns:
        finalize_fn()


def _install_bounded_torch_dist_staging(training_module: Any) -> None:
    """Inject a native DCP strategy with serialized per-host tensor staging."""
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing.strategies.fully_parallel import (
        FullyParallelSaveStrategyWrapper,
    )
    from megatron.core.dist_checkpointing.strategies.torch import (
        TorchDistSaveShardedStrategy,
    )
    from megatron.training import get_args

    class _BoundedTorchDistSaveShardedStrategy(TorchDistSaveShardedStrategy):
        def save(self, sharded_state_dict, checkpoint_dir):
            request = self.async_save(sharded_state_dict, checkpoint_dir, async_strategy="mcore")
            _execute_checkpoint_request_by_local_rank(
                request,
                local_rank=int(os.environ["LOCAL_RANK"]),
                local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
                barrier=torch.distributed.barrier,
            )
            del request

    original_setup = training_module.setup_model_and_optimizer

    def setup_with_bounded_checkpoint_staging(*setup_args, **setup_kwargs):
        context = setup_kwargs.get("checkpointing_context")
        if context is None and len(setup_args) >= 3:
            context = setup_args[2]
        result = original_setup(*setup_args, **setup_kwargs)
        if context is None:
            raise RuntimeError("Megatron did not provide a checkpointing context")
        parsed = get_args()
        strategy: Any = _BoundedTorchDistSaveShardedStrategy(
            thread_count=parsed.dist_ckpt_workers,
            cpu_shm_mode=bool(getattr(parsed, "async_ckpt_use_cpu_shm", False)),
        )
        if parsed.ckpt_fully_parallel_save:
            strategy = FullyParallelSaveStrategyWrapper(
                strategy,
                parallel_state.get_data_parallel_group(with_context_parallel=True),
                parsed.ckpt_assume_constant_structure,
            )
        context["save_strategy"] = strategy
        return result

    training_module.setup_model_and_optimizer = setup_with_bounded_checkpoint_staging


def _completed_checkpoint_iteration(run_dir: Path) -> int:
    """Native cfg-container training does not refresh get_args()' startup cursor."""
    marker = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    iteration = int(marker.read_text().strip())
    if iteration < 1:
        raise RuntimeError("completion requires a positive completed checkpoint iteration")
    return iteration
