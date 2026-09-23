"""Megatron checkpoint adapters for the architecture-owned PLE mechanisms.

The subclasses change serialization only. Table construction, owner routing,
parameter names and forward/backward arithmetic remain in architectures.
"""

from __future__ import annotations

from typing import Any

from archlab.architectures.qwen38_flash_next_full import (
    DistributedPLE as ArchitecturePLE,
)
from archlab.architectures.qwen38_flash_next_full import (
    OwnerShardedPLEEmbedding as ArchitecturePLEEmbedding,
)


class OwnerShardedPLEEmbedding(ArchitecturePLEEmbedding):
    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Expose owner partitions once, with explicit expert-DP replica identity."""
        del metadata
        from megatron.core.dist_checkpointing.mapping import ShardedTensor

        state: dict[str, Any] = {}
        global_key = f"{prefix}tables.weight"
        for slot, (partition, parameter) in enumerate(
            zip(self.global_partitions, self.tables, strict=True)
        ):
            local_key = f"{prefix}tables.{slot}"
            state[local_key] = ShardedTensor.from_rank_offsets(
                global_key,
                parameter,
                *sharded_offsets,
                (len(sharded_offsets), partition, self.partitions),
                prepend_axis_num=len(sharded_offsets),
                replica_id=(0, 0, self.replica_rank),
            )
        return state


class DistributedPLE(ArchitecturePLE):
    _embedding_type = OwnerShardedPLEEmbedding

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Recurse explicitly so the owner-sharded table contract is retained."""
        from megatron.core.transformer.utils import sharded_state_dict_default

        state: dict[str, Any] = {}
        for name, module in self.named_children():
            state.update(
                sharded_state_dict_default(
                    module,
                    f"{prefix}{name}.",
                    sharded_offsets,
                    metadata,
                )
            )
        return state
