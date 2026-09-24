"""Tensor-aware checkpoint packing; large GPU states never enter a BytesIO object."""
import torch
import torch.distributed as dist
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor


def pack_state(payload):
    rank = dist.get_rank()
    tensors = {}

    def pack(value):
        if isinstance(value, torch.Tensor):
            key = f"optimizer.resident.rank_{rank}.tensor_{len(tensors)}"
            tensors[key] = ShardedTensor.from_rank_offsets(key, value.detach())
            return {"__resident_tensor__": key}
        if isinstance(value, dict):
            return {key: pack(item) for key, item in value.items()}
        if isinstance(value, list):
            return [pack(item) for item in value]
        if isinstance(value, tuple):
            return tuple(pack(item) for item in value)
        return value

    metadata = pack(payload)
    return dict(metadata=ShardedObject("optimizer.resident.metadata", metadata,
                global_shape=(dist.get_world_size(),), global_offset=(rank,), replica_id=0),
                tensors=tensors)


def unpack_state(state):
    def unpack(value):
        if isinstance(value, dict):
            if set(value) == {"__resident_tensor__"}:
                return state["tensors"][value["__resident_tensor__"]]
            return {key: unpack(item) for key, item in value.items()}
        if isinstance(value, list):
            return [unpack(item) for item in value]
        if isinstance(value, tuple):
            return tuple(unpack(item) for item in value)
        return value
    return unpack(state["metadata"])
