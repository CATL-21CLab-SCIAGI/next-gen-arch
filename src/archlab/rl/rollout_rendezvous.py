"""Host-only admission to learner collectives after every actor has drained."""

from datetime import timedelta

import torch.distributed as dist


class RolloutRendezvous:
    """Never launch a waiting NCCL kernel while a peer is still decoding."""

    def __init__(self, world_size, *, timeout_seconds=1800):
        if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
            raise ValueError("rollout rendezvous timeout must be a positive integer")
        self.timeout = timedelta(seconds=timeout_seconds)
        self.group = None
        if world_size > 1:
            if not dist.is_initialized() or dist.get_world_size() != world_size:
                raise RuntimeError("rollout rendezvous requires the initialized training world")
            # Construct collectively at startup, before any actor is submitted.
            self.group = dist.new_group(backend="gloo", timeout=self.timeout)

    def __call__(self):
        if self.group is not None:
            dist.monitored_barrier(group=self.group, timeout=self.timeout, wait_all_ranks=True)

    def gather_object(self, value):
        """Gather GRPO log lists in rank order with Accelerate's flattening.

        Completion text never needs a CUDA collective. Reuse the host group
        created at startup so logging cannot send its pickled bytes over NCCL.
        """
        if self.group is None:
            return value
        return [item for packet in self.gather_rank_objects(value) for item in packet]

    def gather_rank_objects(self, value):
        """Gather intact rank-local objects, including checkpoint RNG state."""
        if self.group is None:
            return [value]
        packets = [None] * dist.get_world_size(group=self.group)
        dist.all_gather_object(packets, value, group=self.group)
        return packets
