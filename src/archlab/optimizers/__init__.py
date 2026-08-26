"""Optimizers that are not yet available from the pinned Megatron version."""

from archlab.optimizers.speedrun import DistMuonAdamW, MuonAdamW

__all__ = ["DistMuonAdamW", "MuonAdamW"]
