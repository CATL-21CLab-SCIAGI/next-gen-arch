"""The only project-owned integration boundary with Megatron Core."""

from archlab.megatron.backend import MEGATRON_COMMIT, MegatronBackend

__all__ = ["MEGATRON_COMMIT", "MegatronBackend"]
