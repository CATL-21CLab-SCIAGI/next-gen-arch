"""Architecture definitions only; execution belongs to a backend boundary."""

from archlab.architectures.qwen38_27b import Qwen38Dense, Qwen38DenseConfig
from archlab.architectures.qwen38_flash_next import (
    Qwen38FlashNext,
    Qwen38FlashNextConfig,
)

__all__ = [
    "Qwen38Dense",
    "Qwen38DenseConfig",
    "Qwen38FlashNext",
    "Qwen38FlashNextConfig",
]
