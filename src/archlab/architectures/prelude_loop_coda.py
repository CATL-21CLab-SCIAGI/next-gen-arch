"""Tied core execution and boundary operator from qlabs-eng/scaling-exponents.

The released implementation feeds the raw prelude into the first core, then
normalizes and reinjects after every core pass, including before the coda.
This follows commit 9139c396957a57b6de9a1e7efe7e35a4a863f1a6, rather than
silently substituting the paper's zero-state first-boundary notation.
"""

import math
from dataclasses import dataclass

import torch.nn.functional as F


@dataclass(frozen=True)
class LoopLayout:
    prelude: int = 6
    core: int = 7
    coda: int = 7

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in (self.prelude, self.core, self.coda)):
            raise ValueError("prelude, core, and coda must be positive integer block counts")

    @property
    def stored_depth(self):
        return self.prelude + self.core + self.coda

    def execution(self, repetitions):
        if type(repetitions) is not int or repetitions < 1:
            raise ValueError("recursion count must be a positive integer")
        return (
            list(range(self.prelude))
            + list(range(self.prelude, self.prelude + self.core)) * repetitions
            + list(range(self.prelude + self.core, self.stored_depth))
        )


def boundary(hidden, anchor):
    """No learned parameters; FP32 RMS over features independently per stream."""
    return (F.rms_norm(hidden.float(), (hidden.shape[-1],)) + anchor.float() / math.sqrt(2)).to(
        hidden.dtype
    )


def balanced_layout(depth):
    """Split thirds; assign the first remainder to the core, then to the coda."""
    base, remainder = divmod(depth, 3)
    return LoopLayout(base, base + int(remainder >= 1), base + int(remainder == 2))
