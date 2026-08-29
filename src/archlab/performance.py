"""Explicit throughput measurement protocols for comparable training runs."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass


class PerformanceError(ValueError):
    """Raised when a throughput sample cannot satisfy its declared protocol."""


@dataclass(frozen=True)
class ThroughputProtocol:
    """A predeclared steady-state timing window.

    Timestamps are recorded after each optimizer step. The first interval is
    therefore between timestamps 0 and 1. ``warmup_steps`` drops early compile,
    loader, and process-group transients; ``measurement_steps=0`` consumes every
    remaining interval.
    """

    warmup_steps: int = 10
    measurement_steps: int = 0
    statistic: str = "median"

    def validate(self) -> None:
        if self.warmup_steps < 0 or self.measurement_steps < 0:
            raise PerformanceError("throughput step counts must be non-negative")
        if self.statistic != "median":
            raise PerformanceError("only the robust median throughput statistic is supported")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)


def summarize_step_timestamps(
    timestamps: Sequence[float],
    *,
    tokens_per_step: int,
    protocol: ThroughputProtocol | None = None,
) -> dict[str, float | int | dict[str, int | str]]:
    """Report lifecycle-inclusive and protocol-window throughput separately."""

    protocol = ThroughputProtocol() if protocol is None else protocol
    protocol.validate()
    if tokens_per_step <= 0:
        raise PerformanceError("tokens_per_step must be positive")
    if len(timestamps) < 2:
        return {
            "measured_training_seconds": 0.0,
            "tokens_per_second": 0.0,
            "median_step_seconds": 0.0,
            "p90_step_seconds": 0.0,
            "steady_state_tokens_per_second": 0.0,
            "throughput_sample_intervals": 0,
            "throughput_protocol": protocol.to_dict(),
        }

    intervals = [timestamps[index] - timestamps[index - 1] for index in range(1, len(timestamps))]
    if any(not math.isfinite(value) or value <= 0 for value in intervals):
        raise PerformanceError("step timestamps must be finite and strictly increasing")
    start = min(protocol.warmup_steps, len(intervals))
    stop = len(intervals)
    if protocol.measurement_steps:
        stop = min(stop, start + protocol.measurement_steps)
    selected = intervals[start:stop]
    if not selected:
        return {
            "measured_training_seconds": 0.0,
            "tokens_per_second": 0.0,
            "median_step_seconds": 0.0,
            "p90_step_seconds": 0.0,
            "steady_state_tokens_per_second": 0.0,
            "throughput_sample_intervals": 0,
            "throughput_protocol": protocol.to_dict(),
        }

    elapsed = sum(selected)
    median_step = statistics.median(selected)
    ordered = sorted(selected)
    p90_step = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]
    return {
        "measured_training_seconds": elapsed,
        "tokens_per_second": len(selected) * tokens_per_step / elapsed,
        "median_step_seconds": median_step,
        "p90_step_seconds": p90_step,
        "steady_state_tokens_per_second": tokens_per_step / median_step,
        "throughput_sample_intervals": len(selected),
        "throughput_protocol": protocol.to_dict(),
    }
