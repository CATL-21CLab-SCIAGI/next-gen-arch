"""Compare measured, identical learner workloads; no hardware work at import."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median


@dataclass
class ThroughputContract:
    baseline: str | None = None
    sequence_length: int = 2048
    warmup: int = 2
    samples: int = 5
    max_slowdown: float = 1.2

    def validate(self):
        if self.sequence_length < 2 or self.warmup < 1 or self.samples < 3:
            raise ValueError("throughput measurement needs a nonempty workload, warmup and >=3 samples")
        if not math.isfinite(self.max_slowdown) or not 1 <= self.max_slowdown <= 2:
            raise ValueError("throughput slowdown allowance must be explicit and between 1 and 2")


def compare_throughput(candidate: dict, baseline: dict, max_slowdown: float) -> dict:
    if not math.isfinite(max_slowdown) or not 1 <= max_slowdown <= 2:
        raise ValueError("invalid throughput slowdown allowance")
    if candidate["contract"] != baseline["contract"]:
        raise ValueError("throughput baseline has a different workload, model, runtime or topology")
    for record in (candidate, baseline):
        times = record["seconds"]
        if (len(times) < 3 or len(times) != record["contract"]["settings"]["samples"]
                or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in times)):
            raise ValueError("throughput samples must be finite, positive and exclude warmup")
    actual, reference = median(candidate["seconds"]), median(baseline["seconds"])
    ratio = actual / reference
    return dict(status="passed" if ratio <= max_slowdown else "failed",
                candidate_median_seconds=actual, baseline_median_seconds=reference,
                slowdown=ratio, maximum_slowdown=max_slowdown,
                scope=candidate["contract"]["scope"])


def check_throughput(output: Path, specification: ThroughputContract) -> dict:
    specification.validate()
    records = sorted(Path(output).glob("**/throughput.json"))
    if len(records) != 1:
        raise ValueError("learner benchmark must write exactly one throughput receipt")
    candidate = json.loads(records[0].read_text())
    settings = asdict(specification)
    settings.pop("baseline")
    settings.pop("max_slowdown")
    if candidate["contract"]["settings"] != settings:
        raise ValueError("candidate measurement differs from the declared throughput workload")
    if not specification.baseline:
        return dict(status="pending", reason="measured candidate needs an explicit matched throughput baseline")
    baseline = Path(specification.baseline)
    if baseline.resolve() == records[0].resolve():
        raise ValueError("candidate cannot serve as its own throughput baseline")
    from archlab.artifacts import sha256_file

    return dict(compare_throughput(candidate, json.loads(baseline.read_text()),
                                   specification.max_slowdown),
                candidate=str(records[0]), baseline=str(baseline),
                candidate_sha256=sha256_file(records[0]), baseline_sha256=sha256_file(baseline))
