"""Failure taxonomy used by launchers and run manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FailureClassification:
    category: str
    retriable: bool
    reason: str

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def classify_failure(error: BaseException) -> FailureClassification:
    """Classify conservatively: identical numerical and contract retries are forbidden."""

    message = f"{type(error).__name__}: {error}".lower()
    if any(token in message for token in ("non-finite", "nan", "infinity", "inf gradient")):
        return FailureClassification("numerical", False, "non-finite model state or metric")
    if any(token in message for token in ("out of memory", "cuda oom", "cublas_status_alloc")):
        return FailureClassification("capacity", False, "the declared configuration exceeds capacity")
    if any(
        token in message
        for token in (
            "nccl",
            "connection reset",
            "connection closed",
            "timed out",
            "timeout",
            "preempt",
            "sigterm",
            "temporary failure",
        )
    ):
        return FailureClassification("operational", True, "transient infrastructure failure")
    if isinstance(error, (ValueError, AssertionError, FileNotFoundError)) or any(
        token in message
        for token in ("contract", "manifest", "mismatch", "missing", "refusing to overwrite")
    ):
        return FailureClassification("contract", False, "experiment precondition failed")
    return FailureClassification("unknown", False, "manual triage required before retry")
