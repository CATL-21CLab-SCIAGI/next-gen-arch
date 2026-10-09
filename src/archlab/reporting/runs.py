"""Shared readers and token matching for append-only training evidence."""

import hashlib
import json
import math
from pathlib import Path


def jsonl_rows(path, *, missing_ok=False):
    path = Path(path)
    if missing_ok and not path.exists():
        return []
    return decode_jsonl(path.read_bytes())


def decode_jsonl(raw):
    rows = []
    # JSONL is delimited by LF, not Unicode line/paragraph separators in strings.
    lines = raw.split(b"\n")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not raw.endswith(b"\n"):
                break  # Concurrent writer has not published its last record yet.
            raise
        if not isinstance(value, dict):
            raise ValueError("a training ledger must contain JSON objects")
        rows.append(value)
    return rows


def read_training_run(directory, *, include_history=True):
    directory = Path(directory)
    files = [directory / "train-metrics.jsonl"]
    history = directory / "prior-phase-metrics.jsonl"
    if include_history and history.exists():
        files.insert(0, history)
    rows, sources = [], []
    for path in files:
        raw = path.read_bytes()
        part = decode_jsonl(raw)
        rows.extend(part)
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "rows": len(part),
            }
        )
    validate_training_rows(rows)
    return rows, sources


def validate_training_rows(rows):
    if not rows:
        raise ValueError("empty training ledger")
    previous_step = rows[0]["step"] - 1
    consumed = rows[0]["consumed_supervised_tokens"] - rows[0]["supervised_tokens"]
    if consumed < 0:
        raise ValueError("negative initial token cursor")
    for row in rows:
        if row["step"] != previous_step + 1:
            raise ValueError("non-contiguous training steps")
        if row["supervised_tokens"] <= 0 or not math.isfinite(row["loss"]):
            raise ValueError("invalid token count or nonfinite loss")
        consumed += row["supervised_tokens"]
        if row["consumed_supervised_tokens"] != consumed:
            raise ValueError("token ledger differs from cumulative cursor")
        previous_step = row["step"]


def matched_rows(first, second):
    """Match actual update/data cursors; never interpolate unmatched updates."""
    validate_training_rows(first)
    validate_training_rows(second)
    left = {row["step"]: row for row in first}
    right = {row["step"]: row for row in second}
    steps = sorted(left.keys() & right.keys())
    if not steps:
        raise ValueError("runs have no common training steps")
    fields = (
        "consumed_supervised_tokens",
        "supervised_tokens",
        "input_tokens",
        "window_cursor",
        "learning_rate",
    )
    for step in steps:
        for field in fields:
            if field in left[step] or field in right[step]:
                if (
                    field not in left[step]
                    or field not in right[step]
                    or left[step][field] != right[step][field]
                ):
                    raise ValueError(f"paired {field} differs at step {step}")
    return [left[s] for s in steps], [right[s] for s in steps]


def token_window_mean(rows, low, high):
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("token window must have finite increasing bounds")
    total = weight = 0.0
    for row in rows:
        end = row["consumed_supervised_tokens"]
        overlap = max(0, min(end, high) - max(end - row["supervised_tokens"], low))
        weight += overlap
        total += overlap * row["loss"]
    if weight <= 0:
        raise ValueError("empty token window")
    return total / weight


def token_smooth(rows, width):
    import numpy as np

    if width <= 0:
        raise ValueError("smoothing token count must be positive")
    validate_training_rows(rows)
    end = np.array([r["consumed_supervised_tokens"] for r in rows], dtype=float)
    count = np.array([r["supervised_tokens"] for r in rows], dtype=float)
    loss = np.array([r["loss"] for r in rows], dtype=float)
    start = end[0] - count[0]
    cumulative = np.r_[0.0, np.cumsum(count * loss)]
    beginning = np.maximum(start, end - width)
    values = (cumulative[1:] - np.interp(beginning, np.r_[start, end], cumulative)) / (
        end - beginning
    )
    return end, values
