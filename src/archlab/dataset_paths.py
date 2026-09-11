"""Dataset path selection without trainer initialization or logging."""

import os
from pathlib import Path


def resolve_climbmix_data_dir(base_dir: str | os.PathLike[str]) -> Path:
    """Return the canonical ClimbMix inventory root, with legacy-data fallback."""

    base = Path(base_dir).expanduser().resolve()
    current = base / "base_data_climbmix"
    if current.is_dir():
        return current
    return base / "base_data"
