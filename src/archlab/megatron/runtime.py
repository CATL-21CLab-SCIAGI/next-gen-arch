"""Validate the container-owned Megatron runtime used by execution adapters."""

from __future__ import annotations

import os
import platform
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

MEGATRON_DISTRIBUTION = "megatron-core"
MEGATRON_CONTAINER_PROFILE = "nemo-26.06"


def _megatron_package_path() -> Path:
    package = find_spec("megatron")
    core = find_spec("megatron.core") if package is not None else None
    if package is None or core is None:
        raise RuntimeError(
            "system Megatron Core is unavailable; run inside the validated NeMo container "
            "or install a compatible megatron-core distribution"
        )
    if package.submodule_search_locations:
        return Path(next(iter(package.submodule_search_locations))).resolve()
    if package.origin:
        return Path(package.origin).resolve().parent
    raise RuntimeError("could not resolve the system Megatron package path")


def _resolve_pretrain_script(package_path: Path) -> Path:
    override = os.environ.get("NGA_MEGATRON_PRETRAIN")
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.is_file():
            raise RuntimeError(f"NGA_MEGATRON_PRETRAIN is not a file: {candidate}")
        return candidate

    candidates = (
        package_path.parent / "pretrain_gpt.py",
        package_path.parent.parent / "pretrain_gpt.py",
        Path("/opt/Megatron-LM/pretrain_gpt.py"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "system Megatron Core is importable, but pretrain_gpt.py was not found; "
        "set NGA_MEGATRON_PRETRAIN to the container-provided entrypoint"
    )


def validate_runtime(*, require_pretrain: bool = True) -> dict[str, str]:
    """Validate and describe the system-owned Megatron runtime."""
    package_path = _megatron_package_path()
    try:
        distribution_version = version(MEGATRON_DISTRIBUTION)
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"{MEGATRON_DISTRIBUTION} has no installed distribution metadata"
        ) from error

    runtime = {
        "provider": "system",
        "distribution": MEGATRON_DISTRIBUTION,
        "version": distribution_version,
        "package_path": str(package_path),
        "validated_container_profile": MEGATRON_CONTAINER_PROFILE,
        "python": platform.python_version(),
    }
    for key, distribution in (
        ("torch", "torch"),
        ("transformer_engine", "transformer-engine"),
        ("nemo_toolkit", "nemo-toolkit"),
    ):
        try:
            runtime[key] = version(distribution)
        except PackageNotFoundError:
            runtime[key] = "not-installed"
    if container_digest := os.environ.get("NGA_CONTAINER_DIGEST"):
        runtime["container_digest"] = container_digest
    if require_pretrain:
        runtime["pretrain_script"] = str(_resolve_pretrain_script(package_path))
    return runtime


