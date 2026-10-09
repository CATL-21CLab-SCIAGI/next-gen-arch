"""Load lightweight components from the pinned XiaomiMiMo/verl submodule.

The native Limite executor owns model/cache support. This integration uses the
upstream repetition detector without importing its Ray/FSDP serving stack.
"""

import hashlib
import importlib.util
import subprocess
from functools import lru_cache
from pathlib import Path

VERL_REVISION = "a2ad9f6160b03ff2d47e59832bfb6b289f37c917"
REPETITION_PATH = "recipes/design/repetition.py"
REPETITION_SHA256 = "af0c5182b0ce17ca335a12575708b602b471ec64f5a2b106a920ef5dc2a91411"


@lru_cache(maxsize=4)
def repetition_component(root: Path):
    root = Path(root).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    path = root / REPETITION_PATH
    if revision != VERL_REVISION or hashlib.sha256(path.read_bytes()).hexdigest() != REPETITION_SHA256:
        raise ValueError("verl component differs from the qualified upstream pin")
    spec = importlib.util.spec_from_file_location("archlab_upstream_verl_repetition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.degenerate_repetition, {
        "repository": "https://github.com/XiaomiMiMo/verl",
        "revision": revision, "component": REPETITION_PATH, "sha256": REPETITION_SHA256,
        "executor": "native Limite with upstream components",
    }
