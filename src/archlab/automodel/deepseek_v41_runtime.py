"""Explicit, container-preserving V4.1 qualification helpers.

No installed module is edited or monkey-patched here. The compatible kernel
packages must already exist in a caller-supplied container package directory.
Import this before any TileLang/TVM imports in a fresh process. Import routing
is temporary, and the exact loaded source paths/versions are returned.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

REFERENCE_DIGESTS = {
    "model.py": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "engram.py": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "vision.py": "5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c",
    "image_processor.py": "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272",
}


def implementation_hashes() -> dict[str, str]:
    """Bind qualification receipts to the actual model/data/update implementation."""
    from archlab.artifacts import sha256_file

    root = Path(__file__).parents[1]
    owners = [
        "architectures/deepseek_v41_adapter.py", "architectures/deepseek_v41_math.py",
        "architectures/deepseek_v41_torch.py", "architectures/simplicial_attention.py",
        "architectures/simplicial_kernels.py", "optimizers/headwise_muon.py",
        *[f"automodel/deepseek_v41_{name}.py" for name in
          ("autograd", "data", "execution", "loading", "parallel", "pytorch", "loss", "training", "runtime", "native_quantization")],
    ]
    return {name: sha256_file(root / name) for name in owners}


def select_container_kernel_packages(package_root: Path) -> dict:
    root = package_root.resolve(strict=True)
    if any(name in sys.modules for name in ("tilelang", "tvm", "tvm_ffi")):
        raise RuntimeError("select kernel packages before importing TileLang/TVM in a fresh process")
    for name in ("tilelang", "tvm_ffi"):
        if not (root / name / "__init__.py").is_file():
            raise ValueError(f"container package missing: {name}")
    previous = sys.path[:]
    try:
        sys.path.insert(0, str(root))
        modules = {name: importlib.import_module(name) for name in ("tvm_ffi", "tilelang")}
    finally:
        sys.path[:] = previous
    evidence = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        if root not in path.parents:
            raise RuntimeError(f"unexpected {name} import: {path}")
        evidence[name] = {"version": module.__version__, "source": str(path),
                          "init_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if any(item["version"] != "0.1.9" for item in evidence.values()):
        raise RuntimeError(f"unaudited container kernel versions: {evidence}")
    return evidence


def load_native_reference(checkpoint: Path, *, module_name: str):
    """Import only the reviewed reference files, keeping independent model globals."""
    root = (checkpoint / "inference").resolve(strict=True)
    for filename, expected in REFERENCE_DIGESTS.items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"unreviewed reference source: {filename}")
    for name in ("kernel", "engram", "vision", "image_processor"):
        existing = sys.modules.get(name)
        if existing is not None and Path(existing.__file__).resolve() != (root / f"{name}.py").resolve(strict=True):
            raise RuntimeError(f"reference import would collide with module {name}")
    if module_name in sys.modules:
        raise RuntimeError("use a unique reference module name")
    previous = sys.path[:]
    spec = importlib.util.spec_from_file_location(module_name, root / "model.py")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.path.insert(0, str(root))
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path[:] = previous
    return module
